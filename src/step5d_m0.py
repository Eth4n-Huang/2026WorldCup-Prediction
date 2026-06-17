"""
step5d_m0.py — M0 度量口径修复与 AFCON 2015 诊断
1. 用 metrics.py 重算 dev 基准表（替换旧表）
2. 配对 bootstrap 比较 DC / XGB+int / Ens2
3. AFCON 2015 neutral 标记 + 东道主识别诊断
4. 生成 final_model_spec.md
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from metrics import (
    LABEL_ORDER, multiclass_brier, bla_probs_canonical,
    blb_probs_canonical, evaluate, paired_bootstrap_diff,
    pairwise_comparison_table,
)
from step4_train import TRAIN_START, get_feature_cols, fit_pipeline
from step5c_devset import (
    TOURNS, FIXED_HALF_LIFE,
    get_tournament_matches, add_interaction_features,
    dc_probs_for_matches, fit_dc_fast, metrics as old_metrics,
    bootstrap_ci,
)
from step4_train import tune_draw_threshold

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS   = Path(__file__).parent.parent / "outputs"

# WC+Euro+Copa 决胜子集（实际 342 场，用户标注 462 系估算）
SUBSET_WC_EU_COPA = {"WC2014","WC2018","Euro2016","Euro2020",
                     "Copa2015","Copa2016","Copa2019","Copa2021"}


# ══════════════════════════════════════════════
#  AFCON 2015 诊断
# ══════════════════════════════════════════════

def diagnose_afcon2015(df: pd.DataFrame):
    """
    诊断 AFCON 2015 BLa 22% 惨案根因：
    1. neutral 标记检查
    2. is_host 列（是否识别东道主赤道几内亚）
    3. BLa 逐场预测明细
    """
    print("\n" + "="*62)
    print("  AFCON 2015 诊断：neutral / is_host / BLa 预测")
    print("="*62)

    sub = df[(df["tournament"] == "African Cup of Nations") &
             (df["date"] >= "2015-01-17") & (df["date"] <= "2015-02-10")
             ].sort_values("date").copy()

    print(f"\n  AFCON 2015 match count: {len(sub)}")

    # ── neutral 标记 ──────────────────────────
    n_neutral_0 = (sub["neutral"] == 0).sum()
    n_neutral_1 = (sub["neutral"] == 1).sum()
    print(f"\n  neutral=0: {n_neutral_0}场  neutral=1: {n_neutral_1}场")
    if n_neutral_0 > 0:
        bad = sub[sub["neutral"] == 0][["date","home_team","away_team","neutral"]].head(5)
        print("  [警告] 有neutral=0的场次（应全为1）:")
        print(bad.to_string(index=False))
    else:
        print("  [OK] 所有AFCON 2015场次 neutral=1 ✓")

    # ── is_host ────────────────────────────────
    if "is_host" in sub.columns:
        ih = sub[sub["is_host"] == 1]
        print(f"\n  is_host=1 场次: {len(ih)}")
        if len(ih) > 0:
            print(ih[["date","home_team","away_team","is_host"]].to_string(index=False))
        else:
            print("  [注意] 赤道几内亚东道主未识别 is_host=1 (仅WC东道主被追踪)")
    else:
        print("\n  [注意] features.csv 无 is_host 列")

    # ── BLa 逐场明细 ──────────────────────────
    df_before = df[(df["date"] < "2015-01-17") & (df["date"] >= TRAIN_START)]
    draw_rate  = float((df_before["result"] == "D").mean())
    print(f"\n  训练期平局率: {draw_rate:.3f}")
    print(f"\n  BLa 逐场预测（前20场）:")
    hdr = f"  {'日期':>10}  {'主队':>25}  {'客队':>25}  "
    hdr += f"{'We':>6}  {'BLa预测':>6}  {'真实':>6}  {'correct':>7}"
    print(hdr); print("  " + "-"*95)
    correct, total = 0, 0
    draw_games, draw_correct = 0, 0
    result_counts = {"H":0,"D":0,"A":0}
    bla_pred_counts = {"H":0,"A":0}
    def _safe(s, n=25):
        return str(s).encode("ascii", errors="replace").decode("ascii")[:n]

    for _, r in sub.iterrows():
        h_eff = r["elo_home_pre"] + 100*(1-float(r["neutral"]))
        we    = 1/(1+10**(-( h_eff - r["elo_away_pre"])/400))
        pred  = "H" if we >= 0.5 else "A"
        ok    = (pred == r["result"])
        correct += ok; total += 1
        result_counts[r["result"]] += 1
        bla_pred_counts[pred] += 1
        if r["result"] == "D": draw_games += 1; draw_correct += (pred=="D")
        if total <= 20:
            ht = _safe(r["home_team"]); at = _safe(r["away_team"])
            flag = "OK" if ok else "X"
            print(f"  {str(r['date'].date()):>10}  {ht:>25}  {at:>25}  "
                  f"{we:>6.3f}  {pred:>6}  {r['result']:>6}  {flag:>7}")
    print(f"\n  真实结果分布: H={result_counts['H']}, D={result_counts['D']}, A={result_counts['A']}")
    print(f"  BLa预测分布: H={bla_pred_counts['H']}, A={bla_pred_counts['A']} (从不预测D)")
    print(f"  BLa 准确率: {correct}/{total} = {correct/total:.1%}")
    print(f"  Draw games: {draw_games}  Draw占比: {draw_games/total:.1%}")
    print(f"\n  [根因分析]:")
    draw_pct = draw_games/total
    if draw_pct > 0.30:
        print(f"    → 平局场次占{draw_pct:.0%}，BLa从不预测D，贡献{draw_games}个错误")
    if result_counts["A"] > result_counts["H"]:
        print(f"    → 客队（较弱Elo方）获胜场次({result_counts['A']})多于主队({result_counts['H']})")
        print(f"      说明该届存在大量upset，低Elo队伍表现远超预期")
    if n_neutral_0 > 0:
        print(f"    → [关键] neutral=0 场次存在({n_neutral_0}场)，")
        print(f"      BLa虚增'主队' Elo 100分，可能系统性预测错误")


# ══════════════════════════════════════════════
#  M0 重算 dev 表（使用 metrics.py）
# ══════════════════════════════════════════════

def run_m0_devset(df_raw, df_inter, feat_cols, feat_cols_inter, best_params):
    """
    对所有14个赛事重新运行完整回测，使用 metrics.py 统一指标。
    """
    print("\n" + "="*62)
    print("  M0: 使用 metrics.py 重算 dev 基准表")
    print("="*62)

    # 存储结果：{tourn_key: {model: {"probs": ..., "y_true": ...}}}
    all_res = {}

    for tourn_key, name_pat, d_start, d_end in TOURNS:
        t_matches = get_tournament_matches(df_raw, name_pat, d_start, d_end)
        if len(t_matches) < 16:
            print(f"  [{tourn_key}] 跳过（场次不足）")
            continue

        opening   = t_matches["date"].min()
        n         = len(t_matches)
        y_true    = t_matches["result"].values
        df_train  = df_raw[(df_raw["date"] >= TRAIN_START) &
                            (df_raw["date"] < opening)].copy()
        df_tr_i   = df_inter[(df_inter["date"] >= TRAIN_START) &
                              (df_inter["date"] < opening)].copy()

        print(f"\n  [{tourn_key}] n={n}  opening={opening.date()}", flush=True)

        res = {"y_true": y_true}

        # BLa (metrics.py canonical)
        res["BLa"] = bla_probs_canonical(t_matches, df_train)

        # BLb
        res["BLb"] = blb_probs_canonical(df_train, n)

        # XGB+int
        pkg = fit_pipeline(
            df_tr_i, "xgb", feat_cols=feat_cols_inter,
            xgb_max_depth=best_params["xgb"]["max_depth"],
            xgb_lr=best_params["xgb"]["lr"],
            xgb_n_est=best_params["xgb"]["n_est"],
            lam=best_params["xgb"]["lam"],
        )
        t_inter = add_interaction_features(t_matches)
        res["XGB+int"] = pkg["calibrated_model"].predict_proba(
            t_inter[feat_cols_inter].values.astype("float32"))
        delta_xgb  = pkg["delta"]
        dthr_xgb   = pkg["draw_thr"]

        # DC
        dc = fit_dc_fast(df_train, opening)
        res["DC"] = dc_probs_for_matches(dc, t_matches)

        n_val    = max(int(len(df_train)*0.15), 50)
        df_val_i = df_tr_i.iloc[-n_val:]
        y_val    = df_val_i["result"].values
        p_dc_val = dc_probs_for_matches(dc, df_val_i)
        p_xgb_val = pkg["calibrated_model"].predict_proba(
            df_val_i[feat_cols_inter].values.astype("float32"))
        draw_base = (df_train["result"] == "D").mean()

        # Ens2 w
        best_w, best_ll = 0.5, float("inf")
        for w in np.arange(0.0, 1.01, 0.1):
            p_e = w * p_xgb_val + (1-w) * p_dc_val
            ll_ = float(np.mean([
                -np.log(max(p_e[i, list(LABEL_ORDER).index(yt)], 1e-12))
                for i, yt in enumerate(y_val)
            ]))
            if ll_ < best_ll:
                best_ll, best_w = ll_, w
        res["Ens2"] = best_w * res["XGB+int"] + (1-best_w) * res["DC"]

        all_res[tourn_key] = res
        print(f"    DC_acc={evaluate(y_true,res['DC'])['accuracy']:.3f}  "
              f"XGB_acc={evaluate(y_true,res['XGB+int'])['accuracy']:.3f}  "
              f"Ens2_w={best_w:.1f}")

    return all_res


# ══════════════════════════════════════════════
#  配对比较报告
# ══════════════════════════════════════════════

def run_paired_comparison(all_res):
    """DC vs XGB+int vs Ens2 两两配对比较"""
    print("\n" + "="*62)
    print("  配对 Bootstrap 比较（全dev集，5000次）")
    print("="*62)

    # 合并全部场次
    keys = list(all_res.keys())
    y_all = np.concatenate([all_res[k]["y_true"] for k in keys])
    p_dict = {}
    for mname in ["DC", "XGB+int", "Ens2"]:
        p_dict[mname] = np.vstack([all_res[k][mname] for k in keys])

    # 整体指标
    print(f"\n  全dev集总体指标 (n={len(y_all)}):")
    print(f"  {'模型':<12} {'ACC':>7} {'Brier':>7} {'LL':>8}")
    print("  " + "-"*38)
    for mname in ["BLa","BLb","XGB+int","DC","Ens2"]:
        if mname in ["BLa","BLb"]:
            p = np.vstack([all_res[k][mname] for k in keys])
        else:
            p = p_dict.get(mname)
            if p is None: continue
        ev = evaluate(y_all, p)
        print(f"  {mname:<12} {ev['accuracy']:>7.4f} {ev['brier']:>7.4f} "
              f"{ev['log_loss']:>8.4f}")

    # 配对比较
    pairs = [("DC","XGB+int"), ("Ens2","XGB+int"), ("DC","Ens2")]
    for metric in ["log_loss", "accuracy"]:
        sign = "↓好" if metric == "log_loss" else "↑好"
        print(f"\n  配对Bootstrap: {metric} 差值(A-B)，{sign}")
        print(f"  {'A':>10} vs {'B':>10}  {'差值':>9}  {'CI':>20}  {'p':>7}  {'结论':>8}")
        print("  " + "-"*74)
        for n1, n2 in pairs:
            obs, ci, p = paired_bootstrap_diff(
                y_all, p_dict[n1], p_dict[n2],
                metric=metric, n_boot=5000)
            sig = "显著" if p < 0.05 else "不显著"
            print(f"  {n1:>10} vs {n2:>10}  {obs:>+9.5f}  "
                  f"[{ci[0]:+.4f},{ci[1]:+.4f}]  {p:>7.3f}  {sig:>8}")

    # WC+Euro+Copa 决胜子集
    keys_sub = [k for k in keys if k in SUBSET_WC_EU_COPA]
    y_sub    = np.concatenate([all_res[k]["y_true"] for k in keys_sub])
    n_sub    = len(y_sub)
    print(f"\n  WC+Euro+Copa 决胜子集 (n={n_sub}):")
    print(f"  {'模型':<12} {'ACC':>7} {'Brier':>7} {'LL':>8}")
    print("  " + "-"*38)
    for mname in ["XGB+int","DC","Ens2"]:
        p_s = np.vstack([all_res[k][mname] for k in keys_sub])
        ev  = evaluate(y_sub, p_s)
        print(f"  {mname:<12} {ev['accuracy']:>7.4f} {ev['brier']:>7.4f} "
              f"{ev['log_loss']:>8.4f}")

    return y_all, p_dict, y_sub, keys_sub


# ══════════════════════════════════════════════
#  final_model_spec.md
# ══════════════════════════════════════════════

SPEC_CONTENT = """# 最终模型选择规范 (final_model_spec.md)
版本: v1.0 (2026-06-12)

## Dev 集定义

| 项目 | 值 |
|---|---|
| 全dev集 | 593 场，14 个赛事（2014-2022，不含 2022 WC） |
| 2022 WC | **永久封存**，仅用于最终验证 |
| 详细清单 | outputs/devset_detail.csv |

赛事列表: WC2014/WC2018/Euro2016/Euro2020/Copa2015/Copa2016/Copa2019/Copa2021/
          AsianCup2015/AsianCup2019/AFCON2015/AFCON2017/AFCON2019/AFCON2021

## 统一指标模块
所有指标由 src/metrics.py 计算，定义不得更改。

| 指标 | 定义 |
|---|---|
| Brier | multiclass_brier(): mean( sum_c (p_c−y_c)² )，单场范围[0,2] |
| LogLoss | sklearn log_loss，labels=["A","D","H"] |
| BLa 概率 | bla_probs_canonical(): Elo We 拆分 + 历史平局率 |

## 模型选择准则

### 主准则
全 dev 集（593 场）配对 LogLoss bootstrap：
- 两模型差值 95% CI **全为负** → 较小 LogLoss 模型获胜
- CI 含 0 → 差异**不显著**，优先选结构更简单的模型

### 决胜准则（主准则平手时）
WC+Euro+Copa 子集（**342 场**，8 个赛事）的准确率与 LogLoss。
该子集与世界杯预测目标最相关。

注: 用户规格文件写 "462 场"，经统计实际 342 场（WC128+Euro102+Copa112）。

### 禁止操作
- 不得以 2022 WC 数据做任何选择
- 不得仅凭点估计（未配对检验）声称某模型"更优"
- 不得在 dev 集上调整 draw_adj 阈值后再宣称准确率提升

## 当前最优候选
（M0 完成后更新）
- 概率质量最优: DC（Brier/LogLoss 最低）
- 准确率最优: XGB+交互特征（WC/Euro 子集）
- 无统计显著差异: 见 paired bootstrap 结果

## 后续步骤
- 步骤3: Elo 超参优化 → 目标: 全dev集 LogLoss
- 步骤4: Ordered Logistic Regression
- 步骤5: 集成池扩展 (DC + XGB + OrderedLogit)
"""

def write_spec():
    path = OUTPUTS / "final_model_spec.md"
    path.write_text(SPEC_CONTENT, encoding="utf-8")
    print(f"\n  规范文档已写入: {path}")


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("读取数据...")
    df_raw   = pd.read_csv(PROCESSED / "features.csv", parse_dates=["date"])
    df_raw   = df_raw.sort_values("date").reset_index(drop=True)
    df_inter = add_interaction_features(df_raw)
    feat_cols       = get_feature_cols(df_raw)
    feat_cols_inter = feat_cols + ["elo_diff_ko", "rest_diff_ko"]

    with open(OUTPUTS / "train_params.json", encoding="utf-8") as f:
        best_params = json.load(f)

    # ── AFCON 2015 诊断 ─────────────────────
    diagnose_afcon2015(df_raw)

    # ── M0 重算 dev 表 ──────────────────────
    all_res = run_m0_devset(df_raw, df_inter, feat_cols, feat_cols_inter, best_params)

    # ── 配对比较 ─────────────────────────────
    y_all, p_dict, y_sub, keys_sub = run_paired_comparison(all_res)

    # ── 写规范文档 ────────────────────────────
    write_spec()

    # ── 保存明细 ──────────────────────────────
    detail_rows = []
    for key in all_res:
        y_ = all_res[key]["y_true"]
        for mname in ["BLa","BLb","XGB+int","DC","Ens2"]:
            if mname not in all_res[key]: continue
            p_ = all_res[key][mname]
            ev = evaluate(y_, p_)
            ci = bootstrap_ci(y_, p_)
            detail_rows.append({
                "tournament": key,
                "model": mname,
                "n": len(y_),
                **{k: round(v, 5) for k, v in ev.items()},
                "ci_lo": round(ci[0], 4),
                "ci_hi": round(ci[1], 4),
            })
    pd.DataFrame(detail_rows).to_csv(
        OUTPUTS / "devset_m0_detail.csv", index=False, encoding="utf-8-sig")
    print("\n明细已保存: outputs/devset_m0_detail.csv")
    print("\n[M0 完成] dev 基准表已建立，可进入步骤3(Elo优化)")
