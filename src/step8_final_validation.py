"""
step8_final_validation.py — 最终冻结 + 2022解封验证
==========================================================
改进期已结束（用户宣布）。本脚本执行:
  1. 写入冻结版 final_model_spec.md
  2. 按冻结规格重跑 2014 / 2018 / 2022 三届
  3. 打印所有要求的汇总表（不做任何建议，只报告结果）

冻结规格:
  主模型 : DC 双泊松 (ρ低比分修正, half_life=730d, L-BFGS-B)
           决策口径 = argmax（无 δ 调整）
  对照模型: XGB + 交互特征 (H=125/K_major=40)
           双口径: argmax + draw_adj (δ在训练期选)
  基线a  : BLa — Elo 软概率 We拆分, H_adv=125
  基线b  : BLb — 历史频率
  所有指标: metrics.py (Brier = sum-over-3-classes, [0,2])
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from step4_train import (
    TRAIN_START, get_feature_cols, fit_pipeline,
    predict_with_draw_adj, tune_draw_threshold,
)
from step5b_improve import DixonColesModel
from step5c_devset import (
    add_interaction_features, fit_dc_fast,
    dc_probs_for_matches,
)
from metrics import (
    LABEL_ORDER, multiclass_brier,
    bla_probs_canonical, blb_probs_canonical,
    evaluate, paired_bootstrap_diff,
)

PROC   = Path(__file__).parent.parent / "data" / "processed"
RAW    = Path(__file__).parent.parent / "data" / "raw"
OUT    = Path(__file__).parent.parent / "outputs"

WC_OPENING = {
    2014: pd.Timestamp("2014-06-12"),
    2018: pd.Timestamp("2018-06-14"),
    2022: pd.Timestamp("2022-11-20"),
}
H_ADV_FROZEN = 125.0   # 冻结: 新Elo H=125


# ══════════════════════════════════════════════
#  0. 冻结 final_model_spec.md
# ══════════════════════════════════════════════

FROZEN_SPEC = """\
# 最终模型规格 (冻结版)
版本: v2.0-FROZEN (2026-06-12)

> 本文件在 2022届解封前冻结，2022结果不再触发任何改动。

## 主模型：DC 双泊松 (Dixon-Coles 1997)

| 超参 | 值 |
|------|-----|
| half_life | 730 天（时间衰减半衰期）|
| ρ (低比分修正) | MLE 学习（L-BFGS-B），约束 [-0.4, 0.4] |
| log_gamma (主场) | MLE 学习，约束 [-0.7, 0.7] |
| 正则化 | L2=0.01 on log_a / log_d |
| max_goals | 10（比分积分截断）|
| 决策口径 | argmax（无 δ 调整；dev 证据:δ 调整使 DC 准确率显著下降） |

## 对照模型：XGB + 交互特征

| 超参 | 值 |
|------|-----|
| max_depth | 3 |
| learning_rate | 0.05 |
| n_estimators | 100 |
| time_decay λ | 0（从 train_params.json）|
| 校准 | IsotonicRegression（逐类别）|
| Elo 配置 | H_adv=125, K_major=40（新Elo配置）|
| 交互特征 | elo_diff×is_knockout, rest_diff×is_knockout |
| 决策 | argmax + draw_adj（δ 在每届训练期内选取，不接触测试届）|

## 基线

**BLa — Elo 软概率**
- We = 1/(1+10^(-(elo_h_eff - elo_away_pre)/400))
- elo_h_eff = elo_home_pre + 125*(1−neutral)   ← H_adv=125（冻结）
- draw_rate = 训练期平局占比
- P(H)=We*(1-draw_rate), P(D)=draw_rate, P(A)=(1-We)*(1-draw_rate)

**BLb — 历史频率**
- 所有比赛赋训练期 H/D/A 历史频率

## Elo 配置（已冻结）

H_adv=125, K_wc=60, K_major=40, K_qual=40, K_friendly=20, G=original
- 选参依据: 1998-2014-06-11 训练期 BLa LogLoss（ΔLL=-0.00207）
- dev 配对检验: 方向一致但未达显著（全dev p=0.91, WC+Copa p=0.18）
- 采纳规则: CI跨零时默认采用训练期最优（预注册规则）

## 指标定义（metrics.py）

| 指标 | 定义 |
|------|------|
| Brier | mean(sum_c (p_c−y_c)²)，单场[0,2] |
| LogLoss | sklearn log_loss，labels=["A","D","H"] |
| macro-F1 | sklearn f1_score(average="macro", zero_division=0) |
| 口径A | 90分钟胜平负三分类 |
| 口径B | 淘汰赛晋级二分类: P(home)=P(H)+0.5*P(D) |

## 合规声明

本规格在解封 2022 世界杯前完成冻结。
2022 回测结果纯作验证，不触发任何参数改动。
"""


def write_frozen_spec():
    path = OUT / "final_model_spec.md"
    path.write_text(FROZEN_SPEC, encoding="utf-8")
    print("[冻结] final_model_spec.md 已写入冻结版 v2.0-FROZEN")


# ══════════════════════════════════════════════
#  1. 数据加载
# ══════════════════════════════════════════════

def load_data():
    df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
    df = add_interaction_features(df.sort_values("date").reset_index(drop=True))
    shootouts = pd.read_csv(RAW / "shootouts.csv", parse_dates=["date"])
    with open(OUT / "train_params.json") as f:
        tp = json.load(f)
    return df, shootouts, tp


# ══════════════════════════════════════════════
#  2. 口径B: 淘汰赛真实晋级结果
# ══════════════════════════════════════════════

def get_true_advance(row, shootouts):
    """
    淘汰赛真实晋级方（home/away）。
    - 90min H/A → 直接得出
    - 90min D   → 查 shootouts.csv winner
    """
    if row["result"] == "H":
        return "home"
    if row["result"] == "A":
        return "away"
    # 平局 → 点球
    mask = (
        (shootouts["date"] == row["date"]) &
        (shootouts["home_team"] == row["home_team"]) &
        (shootouts["away_team"] == row["away_team"])
    )
    found = shootouts[mask]
    if len(found) == 0:
        # 用年份松散匹配
        mask2 = (
            (shootouts["date"].dt.year == row["date"].year) &
            (shootouts["home_team"] == row["home_team"]) &
            (shootouts["away_team"] == row["away_team"])
        )
        found = shootouts[mask2]
    if len(found) == 0:
        return None
    winner = found.iloc[0]["winner"]
    if winner == row["home_team"]:
        return "home"
    if winner == row["away_team"]:
        return "away"
    return None


# ══════════════════════════════════════════════
#  3. 阶段标签
# ══════════════════════════════════════════════

def get_stage_label(row):
    if row.get("stage_group_r1", 0) == 1:
        return "R1"
    if row.get("stage_group_r2", 0) == 1:
        return "R2"
    if row.get("stage_group_r3", 0) == 1:
        return "R3"
    if row.get("stage_knockout", 0) == 1:
        return "KO"
    return "OTHER"


# ══════════════════════════════════════════════
#  4. 单届回测
# ══════════════════════════════════════════════

def run_one_year(year, df, shootouts, tp):
    print(f"\n{'='*60}")
    print(f"  {year} 届回测（冻结规格）")
    print(f"{'='*60}")

    opening  = WC_OPENING[year]
    df_train = df[(df["date"] >= pd.Timestamp(TRAIN_START)) &
                  (df["date"] < opening)].copy()
    df_wc    = (df[(df["tournament"] == "FIFA World Cup") &
                   (df["date"].dt.year == year)]
                .sort_values("date").copy())

    print(f"  训练集: {len(df_train)} 场  |  本届WC: {len(df_wc)} 场")

    feat_cols = get_feature_cols(df)
    feat_inter = feat_cols + ["elo_diff_ko", "rest_diff_ko"]
    y_true = df_wc["result"].values

    # ── DC ─────────────────────────────────────
    print("  拟合 DC 模型 (half_life=730)...", end="", flush=True)
    dc = fit_dc_fast(df_train, opening)
    dc_probs = dc_probs_for_matches(dc, df_wc)
    print(" 完成")

    # ── XGB+int ────────────────────────────────
    print("  拟合 XGB+交互特征...", end="", flush=True)
    xgb_pkg = fit_pipeline(
        df_train, "xgb",
        feat_cols=feat_inter,
        xgb_max_depth=tp["xgb"]["max_depth"],
        xgb_lr=tp["xgb"]["lr"],
        xgb_n_est=tp["xgb"]["n_est"],
        lam=tp["xgb"]["lam"],
    )
    X_wc  = df_wc[feat_inter].values.astype("float32")
    xgb_probs = xgb_pkg["calibrated_model"].predict_proba(X_wc)
    xgb_delta = xgb_pkg["delta"]
    xgb_thr   = xgb_pkg["draw_thr"]
    print(f" 完成  δ={xgb_delta:.2f}  thr={xgb_thr:.2f}")

    # ── 基线 ───────────────────────────────────
    bla_probs = bla_probs_canonical(df_wc, df_train, h_adv=H_ADV_FROZEN)
    blb_probs = blb_probs_canonical(df_train, len(df_wc))

    # ── 口径B 晋级 ─────────────────────────────
    ko_mask = df_wc["stage_knockout"].values == 1
    df_ko   = df_wc[ko_mask].copy()
    df_ko["true_advance"] = df_ko.apply(
        lambda r: get_true_advance(r, shootouts), axis=1)

    # ══════════════════════════════════════════
    #  指标汇总
    # ══════════════════════════════════════════

    lo = LABEL_ORDER
    draw_base = (df_train["result"] == "D").mean()

    def argmax_preds(probs):
        return np.array([lo[i] for i in np.argmax(probs, axis=1)])

    def adv_preds(probs):
        ph = probs[:, lo.index("H")]
        pd_ = probs[:, lo.index("D")]
        p_home = ph + 0.5 * pd_
        return np.where(p_home > 0.5, "home", "away")

    def adv_acc(probs, true_adv):
        pred = adv_preds(probs)
        valid = np.array(true_adv) != None
        if valid.sum() == 0:
            return float("nan")
        return float(np.mean(pred[valid] == np.array(true_adv)[valid]))

    # ── 总体指标 ───────────────────────────────
    results = {}
    for name, probs in [("DC", dc_probs), ("XGB+int", xgb_probs),
                         ("BLa", bla_probs), ("BLb", blb_probs)]:
        m = evaluate(y_true, probs)
        if name == "XGB+int":
            preds_adj = predict_with_draw_adj(probs, xgb_delta, xgb_thr, lo)
            m["accuracy_adj"] = float(accuracy_score(y_true, preds_adj))
        results[name] = m

    # ── 分阶段 ─────────────────────────────────
    stages = {}
    df_wc["_stage"] = df_wc.apply(get_stage_label, axis=1)
    for stage in ["R1", "R2", "R3", "KO"]:
        mask_s = df_wc["_stage"].values == stage
        if mask_s.sum() == 0:
            continue
        yt_s = y_true[mask_s]
        stage_results = {}
        for name, probs in [("DC", dc_probs), ("XGB+int", xgb_probs),
                              ("BLa", bla_probs)]:
            probs_s = probs[mask_s]
            preds_s = argmax_preds(probs_s)
            acc_s   = float(accuracy_score(yt_s, preds_s))
            brier_s = multiclass_brier(yt_s, probs_s)
            stage_results[name] = {"acc": acc_s, "brier": brier_s}
        stages[stage] = stage_results

    # ── 淘汰赛口径B ────────────────────────────
    ko_true_adv = df_ko["true_advance"].values
    ko_dc   = dc_probs[ko_mask]
    ko_xgb  = xgb_probs[ko_mask]
    ko_bla  = bla_probs[ko_mask]
    ko_acc_dc  = adv_acc(ko_dc,  ko_true_adv)
    ko_acc_xgb = adv_acc(ko_xgb, ko_true_adv)
    ko_acc_bla = adv_acc(ko_bla, ko_true_adv)

    # ── 平局诊断 ───────────────────────────────
    actual_d   = int((y_true == "D").sum())
    dc_pred_d  = int((argmax_preds(dc_probs) == "D").sum())
    dc_hit_d   = int(((argmax_preds(dc_probs) == "D") & (y_true == "D")).sum())
    xgb_pred_d_adj = int((predict_with_draw_adj(
        xgb_probs, xgb_delta, xgb_thr, lo) == "D").sum())

    # ── 最自信5个错误 (DC) ──────────────────────
    dc_argmax   = argmax_preds(dc_probs)
    dc_conf     = dc_probs.max(axis=1)
    wrong_mask  = dc_argmax != y_true
    wrong_idx   = np.where(wrong_mask)[0]
    wrong_conf  = dc_conf[wrong_idx]
    top5_idx    = wrong_idx[np.argsort(-wrong_conf)[:5]]

    # ── 配对 Bootstrap ────────────────────────
    bs_dc_bla_ll  = paired_bootstrap_diff(y_true, dc_probs, bla_probs,
                                           metric="log_loss", n_boot=10_000)
    bs_dc_bla_acc = paired_bootstrap_diff(y_true, dc_probs, bla_probs,
                                           metric="accuracy", n_boot=10_000)
    bs_dc_xgb_ll  = paired_bootstrap_diff(y_true, dc_probs, xgb_probs,
                                           metric="log_loss", n_boot=10_000)
    bs_dc_xgb_acc = paired_bootstrap_diff(y_true, dc_probs, xgb_probs,
                                           metric="accuracy", n_boot=10_000)

    # ══════════════════════════════════════════
    #  打印结果
    # ══════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  {year} | 总体指标 (64场)")
    print(f"{'─'*60}")
    header = f"  {'模型':<12} {'ACC':>7} {'ACC_adj':>9} {'MacF1':>7} {'Brier':>7} {'LogLoss':>9}"
    print(header)
    print(f"  {'-'*62}")
    for nm in ["DC", "XGB+int", "BLa", "BLb"]:
        m = results[nm]
        acc_adj = m.get("accuracy_adj", m["accuracy"])
        print(f"  {nm:<12} {m['accuracy']:>7.4f} {acc_adj:>9.4f} "
              f"{m['macro_f1']:>7.4f} {m['brier']:>7.4f} {m['log_loss']:>9.5f}")

    print(f"\n  {year} | 分阶段准确率 (DC / XGB+int / BLa)")
    print(f"  {'阶段':<6} {'场数':>5} {'DC_acc':>8} {'XGB_acc':>9} {'BLa_acc':>9}")
    stage_names = {"R1": "小组R1", "R2": "小组R2", "R3": "小组R3", "KO": "淘汰赛A"}
    for stg in ["R1", "R2", "R3", "KO"]:
        if stg not in stages:
            continue
        mask_s = df_wc["_stage"].values == stg
        n_s  = mask_s.sum()
        s    = stages[stg]
        print(f"  {stage_names[stg]:<6} {n_s:>5}"
              f" {s['DC']['acc']:>8.4f} {s['XGB+int']['acc']:>9.4f}"
              f" {s['BLa']['acc']:>9.4f}")

    n_ko_valid = int(np.sum(np.array(ko_true_adv) != None))
    print(f"\n  {year} | 淘汰赛口径B (晋级预测, {n_ko_valid}场有效)")
    print(f"  DC晋级ACC={ko_acc_dc:.4f}  XGB晋级ACC={ko_acc_xgb:.4f}  "
          f"BLa晋级ACC={ko_bla:.4f}" if False else
          f"  DC晋级ACC={ko_acc_dc:.4f}  XGB晋级ACC={ko_acc_xgb:.4f}  "
          f"BLa晋级ACC={ko_acc_bla:.4f}")

    print(f"\n  {year} | 平局诊断")
    print(f"  实际平局: {actual_d}场  DC预测D: {dc_pred_d}场 (命中{dc_hit_d})  "
          f"XGB draw_adj预测D: {xgb_pred_d_adj}场")
    d_recall_dc  = dc_hit_d / max(1, actual_d)
    print(f"  DC平局召回率: {d_recall_dc:.3f}")

    print(f"\n  {year} | 配对Bootstrap (DC-BLa,  DC-XGB;  LL: 负值=DC更好)")
    _fmt_bs = lambda obs, ci, p: f"Δ={obs:+.5f} 95%CI=[{ci[0]:+.5f},{ci[1]:+.5f}] p={p:.3f}"
    print(f"  DC vs BLa  LL : {_fmt_bs(*bs_dc_bla_ll)}")
    print(f"  DC vs BLa  ACC: {_fmt_bs(*bs_dc_bla_acc)}")
    print(f"  DC vs XGB  LL : {_fmt_bs(*bs_dc_xgb_ll)}")
    print(f"  DC vs XGB  ACC: {_fmt_bs(*bs_dc_xgb_acc)}")

    print(f"\n  {year} | 最自信的5个错误 (DC)")
    print(f"  {'日期':<12} {'主队':<22} {'客队':<22} {'预测':>4} {'实际':>4} {'最高概率':>8}")
    for i in top5_idx:
        row = df_wc.iloc[i]
        print(f"  {str(row['date'].date()):<12} "
              f"{str(row['home_team'])[:20]:<22} "
              f"{str(row['away_team'])[:20]:<22} "
              f"{dc_argmax[i]:>4} {y_true[i]:>4} "
              f"{dc_conf[i]:>8.4f}")

    # ── 保存明细 CSV ────────────────────────────
    df_out = df_wc[["date", "home_team", "away_team", "result",
                    "elo_home_pre", "elo_away_pre", "neutral"]].copy()
    df_out["_stage"]    = df_wc["_stage"].values
    df_out["dc_pa"]     = dc_probs[:, 0]
    df_out["dc_pd"]     = dc_probs[:, 1]
    df_out["dc_ph"]     = dc_probs[:, 2]
    df_out["xgb_pa"]    = xgb_probs[:, 0]
    df_out["xgb_pd"]    = xgb_probs[:, 1]
    df_out["xgb_ph"]    = xgb_probs[:, 2]
    df_out["bla_pa"]    = bla_probs[:, 0]
    df_out["bla_pd"]    = bla_probs[:, 1]
    df_out["bla_ph"]    = bla_probs[:, 2]
    df_out["dc_argmax"] = dc_argmax
    df_out["xgb_argmax"]  = argmax_preds(xgb_probs)
    df_out["xgb_draw_adj"]= predict_with_draw_adj(xgb_probs, xgb_delta, xgb_thr, lo)
    df_out["dc_correct"]  = (df_out["dc_argmax"] == df_out["result"]).astype(int)
    df_out["xgb_correct"] = (df_out["xgb_argmax"] == df_out["result"]).astype(int)
    df_out["true_advance"] = ""
    for ii in range(len(df_wc)):
        if ko_mask[ii]:
            ta = get_true_advance(df_wc.iloc[ii], shootouts)
            df_out.iloc[ii, df_out.columns.get_loc("true_advance")] = str(ta) if ta else ""

    csv_path = OUT / f"backtest_final_{year}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  已保存: {csv_path.name}")

    return {
        "year": year,
        "results": results,
        "stages": stages,
        "ko_adv": {"DC": ko_acc_dc, "XGB": ko_acc_xgb, "BLa": ko_acc_bla},
        "draw_diag": {"actual": actual_d, "dc_pred": dc_pred_d, "dc_hit": dc_hit_d},
        "bootstrap": {
            "dc_bla_ll":  bs_dc_bla_ll,
            "dc_bla_acc": bs_dc_bla_acc,
            "dc_xgb_ll":  bs_dc_xgb_ll,
            "dc_xgb_acc": bs_dc_xgb_acc,
        },
    }


# ══════════════════════════════════════════════
#  5. 三届汇总表
# ══════════════════════════════════════════════

def print_three_year_summary(all_results):
    print(f"\n\n{'='*65}")
    print("  三届汇总表（冻结规格，同口径可比）")
    print(f"{'='*65}")

    years  = [r["year"] for r in all_results]
    models = ["DC", "XGB+int", "BLa", "BLb"]

    # 总体 ACC
    print(f"\n【总体准确率 ACC — 口径A(90分钟)】")
    print(f"  {'模型':<12} ", end="")
    for yr in years:
        print(f" {yr}", end="")
    print("   均值")
    print(f"  {'-'*50}")
    for nm in models:
        print(f"  {nm:<12} ", end="")
        vals = []
        for r in all_results:
            v = r["results"][nm]["accuracy"]
            vals.append(v)
            print(f" {v:.4f}", end="")
        print(f"  {np.mean(vals):.4f}")

    # XGB draw_adj ACC
    print(f"\n【XGB draw_adj 准确率】")
    print(f"  {'模型':<12} ", end="")
    for yr in years:
        print(f"   {yr}", end="")
    print("    均值")
    print(f"  {'-'*50}")
    for nm in ["XGB+int"]:
        print(f"  {nm}(adj)  ", end="")
        vals = []
        for r in all_results:
            v = r["results"][nm].get("accuracy_adj", r["results"][nm]["accuracy"])
            vals.append(v)
            print(f"  {v:.4f}", end="")
        print(f"   {np.mean(vals):.4f}")

    # LogLoss
    print(f"\n【LogLoss（越小越好）】")
    print(f"  {'模型':<12} ", end="")
    for yr in years:
        print(f"      {yr}", end="")
    print("     均值")
    print(f"  {'-'*55}")
    for nm in models:
        print(f"  {nm:<12} ", end="")
        vals = []
        for r in all_results:
            v = r["results"][nm]["log_loss"]
            vals.append(v)
            print(f"   {v:.5f}", end="")
        print(f"   {np.mean(vals):.5f}")

    # Brier
    print(f"\n【Brier Score（越小越好，单场范围[0,2]）】")
    print(f"  {'模型':<12} ", end="")
    for yr in years:
        print(f"   {yr}", end="")
    print("    均值")
    print(f"  {'-'*50}")
    for nm in models:
        print(f"  {nm:<12} ", end="")
        vals = []
        for r in all_results:
            v = r["results"][nm]["brier"]
            vals.append(v)
            print(f"  {v:.5f}", end="")
        print(f"   {np.mean(vals):.5f}")

    # 淘汰赛口径B
    print(f"\n【淘汰赛晋级预测准确率 — 口径B】")
    print(f"  {'模型':<10} ", end="")
    for yr in years:
        print(f"   {yr}", end="")
    print("    均值")
    print(f"  {'-'*47}")
    for nm in ["DC", "XGB", "BLa"]:
        print(f"  {nm:<10} ", end="")
        vals = []
        for r in all_results:
            v = r["ko_adv"].get(nm, float("nan"))
            vals.append(v)
            print(f"  {v:.4f}", end="")
        valid = [v for v in vals if not np.isnan(v)]
        print(f"   {np.mean(valid):.4f}" if valid else "")

    # 分阶段 DC
    print(f"\n【DC 分阶段准确率（口径A）】")
    stage_names = {"R1": "小组R1", "R2": "小组R2", "R3": "小组R3", "KO": "淘汰赛"}
    for stg in ["R1", "R2", "R3", "KO"]:
        print(f"  {stage_names[stg]:<6} ", end="")
        vals = []
        for r in all_results:
            v = r["stages"].get(stg, {}).get("DC", {}).get("acc", float("nan"))
            vals.append(v)
            print(f"  {v:.4f}", end="")
        valid = [v for v in vals if not np.isnan(v)]
        print(f"   均值:{np.mean(valid):.4f}" if valid else "")

    # 平局诊断汇总
    print(f"\n【平局诊断汇总】")
    print(f"  {'届次':<6} {'实际D':>7} {'DC预测D':>9} {'DC命中':>8} {'DC召回':>8}")
    for r in all_results:
        d = r["draw_diag"]
        recall = d["dc_hit"] / max(1, d["actual"])
        print(f"  {r['year']:<6} {d['actual']:>7} {d['dc_pred']:>9} "
              f"{d['dc_hit']:>8} {recall:>8.3f}")

    # Bootstrap 显著性
    print(f"\n【配对Bootstrap显著性 (DC-BLa LL, 10000次)】")
    print(f"  {'届次':<6} {'Δ(DC-BLa)':>12} {'95%CI':>28} {'p值':>7}")
    for r in all_results:
        obs, ci, p = r["bootstrap"]["dc_bla_ll"]
        sig = "*" if p < 0.05 else ""
        print(f"  {r['year']:<6} {obs:>12.5f} "
              f"[{ci[0]:+.5f},{ci[1]:+.5f}]{sig:>2} {p:>7.3f}")

    print(f"\n【配对Bootstrap显著性 (DC-XGB LL, 10000次)】")
    print(f"  {'届次':<6} {'Δ(DC-XGB)':>12} {'95%CI':>28} {'p值':>7}")
    for r in all_results:
        obs, ci, p = r["bootstrap"]["dc_xgb_ll"]
        sig = "*" if p < 0.05 else ""
        print(f"  {r['year']:<6} {obs:>12.5f} "
              f"[{ci[0]:+.5f},{ci[1]:+.5f}]{sig:>2} {p:>7.3f}")


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

def main():
    print("="*65)
    print("  step8_final_validation.py — 最终冻结 + 三届验证")
    print("="*65)

    # 1. 写冻结规范
    write_frozen_spec()

    # 2. 加载数据
    df, shootouts, tp = load_data()

    # 3. 三届回测
    all_results = []
    for year in [2014, 2018, 2022]:
        r = run_one_year(year, df, shootouts, tp)
        all_results.append(r)

    # 4. 汇总表
    print_three_year_summary(all_results)

    print(f"\n{'='*65}")
    print("  全部输出完毕。结果已冻结，不再修改。")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
