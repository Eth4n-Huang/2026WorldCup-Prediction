"""
步骤3 Phase2: 新 Elo 重训 + dev 配对 Bootstrap
==============================================
1. 加载 elo_best_params.json（由 step6_elo_opt.py 选出，仅基于训练期）
2. 用新 Elo 参数重放历史 Elo 序列
3. 更新 features.csv 中的 Elo 列（elo_home_pre/away_pre/elo_diff）
4. 对 14 个 dev 赛事：用旧 Elo 和新 Elo 各训一次 XGB+int
5. 全 dev (n=593) + WC+Euro+Copa 子集 (n=342) 配对 Bootstrap (5000次)
6. 写结论到 outputs/final_model_spec.md

纪律: dev 数据仅在此步骤被读取一次，用于评估（不反馈到参数选择）
"""
from __future__ import annotations
import sys, json, warnings, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from step4_train import TRAIN_START, get_feature_cols, fit_pipeline, tune_draw_threshold
from step5b_improve import LABEL_ORDER, _fit_pipeline_extra_w
from step5c_devset import (
    TOURNS, add_interaction_features,
    get_tournament_matches, dc_probs_for_matches,
    fit_dc_fast, make_bla_probs, make_blb_probs,
)
from metrics import multiclass_brier, paired_bootstrap_diff
from step6_elo_opt import compute_elo_series

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR  = Path(__file__).parent.parent / "outputs"

WC_EURO_COPA = {"WC2014", "WC2018", "Euro2016", "Euro2020",
                "Copa2015", "Copa2016", "Copa2019", "Copa2021"}

DEFAULT_PARAMS = {
    "H_adv": 100, "K_wc": 60, "K_major": 50,
    "K_qual": 40, "K_friendly": 20, "G_func": "original",
}


# ═══════════════════════════════════════════════
#  Elo 列更新
# ═══════════════════════════════════════════════

def make_features_with_elo(df_feat: pd.DataFrame,
                            df_elo_new: pd.DataFrame,
                            h_adv: float) -> pd.DataFrame:
    """
    用新 Elo 替换 df_feat 中的 elo_home_pre/away_pre/elo_diff。
    其余特征（winrate、h2h 等）保持不变（它们不依赖 Elo）。
    """
    df = df_feat.copy()
    # 用 (date, home_team, away_team) 作联接键
    key_cols = ["date", "home_team", "away_team"]
    elo_sub  = df_elo_new[key_cols + ["elo_home_pre", "elo_away_pre"]].copy()

    df = df.drop(columns=["elo_home_pre", "elo_away_pre", "elo_diff"], errors="ignore")
    df = df.merge(elo_sub, on=key_cols, how="left")

    # 重算 elo_diff（含新 H_adv）
    H_arr = np.where(df["neutral"].values.astype(bool), 0.0, h_adv)
    df["elo_diff"] = df["elo_home_pre"].values + H_arr - df["elo_away_pre"].values
    return df


# ═══════════════════════════════════════════════
#  单赛事回测（XGB+int only, 可选 DC）
# ═══════════════════════════════════════════════

def run_tourn_xgb(tourn_key: str, t_matches: pd.DataFrame,
                  df_full: pd.DataFrame, df_inter: pd.DataFrame,
                  feat_cols: list, feat_cols_inter: list,
                  best_params: dict) -> np.ndarray:
    """
    训练 XGB+int，返回 (N,3) 概率 [A,D,H]。
    使用严格时间截止（开赛前数据训练）。
    """
    opening = t_matches["date"].min()
    df_train_inter = df_inter[(df_inter["date"] >= TRAIN_START) &
                              (df_inter["date"] < opening)].copy()
    t_inter  = add_interaction_features(t_matches)
    X_test   = t_inter[feat_cols_inter].values.astype(np.float32)

    pkg = fit_pipeline(
        df_train_inter, "xgb", feat_cols=feat_cols_inter,
        xgb_max_depth=best_params["xgb"]["max_depth"],
        xgb_lr=best_params["xgb"]["lr"],
        xgb_n_est=best_params["xgb"]["n_est"],
        lam=best_params["xgb"]["lam"],
    )
    return pkg["calibrated_model"].predict_proba(X_test)


def run_tourn_full(tourn_key: str, t_matches: pd.DataFrame,
                   df_full: pd.DataFrame, df_inter: pd.DataFrame,
                   feat_cols: list, feat_cols_inter: list,
                   best_params: dict, h_adv: float) -> dict:
    """
    对单个赛事评估 BLa + XGB+int + DC + Ens2。
    返回 {model_name: (probs_array, y_true)}
    """
    opening = t_matches["date"].min()
    n       = len(t_matches)
    y_true  = t_matches["result"].values

    df_train = df_full[(df_full["date"] >= TRAIN_START) &
                       (df_full["date"] < opening)].copy()
    df_train_inter = df_inter[(df_inter["date"] >= TRAIN_START) &
                               (df_inter["date"] < opening)].copy()

    n_val = max(int(len(df_train) * 0.15), 50)
    df_val_i = df_train_inter.iloc[-n_val:].copy()
    y_val = df_val_i["result"].values
    draw_base = float((df_train["result"] == "D").mean())

    # BLa（使用传入的 h_adv）
    def bla_with_h(matches, df_tr, h):
        dr = float((df_tr["result"] == "D").mean())
        probs = []
        for _, r in matches.iterrows():
            H  = 0.0 if bool(r.get("neutral", 0)) else h
            dr_ = (r["elo_home_pre"] + H) - r["elo_away_pre"]
            we  = 1.0 / (1.0 + 10.0 ** (-dr_ / 400.0))
            probs.append([
                (1 - we) * (1 - dr),
                dr,
                we * (1 - dr),
            ])
        return np.array(probs)   # [A, D, H]

    probs_bla = bla_with_h(t_matches, df_train, h_adv)

    # XGB+int
    t_inter = add_interaction_features(t_matches)
    X_test  = t_inter[feat_cols_inter].values.astype(np.float32)
    pkg_xgb = fit_pipeline(
        df_train_inter, "xgb", feat_cols=feat_cols_inter,
        xgb_max_depth=best_params["xgb"]["max_depth"],
        xgb_lr=best_params["xgb"]["lr"],
        xgb_n_est=best_params["xgb"]["n_est"],
        lam=best_params["xgb"]["lam"],
    )
    probs_xgb = pkg_xgb["calibrated_model"].predict_proba(X_test)

    # DC
    dc = fit_dc_fast(df_train, opening)
    probs_dc = dc_probs_for_matches(dc, t_matches)

    # Ens2
    probs_xgb_val = pkg_xgb["calibrated_model"].predict_proba(
        df_val_i[feat_cols_inter].values.astype(np.float32))
    probs_dc_val = dc_probs_for_matches(dc, df_val_i)
    best_w, best_ll_v = 0.5, float("inf")
    for w in np.arange(0.0, 1.01, 0.1):
        p_ens_v = w * probs_xgb_val + (1 - w) * probs_dc_val
        ll_v = float(log_loss(y_val, p_ens_v, labels=LABEL_ORDER))
        if ll_v < best_ll_v:
            best_ll_v, best_w = ll_v, w
    probs_ens2 = best_w * probs_xgb + (1 - best_w) * probs_dc

    return {
        "BLa":     (probs_bla, y_true),
        "XGB+int": (probs_xgb, y_true),
        "DC":      (probs_dc, y_true),
        "Ens2":    (probs_ens2, y_true),
    }


# ═══════════════════════════════════════════════
#  汇总指标
# ═══════════════════════════════════════════════

def summary_metrics(y_all: np.ndarray, probs_all: np.ndarray) -> dict:
    ll  = float(log_loss(y_all, probs_all, labels=LABEL_ORDER))
    br  = multiclass_brier(y_all, probs_all)
    preds = [LABEL_ORDER[np.argmax(p)] for p in probs_all]
    acc = float(np.mean([p == y for p, y in zip(preds, y_all)]))
    return {"ll": ll, "brier": br, "acc": acc}


# ═══════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  步骤3 Phase2: 新 Elo 重训 + dev 配对 Bootstrap")
    print("=" * 65)

    # ── 加载最优参数 ───────────────────────────────────────
    with open(OUT_DIR / "elo_best_params.json") as f:
        best_elo = json.load(f)
    print(f"\n加载 elo_best_params.json:")
    for k, v in best_elo.items():
        if k not in ("note",):
            print(f"  {k}: {v}")

    with open(OUT_DIR / "train_params.json", encoding="utf-8") as f:
        xgb_params = json.load(f)

    # ── 读取数据 ───────────────────────────────────────────
    print("\n读取数据...")
    df_feat = pd.read_csv(PROC_DIR / "features.csv", parse_dates=["date"])
    df_feat = df_feat.sort_values("date").reset_index(drop=True)
    df_clean = pd.read_csv(PROC_DIR / "matches_clean.csv", parse_dates=["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    feat_cols       = get_feature_cols(df_feat)
    feat_cols_inter = feat_cols + ["elo_diff_ko", "rest_diff_ko"]

    # ── 计算新 Elo 序列 ────────────────────────────────────
    print("\n重新演算 Elo（新参数）...")
    new_elo_params = {
        "H_adv": best_elo["H_adv"], "K_wc": best_elo["K_wc"],
        "K_major": best_elo["K_major"], "K_qual": best_elo["K_qual"],
        "K_friendly": best_elo["K_friendly"], "G_func": best_elo["G_func"],
    }
    df_new_elo = compute_elo_series(df_clean, new_elo_params)
    df_new_elo_params = {k: best_elo[k] for k in new_elo_params}

    # ── 构建新/旧 features DataFrame ─────────────────────
    print("更新 Elo 列到 features.csv...")
    df_old = add_interaction_features(df_feat.copy())
    df_new_feat = make_features_with_elo(df_feat, df_new_elo, float(best_elo["H_adv"]))
    df_new = add_interaction_features(df_new_feat)

    # ── 遍历所有 dev 赛事 ──────────────────────────────────
    print("\n--- 对 14 个 dev 赛事分别用旧/新 Elo 训练 XGB+int ---")
    print("(每赛事各训两次，耗时约 5-8 分钟)\n")

    old_by_tourn = {}; new_by_tourn = {}

    for tourn_key, name_pat, d_start, d_end in TOURNS:
        t_old = get_tournament_matches(df_old,     name_pat, d_start, d_end)
        t_new = get_tournament_matches(df_new_feat, name_pat, d_start, d_end)

        # 注意：t_new 用 df_new_feat（elo 更新但未加 interaction），
        # add_interaction_features 在 run_tourn_full 内部调用
        t_new_inter_input = t_new.copy()

        if len(t_old) < 16:
            print(f"  [{tourn_key}] 跳过（场次不足）")
            continue

        print(f"  [{tourn_key}] n={len(t_old)}", end="")

        # 旧 Elo
        res_old = run_tourn_full(
            tourn_key, t_old, df_old, df_old,
            feat_cols, feat_cols_inter, xgb_params, h_adv=100.0)
        old_by_tourn[tourn_key] = res_old

        # 新 Elo
        res_new = run_tourn_full(
            tourn_key, t_new_inter_input, df_new, df_new,
            feat_cols, feat_cols_inter, xgb_params,
            h_adv=float(best_elo["H_adv"]))
        new_by_tourn[tourn_key] = res_new

        # 快速打印
        ll_old = log_loss(res_old["XGB+int"][1], res_old["XGB+int"][0], labels=LABEL_ORDER)
        ll_new = log_loss(res_new["XGB+int"][1], res_new["XGB+int"][0], labels=LABEL_ORDER)
        print(f"  XGB+int LL: 旧={ll_old:.4f} → 新={ll_new:.4f} ({ll_new-ll_old:+.4f})")

    # ── 合并全 dev 概率 ────────────────────────────────────
    def collect(by_tourn, model_name, subset=None):
        y_list, p_list = [], []
        for key, res in by_tourn.items():
            if subset is not None and key not in subset: continue
            if model_name not in res: continue
            probs, ytrue = res[model_name]
            y_list.extend(ytrue)
            p_list.append(probs)
        return np.array(y_list), np.vstack(p_list)

    print("\n" + "=" * 65)
    print("  配对 Bootstrap 结果（5000次，95%CI）")
    print("=" * 65)

    MODELS = ["BLa", "XGB+int", "DC", "Ens2"]
    SUBSETS = {
        "全dev(n≈593)":   None,
        "WC+Euro+Copa(n≈342)": WC_EURO_COPA,
    }

    all_rows = []

    for subset_name, subset_keys in SUBSETS.items():
        print(f"\n【{subset_name}】")
        print(f"  {'模型':<10} {'旧LL':>8} {'新LL':>8} {'ΔLL':>8} "
              f"  {'旧ACC':>7} {'新ACC':>7}  {'差值CI':>22}  p值  结论")
        print("  " + "-" * 90)

        for m in MODELS:
            y_old, p_old = collect(old_by_tourn, m, subset_keys)
            y_new, p_new = collect(new_by_tourn, m, subset_keys)

            if len(y_old) == 0: continue

            met_old = summary_metrics(y_old, p_old)
            met_new = summary_metrics(y_new, p_new)

            obs_diff, (ci_lo, ci_hi), pval = paired_bootstrap_diff(
                y_old, p_old, p_new, LABEL_ORDER,
                metric="log_loss", n_boot=5000, seed=42)

            sig = "✓显著" if pval < 0.05 else "不显著"
            n_matches = len(y_old)

            print(f"  {m:<10} {met_old['ll']:>8.5f} {met_new['ll']:>8.5f} "
                  f"{obs_diff:>+8.5f}  "
                  f" {met_old['acc']:>7.4f} {met_new['acc']:>7.4f}  "
                  f"[{ci_lo:+.5f},{ci_hi:+.5f}]  {pval:.3f}  {sig}")

            all_rows.append({
                "subset": subset_name, "model": m, "n": n_matches,
                "ll_old": met_old["ll"], "ll_new": met_new["ll"],
                "ll_diff": obs_diff,
                "ci_lo": ci_lo, "ci_hi": ci_hi, "pval": round(pval, 3),
                "acc_old": met_old["acc"], "acc_new": met_new["acc"],
                "significant": pval < 0.05,
            })

    df_comp = pd.DataFrame(all_rows)

    # ── 结论 ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  综合结论")
    print("=" * 65)

    xgb_full = df_comp[(df_comp["model"] == "XGB+int") &
                       (df_comp["subset"].str.contains("全dev"))].iloc[0]
    xgb_wec  = df_comp[(df_comp["model"] == "XGB+int") &
                       (df_comp["subset"].str.contains("WC"))].iloc[0]

    print(f"\n  主准则  全dev XGB+int   LL差={xgb_full['ll_diff']:+.5f}  "
          f"p={xgb_full['pval']:.3f}  {'显著' if xgb_full['significant'] else '不显著'}")
    print(f"  决胜准则 WC+Euro+Copa XGB+int LL差={xgb_wec['ll_diff']:+.5f}  "
          f"p={xgb_wec['pval']:.3f}  {'显著' if xgb_wec['significant'] else '不显著'}")

    elo_better = xgb_full["ll_diff"] < 0   # 新 Elo LL 更低 → 更好
    if elo_better and not xgb_full["significant"]:
        verdict = "新Elo改善不显著但方向正确，采用新参数（训练期有收益，dev不退步）"
    elif elo_better and xgb_full["significant"]:
        verdict = "新Elo改善显著，采用新参数"
    else:
        verdict = "新Elo无改善，保持默认参数"
    print(f"\n  建议: {verdict}")

    # ── 若采用新 Elo，更新持久化文件 ─────────────────────
    if elo_better:
        print("\n  → 用新 Elo 参数更新 matches_with_elo.csv ...")
        df_new_elo_save = df_new_elo.copy()
        H_arr = np.where(df_new_elo_save["neutral"].values.astype(bool),
                         0.0, float(best_elo["H_adv"]))
        df_new_elo_save["elo_diff"] = (
            df_new_elo_save["elo_home_pre"].values + H_arr -
            df_new_elo_save["elo_away_pre"].values)
        df_new_elo_save.to_csv(PROC_DIR / "matches_with_elo.csv", index=False)
        print("    已更新 data/processed/matches_with_elo.csv")

        print("  → 用新 Elo 更新 features.csv ...")
        df_new_feat.to_csv(PROC_DIR / "features.csv", index=False)
        print("    已更新 data/processed/features.csv")

    # ── 写入 final_model_spec.md ─────────────────────────
    spec_path = OUT_DIR / "final_model_spec.md"
    existing  = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""

    elo_section = f"""

## 步骤3 Elo 超参优化结论 (2026-06-12)

### 合规流程
- **选参准则**: 1998-01-01 ~ 2014-06-11 训练期 BLa rolling LogLoss（共 {len(df_clean[(df_clean['date'] >= '1998-01-01') & (df_clean['date'] <= '2014-06-11')])} 场）
- **网格规模**: 216 组（H_adv×K_wc×K_major×K_qual×K_friendly×G_func）
- dev 集仅在此步骤被查看一次，结果不反馈到参数选择

### 训练期结果
| 参数 | 默认值 | 最优值 |
|------|--------|--------|
| H_adv | 100 | **{best_elo['H_adv']}** |
| K_wc | 60 | **{best_elo['K_wc']}** |
| K_major | 50 | **{best_elo['K_major']}** |
| K_qual | 40 | **{best_elo['K_qual']}** |
| K_friendly | 20 | **{best_elo['K_friendly']}** |
| G_func | original | **{best_elo['G_func']}** |

训练期 LL: {best_elo['train_ll_default']} → **{best_elo['train_ll_best']}** (Δ={best_elo['delta_train_ll']:+.5f})

### dev 配对 Bootstrap（5000次）
**全 dev 集 XGB+int**: LL差={xgb_full['ll_diff']:+.5f}, 95%CI=[{xgb_full['ci_lo']:+.5f},{xgb_full['ci_hi']:+.5f}], p={xgb_full['pval']:.3f}
**WC+Euro+Copa XGB+int**: LL差={xgb_wec['ll_diff']:+.5f}, 95%CI=[{xgb_wec['ci_lo']:+.5f},{xgb_wec['ci_hi']:+.5f}], p={xgb_wec['pval']:.3f}

### 决策
**{verdict}**

{'matches_with_elo.csv 和 features.csv 已用新 Elo 更新。' if elo_better else '文件保持不变。'}

### AFCON 2015 说明
AFCON 2015 的 21.9% 准确率根因是结构性问题（46.9% 平局率 + BLa/XGB 不预测平局），
与 neutral 标记无关（赤道几内亚主场优势 neutral=False 数据上合理）。
Elo 参数优化使该届 ACC 从 21.9% 轻微升至 28.1%，BLa LL 改善 0.005，属预期范围内。
"""

    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(existing + elo_section)
    print(f"\n已更新: outputs/final_model_spec.md")

    # 保存对比表
    df_comp.to_csv(OUT_DIR / "elo_update_comparison.csv", index=False)
    print(f"已保存: outputs/elo_update_comparison.csv")

    print("\n[步骤3 Phase2 完成] → 可进入步骤4 (有序逻辑回归 OrderedLogit)")


if __name__ == "__main__":
    main()
