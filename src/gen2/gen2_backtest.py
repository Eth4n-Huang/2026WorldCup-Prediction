"""
src/gen2/gen2_backtest.py
二代模型回测: OLR (有序逻辑回归) + 三模型融合 (DC + XGB + OLR)

任务1: OLR 作为第三个模型，三届WC上单独报告ACC/Brier
任务2: 等权融合 + 训练期优化权重融合，对比表

防泄漏铁律:
  - 所有参数(OLR系数、融合权重)只在训练期1998-2014-06-10数据上拟合
  - 融合权重在1998-2014训练期内一次性优化，三届回测共用，不再调整
  - 每届WC回测的三个单模型各自在该届开赛前数据上独立重训
  - 禁止随机划分/K折/打乱

输出:
  outputs/gen2/backtest_gen2_YYYY.csv  — 三届明细
  outputs/gen2/gen2_summary.json       — 汇总数字
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 路径设置：src/gen2/ → src/ → 项目根 ──────────────────────────────
HERE     = Path(__file__).parent           # src/gen2/
SRC_DIR  = HERE.parent                     # src/
ROOT_DIR = SRC_DIR.parent                  # 项目根
sys.path.insert(0, str(SRC_DIR))

from step4_train import TRAIN_START, get_feature_cols, fit_pipeline
from step5c_devset import (
    fit_dc_fast,
    dc_probs_for_matches,
    add_interaction_features,
)
from step6d_new_models import fit_ordered_logit, predict_ordered_logit
from metrics import multiclass_brier

PROC_DIR = ROOT_DIR / "data" / "processed"
OUT_DIR  = ROOT_DIR / "outputs" / "gen2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["A", "D", "H"]

WC_OPENING = {
    2014: pd.Timestamp("2014-06-12"),
    2018: pd.Timestamp("2018-06-14"),
    2022: pd.Timestamp("2022-11-20"),
}

# 权重优化截止日：WC2014 开赛前一天
WEIGHT_OPT_END = pd.Timestamp("2014-06-11")


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def argmax_acc(y_true, probs):
    preds = [LABEL_ORDER[i] for i in np.argmax(probs, axis=1)]
    return float(np.mean(np.array(preds) == np.array(y_true)))


def safe_brier(y_true, probs):
    return multiclass_brier(y_true, probs, LABEL_ORDER)


def find_optimal_weights(y_val, p_dc, p_xgb, p_olr, step=0.05):
    """
    网格搜索 (w_dc, w_xgb, w_olr)，约束非负和为1，使验证集Brier最小。
    step=0.05 → 21×21 = 231个候选组合。
    """
    best_w     = (1 / 3, 1 / 3, 1 / 3)
    best_brier = float("inf")

    vals = np.round(np.arange(0.0, 1.0 + step, step), 10)
    for w_dc in vals:
        for w_xgb in vals:
            w_olr = round(1.0 - float(w_dc) - float(w_xgb), 10)
            if w_olr < -1e-9:
                continue
            w_olr = max(w_olr, 0.0)
            p_ens = float(w_dc) * p_dc + float(w_xgb) * p_xgb + float(w_olr) * p_olr
            br    = safe_brier(y_val, p_ens)
            if br < best_brier:
                best_brier = br
                best_w     = (float(w_dc), float(w_xgb), float(w_olr))

    return best_w, best_brier


# ══════════════════════════════════════════════════════════════
#  核心: 拟合三模型 + 预测
# ══════════════════════════════════════════════════════════════

def fit_three_models(df_train, df_train_inter, feat_cols_inter,
                     xgb_params, opening_date):
    """
    在 df_train / df_train_inter 上拟合 DC、XGB、OLR 三个模型。
    opening_date: DC 时间衰减的参考日期（比赛越近权重越高）
    返回: (dc_model, xgb_pkg, olr_pkg)
    """
    # DC: Dixon-Coles 双泊松，half_life=730天（已从最终模型规格冻结）
    dc = fit_dc_fast(df_train, opening_date)

    # XGB + 交互特征（保序回归校准，与一代最终模型同参数）
    pkg_xgb = fit_pipeline(
        df_train_inter, "xgb",
        feat_cols=feat_cols_inter,
        xgb_max_depth=xgb_params["max_depth"],
        xgb_lr=xgb_params["lr"],
        xgb_n_est=xgb_params["n_est"],
        lam=xgb_params["lam"],
    )

    # OLR: 有序逻辑回归 (A < D < H)，statsmodels OrderedModel
    olr_res, olr_scaler, olr_cols = fit_ordered_logit(
        df_train_inter, feat_cols_inter)

    return dc, pkg_xgb, (olr_res, olr_scaler, olr_cols)


def get_all_probs(dc, pkg_xgb, olr_pkg,
                  df_matches, feat_cols_inter):
    """
    三模型对 df_matches 输出概率，均为 (N,3) [A,D,H] 顺序。
    df_matches 必须是原始特征DataFrame（不含交互特征），
    函数内部会自动添加交互特征列。
    """
    df_inter = add_interaction_features(df_matches)

    # DC: 向量化双泊松，返回 [A, D, H]
    p_dc = dc_probs_for_matches(dc, df_matches)

    # XGB: 校准后概率，label_order=['A','D','H']
    X_xgb = df_inter[feat_cols_inter].values.astype(np.float32)
    p_xgb = pkg_xgb["calibrated_model"].predict_proba(X_xgb)

    # OLR: statsmodels OrderedModel 预测，列顺序 [A=0, D=1, H=2]
    olr_res, olr_scaler, olr_cols = olr_pkg
    X_olr = df_inter[feat_cols_inter].values.astype(float)
    if olr_res is not None:
        p_olr = predict_ordered_logit(olr_res, olr_scaler, X_olr, olr_cols)
    else:
        print("  [警告] OLR 拟合失败，回退为 DC 概率")
        p_olr = p_dc.copy()

    return p_dc, p_xgb, p_olr


# ══════════════════════════════════════════════════════════════
#  步骤1: 融合权重优化（在 1998-2014-06-10 内完成，不接触测试届）
# ══════════════════════════════════════════════════════════════

def optimize_ensemble_weights(df_raw, df_inter, feat_cols_inter, xgb_params):
    """
    数据窗口: TRAIN_START ~ WEIGHT_OPT_END (=2014-06-11)
    最后15% 作为验证集（时间最新），搜索最小化Brier的(w_dc,w_xgb,w_olr)。
    权重在训练期定死，三届回测共用，不再调整。
    """
    print("\n" + "=" * 64)
    print("  步骤1: 融合权重优化 (1998-01-01 ~ 2014-06-10)")
    print("=" * 64)

    mask     = (df_raw["date"] >= TRAIN_START) & (df_raw["date"] <= WEIGHT_OPT_END)
    df_opt   = df_raw[mask].copy().reset_index(drop=True)
    df_opt_i = df_inter[mask].copy().reset_index(drop=True)

    n      = len(df_opt)
    n_val  = max(int(n * 0.15), 50)
    df_fit   = df_opt.iloc[:n - n_val].copy()
    df_val   = df_opt.iloc[n - n_val:].copy()
    df_fit_i = df_opt_i.iloc[:n - n_val].copy()

    opening_for_dc = df_val["date"].min()   # DC时间衰减参考日
    print(f"  拟合期: {df_fit['date'].min().date()} ~ {df_fit['date'].max().date()}  ({len(df_fit)} 场)")
    print(f"  验证期: {df_val['date'].min().date()} ~ {df_val['date'].max().date()}  ({len(df_val)} 场)")

    print("  拟合 DC, XGB, OLR...", end="", flush=True)
    dc, pkg_xgb, olr_pkg = fit_three_models(
        df_fit, df_fit_i, feat_cols_inter, xgb_params, opening_for_dc)
    print(" 完成")

    p_dc, p_xgb, p_olr = get_all_probs(dc, pkg_xgb, olr_pkg, df_val, feat_cols_inter)
    y_val = df_val["result"].values

    print(f"\n  验证集各模型表现 (n={len(y_val)}):")
    print(f"  {'模型':<10} {'ACC':>7} {'Brier':>8}")
    print(f"  {'-'*26}")
    for name, p in [("DC", p_dc), ("XGB", p_xgb), ("OLR", p_olr)]:
        a = argmax_acc(y_val, p)
        b = safe_brier(y_val, p)
        print(f"  {name:<10} {a:>7.4f} {b:>8.4f}")

    p_eq  = (p_dc + p_xgb + p_olr) / 3
    a_eq  = argmax_acc(y_val, p_eq)
    b_eq  = safe_brier(y_val, p_eq)
    print(f"  {'Ens①等权':<10} {a_eq:>7.4f} {b_eq:>8.4f}")

    print(f"\n  搜索最优权重 (step=0.05)...", end="", flush=True)
    (w_dc, w_xgb, w_olr), best_br = find_optimal_weights(
        y_val, p_dc, p_xgb, p_olr, step=0.05)
    print(" 完成")

    p_opt = w_dc * p_dc + w_xgb * p_xgb + w_olr * p_olr
    a_opt = argmax_acc(y_val, p_opt)
    print(f"  {'Ens②优化':<10} {a_opt:>7.4f} {best_br:>8.4f}")
    print(f"\n  最优权重: w_DC={w_dc:.2f}  w_XGB={w_xgb:.2f}  w_OLR={w_olr:.2f}")
    print(f"  (权重冻结，三届回测直接套用，不再调整)")

    return (w_dc, w_xgb, w_olr)


# ══════════════════════════════════════════════════════════════
#  步骤2: 单届 WC 回测
# ══════════════════════════════════════════════════════════════

def run_wc_backtest_gen2(year, df_raw, df_inter, feat_cols_inter,
                         xgb_params, ens_weights):
    """
    对指定届WC运行gen2回测。
    - 每个单模型在该届开赛前数据上独立重训（防泄漏）
    - 融合权重 ens_weights 固定，不在本届重优化
    返回: dict {model_name: {"acc": float, "brier": float, "n": int}}
    """
    w_dc, w_xgb, w_olr = ens_weights
    opening = WC_OPENING[year]

    print(f"\n{'=' * 64}")
    print(f"  {year} WC 回测 (Gen2)")
    print(f"{'=' * 64}")

    mask_train = (df_raw["date"] >= TRAIN_START) & (df_raw["date"] < opening)
    df_train   = df_raw[mask_train].copy()
    df_train_i = df_inter[mask_train].copy()
    print(f"  训练: {TRAIN_START} ~ {(opening - pd.Timedelta(days=1)).date()}  ({len(df_train)} 场)")

    print(f"  拟合 DC, XGB, OLR...", end="", flush=True)
    dc, pkg_xgb, olr_pkg = fit_three_models(
        df_train, df_train_i, feat_cols_inter, xgb_params, opening)
    print(" 完成")

    # WC 测试场次
    df_wc = (df_raw[
        (df_raw["tournament"] == "FIFA World Cup") &
        (df_raw["date"].dt.year == year)
    ].sort_values("date").copy())
    print(f"  测试: {len(df_wc)} 场  ({df_wc['date'].min().date()} ~ {df_wc['date'].max().date()})")

    p_dc, p_xgb, p_olr = get_all_probs(dc, pkg_xgb, olr_pkg, df_wc, feat_cols_inter)
    p_eq  = (p_dc + p_xgb + p_olr) / 3
    p_opt = w_dc * p_dc + w_xgb * p_xgb + w_olr * p_olr

    y_true = df_wc["result"].values

    results = {}
    for name, p in [("DC",      p_dc),
                    ("XGB",     p_xgb),
                    ("OLR",     p_olr),
                    ("Ens①等权", p_eq),
                    ("Ens②优化", p_opt)]:
        results[name] = {
            "acc":   argmax_acc(y_true, p),
            "brier": safe_brier(y_true, p),
            "n":     len(y_true),
        }

    # 打印本届小表
    print(f"\n  {'模型':<10} {'ACC':>7} {'Brier':>8}")
    print(f"  {'-'*26}")
    for name, r in results.items():
        print(f"  {name:<10} {r['acc']:>7.4f} {r['brier']:>8.4f}")

    # 保存明细 CSV
    df_out = df_wc[["date", "home_team", "away_team", "result",
                     "elo_home_pre", "elo_away_pre"]].copy()
    for model_key, p in [("dc", p_dc), ("xgb", p_xgb), ("olr", p_olr),
                          ("ens1", p_eq), ("ens2", p_opt)]:
        for j, c in enumerate(LABEL_ORDER):
            df_out[f"{model_key}_p{c}"] = p[:, j]
        preds = [LABEL_ORDER[i] for i in np.argmax(p, axis=1)]
        df_out[f"{model_key}_pred"]    = preds
        df_out[f"{model_key}_correct"] = (
            np.array(preds) == np.array(y_true)).astype(int)

    csv_path = OUT_DIR / f"backtest_gen2_{year}.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  明细已保存: {csv_path.relative_to(ROOT_DIR)}")

    return results


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Gen2 回测: OLR + 三模型融合 (DC + XGB + OLR)")
    print("=" * 64)

    # 读取数据
    df_raw   = pd.read_csv(PROC_DIR / "features.csv", parse_dates=["date"])
    df_raw   = df_raw.sort_values("date").reset_index(drop=True)
    df_inter = add_interaction_features(df_raw)

    feat_cols       = get_feature_cols(df_raw)
    # 确保交互特征不重复
    extra = [c for c in ["elo_diff_ko", "rest_diff_ko"] if c not in feat_cols]
    feat_cols_inter = feat_cols + extra

    print(f"基础特征: {len(feat_cols)}  +  交互特征: {len(extra)}  =  {len(feat_cols_inter)} 列")

    # 读取 XGB 超参（从一代 train_params.json，不重新调参）
    with open(ROOT_DIR / "outputs" / "train_params.json", encoding="utf-8") as f:
        params = json.load(f)
    xgb_params = {
        "max_depth": params["xgb"]["max_depth"],
        "lr":        params["xgb"]["lr"],
        "n_est":     params["xgb"]["n_est"],
        "lam":       params["xgb"]["lam"],
    }
    print(f"XGB超参: depth={xgb_params['max_depth']}  lr={xgb_params['lr']}  "
          f"n={xgb_params['n_est']}  λ={xgb_params['lam']}")

    # ── 步骤1: 权重优化 ──────────────────────────────────────────
    ens_weights = optimize_ensemble_weights(
        df_raw, df_inter, feat_cols_inter, xgb_params)
    w_dc, w_xgb, w_olr = ens_weights

    # ── 步骤2: 三届回测 ──────────────────────────────────────────
    all_results = {}
    for year in [2014, 2018, 2022]:
        all_results[year] = run_wc_backtest_gen2(
            year, df_raw, df_inter, feat_cols_inter, xgb_params, ens_weights)

    # ══════════════════════════════════════════════════════════════
    #  汇总对比表
    # ══════════════════════════════════════════════════════════════
    MODELS = ["DC", "XGB", "OLR", "Ens①等权", "Ens②优化"]

    print(f"\n\n{'#' * 64}")
    print(f"  Gen2 三届回测汇总对比表")
    print(f"{'#' * 64}")

    # 准确率表
    col_w = 10
    print(f"\n--- 准确率 (ACC, argmax) ---")
    hdr = f"{'届':<8} {'n':>4}"
    for m in MODELS:
        hdr += f"  {m:>{col_w}}"
    print(hdr)
    print("-" * (13 + (col_w + 2) * len(MODELS)))
    for year in [2014, 2018, 2022]:
        r   = all_results[year]
        n   = r["DC"]["n"]
        row = f"{year:<8} {n:>4}"
        for m in MODELS:
            row += f"  {r[m]['acc']:>{col_w}.4f}"
        print(row)

    # 三届均值
    print("-" * (13 + (col_w + 2) * len(MODELS)))
    avg_acc   = {m: float(np.mean([all_results[y][m]["acc"]   for y in [2014, 2018, 2022]])) for m in MODELS}
    avg_brier = {m: float(np.mean([all_results[y][m]["brier"] for y in [2014, 2018, 2022]])) for m in MODELS}
    n_total   = sum(all_results[y]["DC"]["n"] for y in [2014, 2018, 2022])

    row = f"{'均值':<8} {n_total:>4}"
    for m in MODELS:
        row += f"  {avg_acc[m]:>{col_w}.4f}"
    print(row)

    # Brier 表
    print(f"\n--- Brier Score (越低越好) ---")
    print(hdr)
    print("-" * (13 + (col_w + 2) * len(MODELS)))
    for year in [2014, 2018, 2022]:
        r   = all_results[year]
        n   = r["DC"]["n"]
        row = f"{year:<8} {n:>4}"
        for m in MODELS:
            row += f"  {r[m]['brier']:>{col_w}.4f}"
        print(row)
    print("-" * (13 + (col_w + 2) * len(MODELS)))
    row = f"{'均值':<8} {n_total:>4}"
    for m in MODELS:
        row += f"  {avg_brier[m]:>{col_w}.4f}"
    print(row)

    # ── 结论 ──────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  融合权重 (训练期优化，三届共用):")
    print(f"    w_DC={w_dc:.2f}  w_XGB={w_xgb:.2f}  w_OLR={w_olr:.2f}")

    single_models = ["DC", "XGB", "OLR"]
    best_single_acc   = max(single_models, key=lambda m: avg_acc[m])
    best_single_brier = min(single_models, key=lambda m: avg_brier[m])

    print(f"\n  三届平均ACC:")
    for m in MODELS:
        marker = " ←最强" if avg_acc[m] == max(avg_acc.values()) else ""
        print(f"    {m:<10}: {avg_acc[m]:.4f}{marker}")

    print(f"\n  三届平均Brier:")
    for m in MODELS:
        marker = " ←最低" if avg_brier[m] == min(avg_brier.values()) else ""
        print(f"    {m:<10}: {avg_brier[m]:.4f}{marker}")

    # 诚实评估融合效果
    print(f"\n  [诚实报告] 融合效果评估:")
    for ens in ["Ens①等权", "Ens②优化"]:
        delta_acc   = avg_acc[ens]   - avg_acc[best_single_acc]
        delta_brier = avg_brier[ens] - avg_brier[best_single_brier]
        acc_improved   = delta_acc   > 0
        brier_improved = delta_brier < 0
        print(f"  {ens} vs 最强单模型({best_single_acc}/{best_single_brier}):")
        print(f"    ACC  : {avg_acc[ens]:.4f} vs {avg_acc[best_single_acc]:.4f}  "
              f"Δ={delta_acc:+.4f}  ({'改善' if acc_improved else '未改善'})")
        print(f"    Brier: {avg_brier[ens]:.4f} vs {avg_brier[best_single_brier]:.4f}  "
              f"Δ={delta_brier:+.4f}  ({'改善' if brier_improved else '未改善'})")

    # ── 保存汇总 JSON ──────────────────────────────────────────────
    summary = {
        "ens_weights":   {"w_dc": w_dc, "w_xgb": w_xgb, "w_olr": w_olr},
        "weight_opt_period": f"{TRAIN_START} ~ {WEIGHT_OPT_END.date()}",
        "by_year": {
            str(y): {m: {"acc": all_results[y][m]["acc"],
                         "brier": all_results[y][m]["brier"],
                         "n":    all_results[y][m]["n"]}
                     for m in MODELS}
            for y in [2014, 2018, 2022]
        },
        "avg_3wc": {m: {"acc": avg_acc[m], "brier": avg_brier[m]}
                    for m in MODELS},
    }
    with open(OUT_DIR / "gen2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  汇总: outputs/gen2/gen2_summary.json")
    print(f"  明细: outputs/gen2/backtest_gen2_YYYY.csv")
    print(f"\n[Gen2 完成] 等待确认后可继续校准阶段")


if __name__ == "__main__":
    main()
