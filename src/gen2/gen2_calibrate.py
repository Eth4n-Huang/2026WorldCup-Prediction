"""
src/gen2/gen2_calibrate.py
DC 概率校准: Platt Scaling vs Isotonic Regression

防泄漏铁律:
  - 校准函数在 1998-2014-06-10 训练期末15%上拟合，三届回测期套用
  - DC 模型仍各届独立重训（与 gen2_backtest.py 一致）
  - 不接触一代代码，不接触 live 数据

输出:
  outputs/gen2/dc_calibration_curve.png      — 校准集可靠性曲线
  outputs/gen2/dc_calibration_curve_wc.png   — 三届WC合并可靠性曲线
  outputs/gen2/calibration_results.json      — 数字汇总
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

HERE     = Path(__file__).parent
SRC_DIR  = HERE.parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from step4_train import TRAIN_START
from step5c_devset import fit_dc_fast, dc_probs_for_matches
from metrics import multiclass_brier

PROC_DIR = ROOT_DIR / "data" / "processed"
OUT_DIR  = ROOT_DIR / "outputs" / "gen2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER    = ["A", "D", "H"]
WEIGHT_OPT_END = pd.Timestamp("2014-06-11")

WC_OPENING = {
    2014: pd.Timestamp("2014-06-12"),
    2018: pd.Timestamp("2018-06-14"),
    2022: pd.Timestamp("2022-11-20"),
}


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def argmax_acc(y_true, probs):
    preds = [LABEL_ORDER[i] for i in np.argmax(probs, axis=1)]
    return float(np.mean(np.array(preds) == np.array(y_true)))


def safe_brier(y_true, probs):
    return multiclass_brier(y_true, probs, LABEL_ORDER)


def compute_ece(y_true, probs, n_bins=10):
    """Expected Calibration Error: 三类别 one-vs-rest 均值"""
    n  = len(y_true)
    ea = []
    for i, cls in enumerate(LABEL_ORDER):
        y_bin = (np.array(y_true) == cls).astype(float)
        p_cls = probs[:, i]
        bins  = np.linspace(0, 1, n_bins + 1)
        ece   = 0.0
        for j in range(n_bins):
            mask = (p_cls >= bins[j]) & (p_cls < bins[j + 1])
            if mask.sum() == 0:
                continue
            ece += mask.sum() / n * abs(y_bin[mask].mean() - p_cls[mask].mean())
        ea.append(ece)
    return float(np.mean(ea))


def apply_platt(platt_model, probs: np.ndarray) -> np.ndarray:
    """Platt: LogReg on raw probs, 确保列序 = LABEL_ORDER"""
    out = platt_model.predict_proba(probs)
    # platt.classes_ 是训练时的 unique label，顺序不一定是 LABEL_ORDER
    idx = [list(platt_model.classes_).index(c) for c in LABEL_ORDER]
    return out[:, idx]


def apply_isotonic(iso_regs: list, probs: np.ndarray) -> np.ndarray:
    """Isotonic: 逐类预测 + clip + 行归一化"""
    cal = np.column_stack([iso_regs[i].predict(probs[:, i]) for i in range(3)])
    cal = np.clip(cal, 1e-7, 1.0)
    cal /= cal.sum(axis=1, keepdims=True)
    return cal


# ══════════════════════════════════════════════════════════════
#  Step 1: 在训练期末15%拟合校准函数
# ══════════════════════════════════════════════════════════════

def fit_calibrators(df_raw):
    """
    训练期: 1998-01-01 ~ 2014-06-10
    DC 在前85%数据上拟合；
    Platt / Isotonic 在末15%（校准集）DC预测上拟合。
    返回: (platt, iso_regs, p_dc_cal, y_cal, best_method)
    """
    print("\n" + "=" * 62)
    print("  Step 1: 校准函数拟合 (1998-01-01 ~ 2014-06-10, 末15%校准)")
    print("=" * 62)

    mask   = (df_raw["date"] >= TRAIN_START) & (df_raw["date"] <= WEIGHT_OPT_END)
    df_opt = df_raw[mask].copy().reset_index(drop=True)

    n     = len(df_opt)
    n_cal = max(int(n * 0.15), 50)
    df_fit = df_opt.iloc[:n - n_cal].copy()
    df_cal = df_opt.iloc[n - n_cal:].copy()

    opening_dc = df_cal["date"].min()
    print(f"  DC 训练: {df_fit['date'].min().date()} ~ {df_fit['date'].max().date()}  ({len(df_fit)} 场)")
    print(f"  校准集:  {df_cal['date'].min().date()} ~ {df_cal['date'].max().date()}  ({len(df_cal)} 场)")

    print("  拟合 DC...", end="", flush=True)
    dc = fit_dc_fast(df_fit, opening_dc)
    print(" 完成")

    p_dc_cal = dc_probs_for_matches(dc, df_cal)   # (N, 3) [A, D, H]
    y_cal    = df_cal["result"].values

    # ── Platt Scaling ──────────────────────────────────────────
    print("  拟合 Platt Scaling (多分类 LogReg on DC probs)...", end="", flush=True)
    platt = LogisticRegression(
        C=1.0, multi_class="multinomial", solver="lbfgs",
        max_iter=1000, random_state=42,
    )
    platt.fit(p_dc_cal, y_cal)
    print(" 完成")

    # ── Isotonic Regression ────────────────────────────────────
    print("  拟合 Isotonic Regression (逐类别)...", end="", flush=True)
    iso_regs = []
    for i, cls in enumerate(LABEL_ORDER):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_dc_cal[:, i], (y_cal == cls).astype(float))
        iso_regs.append(iso)
    print(" 完成")

    # ── 校准集对比 ──────────────────────────────────────────────
    p_platt_cal = apply_platt(platt, p_dc_cal)
    p_iso_cal   = apply_isotonic(iso_regs, p_dc_cal)

    print(f"\n  校准集表现 (n={len(y_cal)}):")
    print(f"  {'方法':<15} {'ACC':>7} {'Brier':>8} {'LogLoss':>9} {'ECE':>7}")
    print(f"  {'-' * 47}")
    cal_metrics = {}
    for name, p in [("DC 原始", p_dc_cal),
                    ("DC + Platt", p_platt_cal),
                    ("DC + Isotonic", p_iso_cal)]:
        acc = argmax_acc(y_cal, p)
        br  = safe_brier(y_cal, p)
        ll  = log_loss(y_cal, p, labels=LABEL_ORDER)
        ece = compute_ece(y_cal, p)
        print(f"  {name:<15} {acc:>7.4f} {br:>8.4f} {ll:>9.5f} {ece:>7.4f}")
        cal_metrics[name] = {"acc": acc, "brier": br, "logloss": ll, "ece": ece}

    best_method = ("Platt" if cal_metrics["DC + Platt"]["brier"]
                   <= cal_metrics["DC + Isotonic"]["brier"] else "Isotonic")
    print(f"\n  校准集最优方法: DC + {best_method}  "
          f"(Platt Brier={cal_metrics['DC + Platt']['brier']:.4f}, "
          f"Isotonic Brier={cal_metrics['DC + Isotonic']['brier']:.4f})")

    return platt, iso_regs, p_dc_cal, p_platt_cal, p_iso_cal, y_cal, cal_metrics, best_method


# ══════════════════════════════════════════════════════════════
#  Step 2: 三届回测（校准函数固定，DC各届独立重训）
# ══════════════════════════════════════════════════════════════

def run_calibrated_backtest(year, df_raw, platt, iso_regs):
    """
    DC 在该届开赛前数据重训；
    Platt / Isotonic 直接套用训练期校准函数。
    """
    opening  = WC_OPENING[year]
    df_train = df_raw[(df_raw["date"] >= TRAIN_START) &
                      (df_raw["date"] < opening)].copy()
    dc = fit_dc_fast(df_train, opening)

    df_wc  = (df_raw[(df_raw["tournament"] == "FIFA World Cup") &
                     (df_raw["date"].dt.year == year)]
              .sort_values("date").copy())
    y_true = df_wc["result"].values

    p_raw   = dc_probs_for_matches(dc, df_wc)
    p_platt = apply_platt(platt, p_raw)
    p_iso   = apply_isotonic(iso_regs, p_raw)

    results = {}
    for name, p in [("DC原始", p_raw), ("DC+Platt", p_platt), ("DC+Isotonic", p_iso)]:
        results[name] = {
            "acc":     argmax_acc(y_true, p),
            "brier":   safe_brier(y_true, p),
            "logloss": log_loss(y_true, p, labels=LABEL_ORDER),
            "ece":     compute_ece(y_true, p),
            "n":       len(y_true),
        }

    return results, p_raw, p_platt, p_iso, y_true


# ══════════════════════════════════════════════════════════════
#  Step 3: 可靠性曲线（Reliability Diagram）
# ══════════════════════════════════════════════════════════════

def _draw_reliability(ax, y_bin, probs_dict, n_bins, title):
    """单子图：多条曲线 + 完美对角线"""
    COLORS = {
        "DC Raw":       "#2196F3",
        "DC+Platt":     "#FF9800",
        "DC+Isotonic":  "#4CAF50",
        "Perfect":      "#888888",
    }
    MARKERS = {"DC Raw": "o", "DC+Platt": "s", "DC+Isotonic": "^"}

    for label, p_cls in probs_dict.items():
        try:
            frac, mean_p = calibration_curve(y_bin, p_cls, n_bins=n_bins)
            ax.plot(mean_p, frac, marker=MARKERS[label], linestyle="-",
                    label=label, color=COLORS[label], linewidth=2, markersize=6)
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "--", color=COLORS["Perfect"],
            alpha=0.6, linewidth=1.5, label="Perfect")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)


def plot_calibration_curves(p_dc_cal, p_platt_cal, p_iso_cal, y_cal,
                            p_dc_wc, p_platt_wc, p_iso_wc, y_wc):
    """
    图1: 校准集（训练期末15%）可靠性曲线
    图2: 三届WC合并可靠性曲线
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    CLASS_NAMES = {"A": "Away Win", "D": "Draw", "H": "Home Win"}

    # ── 图1: 校准集 ────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    for i, cls in enumerate(LABEL_ORDER):
        y_bin = (np.array(y_cal) == cls).astype(float)
        _draw_reliability(
            axes1[i], y_bin,
            {"DC Raw": p_dc_cal[:, i],
             "DC+Platt": p_platt_cal[:, i],
             "DC+Isotonic": p_iso_cal[:, i]},
            n_bins=10,
            title=CLASS_NAMES[cls],
        )
    fig1.suptitle(
        "DC Probability Calibration — Reliability Diagram\n"
        "Training holdout (last 15% of 1998–2014 period)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path1 = OUT_DIR / "dc_calibration_curve.png"
    plt.savefig(path1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  校准集可靠性曲线: {path1.relative_to(ROOT_DIR)}")

    # ── 图2: 三届 WC 合并 ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for i, cls in enumerate(LABEL_ORDER):
        y_bin = (np.array(y_wc) == cls).astype(float)
        _draw_reliability(
            axes2[i], y_bin,
            {"DC Raw": p_dc_wc[:, i],
             "DC+Platt": p_platt_wc[:, i],
             "DC+Isotonic": p_iso_wc[:, i]},
            n_bins=5,   # WC样本少(192场)，用5个bin
            title=CLASS_NAMES[cls],
        )
    fig2.suptitle(
        "DC Probability Calibration — Reliability Diagram\n"
        "Test: WC 2014 + 2018 + 2022 combined (n=192, 5 bins)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path2 = OUT_DIR / "dc_calibration_curve_wc.png"
    plt.savefig(path2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  WC回测可靠性曲线:  {path2.relative_to(ROOT_DIR)}")


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Gen2 DC 概率校准")
    print("  方案A: Platt Scaling   方案B: Isotonic Regression")
    print("=" * 62)

    df_raw = pd.read_csv(PROC_DIR / "features.csv", parse_dates=["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # ── Step 1: 拟合校准函数 ──────────────────────────────────
    (platt, iso_regs,
     p_dc_cal, p_platt_cal, p_iso_cal,
     y_cal, cal_metrics, best_method) = fit_calibrators(df_raw)

    # ── Step 2: 三届回测 ──────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  Step 2: 三届回测 (校准函数固定，DC各届独立重训)")
    print(f"{'=' * 62}")

    all_results = {}
    wc_p_raw_list, wc_p_platt_list, wc_p_iso_list, wc_y_list = [], [], [], []

    for year in [2014, 2018, 2022]:
        print(f"\n  [{year}] DC 重训...", end="", flush=True)
        r, p_raw, p_platt, p_iso, y_true = run_calibrated_backtest(
            year, df_raw, platt, iso_regs)
        all_results[year] = r
        wc_p_raw_list.append(p_raw)
        wc_p_platt_list.append(p_platt)
        wc_p_iso_list.append(p_iso)
        wc_y_list.extend(y_true)
        print(f" 完成 ({len(y_true)}场)")

        print(f"  {'方法':<15} {'ACC':>7} {'Brier':>8} {'LogLoss':>9} {'ECE':>7}")
        print(f"  {'-' * 47}")
        for name in ["DC原始", "DC+Platt", "DC+Isotonic"]:
            m = r[name]
            print(f"  {name:<15} {m['acc']:>7.4f} {m['brier']:>8.4f} "
                  f"{m['logloss']:>9.5f} {m['ece']:>7.4f}")

    # ── Step 3: 可靠性曲线 ────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  Step 3: 可靠性曲线绘制")
    print(f"{'=' * 62}")
    p_dc_wc    = np.vstack(wc_p_raw_list)
    p_platt_wc = np.vstack(wc_p_platt_list)
    p_iso_wc   = np.vstack(wc_p_iso_list)
    y_wc       = np.array(wc_y_list)

    plot_calibration_curves(
        p_dc_cal, p_platt_cal, p_iso_cal, y_cal,
        p_dc_wc, p_platt_wc, p_iso_wc, y_wc,
    )

    # ── 汇总对比表 ──────────────────────────────────────────────
    METHODS   = ["DC原始", "DC+Platt", "DC+Isotonic"]
    n_total   = sum(all_results[y]["DC原始"]["n"] for y in [2014, 2018, 2022])
    avg       = {m: {k: float(np.mean([all_results[y][m][k]
                                        for y in [2014, 2018, 2022]]))
                     for k in ["acc", "brier", "logloss", "ece"]}
                 for m in METHODS}

    print(f"\n\n{'#' * 62}")
    print(f"  三届回测汇总对比表  (DC原始 / DC+Platt / DC+Isotonic)")
    print(f"{'#' * 62}")

    for metric_label, key, fmt in [
        ("Brier Score (越低越好)", "brier",   "8.4f"),
        ("LogLoss    (越低越好)", "logloss", "9.5f"),
        ("ECE        (越低越好)", "ece",     "8.4f"),
        ("ACC        (argmax)",  "acc",     "7.4f"),
    ]:
        print(f"\n--- {metric_label} ---")
        hdr = f"{'届':<8} {'n':>4}"
        for m in METHODS:
            hdr += f"  {m:>15}"
        print(hdr)
        print("-" * (13 + 17 * len(METHODS)))
        for year in [2014, 2018, 2022]:
            r   = all_results[year]
            n   = r["DC原始"]["n"]
            row = f"{year:<8} {n:>4}"
            for m in METHODS:
                row += f"  {r[m][key]:>{fmt}}"
            print(row)
        print("-" * (13 + 17 * len(METHODS)))
        row = f"{'均值':<8} {n_total:>4}"
        for m in METHODS:
            row += f"  {avg[m][key]:>{fmt}}"
        print(row)

    # ── 结论 ──────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  [诚实报告] 校准效果评估 (三届WC均值):")
    print(f"  校准集最优方法: DC + {best_method}\n")

    raw_brier = avg["DC原始"]["brier"]
    raw_ll    = avg["DC原始"]["logloss"]
    raw_ece   = avg["DC原始"]["ece"]

    for method in ["DC+Platt", "DC+Isotonic"]:
        db = avg[method]["brier"]   - raw_brier
        dl = avg[method]["logloss"] - raw_ll
        de = avg[method]["ece"]     - raw_ece
        print(f"  {method} vs DC原始 (均值):")
        print(f"    Brier:   {avg[method]['brier']:.4f} vs {raw_brier:.4f}  "
              f"Δ={db:+.4f}  ({'改善' if db < 0 else '未改善'})")
        print(f"    LogLoss: {avg[method]['logloss']:.5f} vs {raw_ll:.5f}  "
              f"Δ={dl:+.5f}  ({'改善' if dl < 0 else '未改善'})")
        print(f"    ECE:     {avg[method]['ece']:.4f} vs {raw_ece:.4f}  "
              f"Δ={de:+.4f}  ({'改善' if de < 0 else '未改善'})")
        print()

    # ── 保存 JSON ─────────────────────────────────────────────
    summary = {
        "calibration_period":    f"{TRAIN_START} ~ {WEIGHT_OPT_END.date()}",
        "best_method_on_calset": f"DC + {best_method}",
        "cal_set_metrics":       cal_metrics,
        "by_year": {
            str(y): {m: all_results[y][m] for m in METHODS}
            for y in [2014, 2018, 2022]
        },
        "avg_3wc": avg,
    }
    out_json = OUT_DIR / "calibration_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  JSON : outputs/gen2/calibration_results.json")
    print(f"  图1  : outputs/gen2/dc_calibration_curve.png")
    print(f"  图2  : outputs/gen2/dc_calibration_curve_wc.png")
    print(f"\n[校准完成] 等待确认")


if __name__ == "__main__":
    main()
