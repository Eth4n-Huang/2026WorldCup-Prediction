"""
阶段3: 模型训练
输入: data/processed/features.csv
输出: outputs/model_lr.pkl, outputs/model_xgb.pkl, outputs/train_params.json
      outputs/calibration_curve.png

模型A: 多分类逻辑回归（基线）
模型B: XGBoost (multi:softprob)
此脚本产出的函数/常量可被 step5_backtest.py 直接导入。
"""
from __future__ import annotations
import json
import pickle
import warnings
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CLASSES     = ["H", "D", "A"]
TRAIN_START = "1998-01-01"
TUNE_END    = "2013-12-31"

META_COLS = frozenset({
    "date", "home_team", "away_team",
    "home_score", "away_score", "result",
    "tournament", "k_factor", "neutral",
    "elo_home_post", "elo_away_post",
})


# ══════════════════════════════════════════════
#  工具函数（可被 step5 导入）
# ══════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def time_decay_weights(dates: pd.Series, ref_date, lam: float) -> np.ndarray:
    """w_i = exp(−λ × 距今年数)，lam=0 → 等权"""
    if lam == 0.0:
        return np.ones(len(dates))
    years_ago = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.days / 365.25
    return np.exp(-lam * years_ago.clip(lower=0).values)


def predict_with_draw_adj(
    probs: np.ndarray,
    delta: float,
    draw_thr: float,
    label_order: list,
) -> np.ndarray:
    """
    平局调整规则：
    |P(H)−P(A)| < delta 且 P(D) > draw_thr → 预测 D，否则 argmax
    delta/draw_thr 在训练期内调，不接触测试届数据
    """
    ih  = label_order.index("H")
    id_ = label_order.index("D")
    ia  = label_order.index("A")
    preds = np.array([label_order[i] for i in np.argmax(probs, axis=1)])
    mask = (np.abs(probs[:, ih] - probs[:, ia]) < delta) & (probs[:, id_] > draw_thr)
    preds[mask] = "D"
    return preds


def tune_draw_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    label_order: list,
    draw_base_rate: float = 0.235,
) -> tuple[float, float]:
    """
    约束调参（v2版）：
    约束: 预测平局比率 ∈ [draw_base_rate ± 3ppt]
    目标: 在满足约束的候选中，选 macro-F1 最高者
    使用训练期平局基率（不得使用测试届数据）
    """
    candidates = []
    for delta in np.arange(0.05, 0.55, 0.05):
        for draw_thr in np.arange(0.15, 0.48, 0.02):
            preds   = predict_with_draw_adj(probs, float(delta), float(draw_thr), label_order)
            pred_dr = (preds == "D").mean()
            if abs(pred_dr - draw_base_rate) > 0.03:
                continue
            mf1 = f1_score(y_true, preds, average="macro", zero_division=0)
            candidates.append((mf1, float(delta), float(draw_thr), pred_dr))

    if candidates:
        candidates.sort(reverse=True)
        _, best_delta, best_draw_thr, best_dr = candidates[0]
        return best_delta, best_draw_thr

    # fallback: 找最接近基率的组合
    best_diff = float("inf")
    best_delta, best_draw_thr = 0.20, 0.30
    for delta in np.arange(0.05, 0.55, 0.05):
        for draw_thr in np.arange(0.15, 0.48, 0.02):
            preds = predict_with_draw_adj(probs, float(delta), float(draw_thr), label_order)
            diff  = abs((preds == "D").mean() - draw_base_rate)
            if diff < best_diff:
                best_diff, best_delta, best_draw_thr = diff, float(delta), float(draw_thr)
    return best_delta, best_draw_thr


class XGBIsotonicCalibrated:
    """XGBClassifier + 逐类别保序回归校准，不依赖 CalibratedClassifierCV"""
    def __init__(self, xgb_model, iso_regs, label_order):
        self.xgb_model   = xgb_model
        self.iso_regs    = iso_regs
        self.label_order = label_order
        self.classes_    = np.array(label_order)

    def predict_proba(self, X) -> np.ndarray:
        raw = self.xgb_model.predict_proba(X)
        cal = np.column_stack([self.iso_regs[i].predict(raw[:, i])
                               for i in range(len(self.label_order))])
        cal = np.clip(cal, 1e-7, 1.0)
        return cal / cal.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return np.array([self.label_order[i]
                         for i in np.argmax(self.predict_proba(X), axis=1)])


def compute_metrics(y_true, y_pred, probs, label_order, tag="") -> dict:
    """准确率 / macro-F1 / Brier / LogLoss"""
    acc    = accuracy_score(y_true, y_pred)
    mf1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
    y_bin  = label_binarize(y_true, classes=label_order)
    if y_bin.shape[1] == len(label_order):
        brier = float(np.mean([brier_score_loss(y_bin[:, i], probs[:, i])
                               for i in range(len(label_order))]))
    else:
        brier = float("nan")
    ll = log_loss(y_true, probs, labels=label_order)
    if tag:
        print(f"  {tag:<26}: acc={acc:.4f} mF1={mf1:.4f} Brier={brier:.4f} LL={ll:.4f}")
    return {"accuracy": acc, "macro_f1": mf1, "brier": brier, "log_loss": ll}


# ══════════════════════════════════════════════
#  核心训练函数（step5 直接调用）
# ══════════════════════════════════════════════

def fit_pipeline(
    df_train: pd.DataFrame,
    model_type: str = "xgb",
    feat_cols: list | None = None,
    lr_C: float = 0.1,
    xgb_max_depth: int = 5,
    xgb_lr: float = 0.1,
    xgb_n_est: int = 200,
    lam: float = 0.1,
    delta: float | None = None,
    draw_thr: float | None = None,
    cal_ratio: float = 0.15,
) -> dict:
    """
    训练 + 时间切割校准 + 约束平局调参，返回可预测的模型包。
    draw_base_rate 从 df_train 自动计算（不接触任何测试届数据）。
    """
    df_train = df_train.sort_values("date").reset_index(drop=True)
    if feat_cols is None:
        feat_cols = get_feature_cols(df_train)

    draw_base_rate = (df_train["result"] == "D").mean()

    n      = len(df_train)
    n_cal  = max(int(n * cal_ratio), 50)
    df_fit = df_train.iloc[: n - n_cal].copy()
    df_cal = df_train.iloc[n - n_cal :].copy()

    X_fit = df_fit[feat_cols].values.astype(np.float32)
    y_fit = df_fit["result"].values
    X_cal = df_cal[feat_cols].values.astype(np.float32)
    y_cal = df_cal["result"].values

    ref   = df_fit["date"].max()
    w_fit = time_decay_weights(df_fit["date"], ref, lam)

    label_order = sorted(np.unique(df_train["result"]))  # ['A','D','H']

    # ── 逻辑回归 ──────────────────────────────
    if model_type == "lr":
        scaler  = StandardScaler()
        X_fit_s = scaler.fit_transform(X_fit)
        X_cal_s = scaler.transform(X_cal)

        base = LogisticRegression(
            C=lr_C, max_iter=1000, random_state=42,
            multi_class="multinomial", solver="lbfgs",
        )
        base.fit(X_fit_s, y_fit, sample_weight=w_fit)
        cal_model = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        cal_model.fit(X_cal_s, y_cal)

        probs_cal = cal_model.predict_proba(X_cal_s)
        lo        = list(cal_model.classes_)
        if delta is None:
            delta, draw_thr = tune_draw_threshold(probs_cal, y_cal, lo, draw_base_rate)

        return dict(
            calibrated_model=cal_model, scaler=scaler,
            feat_cols=feat_cols, label_order=lo,
            delta=delta, draw_thr=draw_thr, lam=lam, model_type="lr",
        )

    # ── XGBoost ───────────────────────────────
    le        = LabelEncoder().fit(label_order)
    y_fit_enc = le.transform(y_fit)

    base_xgb = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(label_order),
        max_depth=xgb_max_depth, learning_rate=xgb_lr, n_estimators=xgb_n_est,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, eval_metric="mlogloss",
    )
    base_xgb.fit(X_fit, y_fit_enc, sample_weight=w_fit)

    raw_cal  = base_xgb.predict_proba(X_cal)
    iso_regs = []
    for i, cls in enumerate(label_order):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_cal[:, i], (y_cal == cls).astype(float))
        iso_regs.append(iso)

    cal_model = XGBIsotonicCalibrated(base_xgb, iso_regs, label_order)
    probs_cal = cal_model.predict_proba(X_cal)
    lo        = label_order
    if delta is None:
        delta, draw_thr = tune_draw_threshold(probs_cal, y_cal, lo, draw_base_rate)

    return dict(
        calibrated_model=cal_model,
        feat_cols=feat_cols, label_order=lo,
        delta=delta, draw_thr=draw_thr, lam=lam,
        label_encoder=le, model_type="xgb",
    )


# ══════════════════════════════════════════════
#  TimeSeriesSplit 超参数搜索
# ══════════════════════════════════════════════

def cv_search(
    df: pd.DataFrame,
    model_type: str,
    param_grid: dict,
    feat_cols: list,
    n_splits: int = 5,
) -> tuple[dict, float]:
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    X      = df[feat_cols].values.astype(np.float32)
    y      = df["result"].values
    dates  = df["date"]
    le     = LabelEncoder().fit(sorted(np.unique(y)))

    keys   = list(param_grid.keys())
    combos = list(iproduct(*[param_grid[k] for k in keys]))
    print(f"  {model_type}: {len(combos)} 参数组合 × {n_splits} 折")

    best_params, best_score = None, -1.0

    for combo in combos:
        params    = dict(zip(keys, combo))
        fold_accs = []
        lam       = params.get("lam", 0.1)

        for tr_idx, te_idx in tscv.split(X):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            d_tr       = dates.iloc[tr_idx]
            w          = time_decay_weights(d_tr, d_tr.max(), lam)
            try:
                if model_type == "lr":
                    sc = StandardScaler()
                    m  = LogisticRegression(
                        C=params["C"], max_iter=300, random_state=42,
                        multi_class="multinomial", solver="lbfgs",
                    )
                    m.fit(sc.fit_transform(X_tr), y_tr, sample_weight=w)
                    preds = m.predict(sc.transform(X_te))
                else:
                    m = xgb.XGBClassifier(
                        objective="multi:softprob", num_class=3,
                        max_depth=params["max_depth"], learning_rate=params["lr"],
                        n_estimators=params["n_est"],
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, verbosity=0,
                    )
                    m.fit(X_tr, le.transform(y_tr), sample_weight=w)
                    preds = le.inverse_transform(m.predict(X_te))
                fold_accs.append(accuracy_score(y_te, preds))
            except Exception:
                fold_accs.append(0.0)

        mean_acc = float(np.mean(fold_accs))
        if mean_acc > best_score:
            best_score  = mean_acc
            best_params = params

    print(f"  最佳参数: {best_params}  CV均值准确率: {best_score:.4f}")
    return best_params, best_score


# ══════════════════════════════════════════════
#  诊断工具函数
# ══════════════════════════════════════════════

def compute_ece(y_true, probs, label_order, n_bins=10) -> float:
    """
    Expected Calibration Error（one-vs-rest 各类别均值）
    ECE = Σ_bin (n_bin/N) × |mean_conf_bin - mean_acc_bin|
    """
    n = len(y_true)
    ece_classes = []
    for i, cls in enumerate(label_order):
        y_bin  = (y_true == cls).astype(float)
        p_cls  = probs[:, i]
        bins   = np.linspace(0, 1, n_bins + 1)
        ece    = 0.0
        for j in range(n_bins):
            mask = (p_cls >= bins[j]) & (p_cls < bins[j + 1])
            if mask.sum() == 0:
                continue
            ece += mask.sum() / n * abs(y_bin[mask].mean() - p_cls[mask].mean())
        ece_classes.append(ece)
    return float(np.mean(ece_classes))


def plot_calibration_curve(y_true, probs_dict, label_order, save_path):
    """
    绘制可靠性曲线（校准图）
    probs_dict: {'模型名': ndarray(N,3)}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, cls in enumerate(label_order):
        ax = axes[i]
        y_bin = (y_true == cls).astype(float)
        for mname, probs in probs_dict.items():
            frac, mean_p = calibration_curve(y_bin, probs[:, i], n_bins=10)
            ax.plot(mean_p, frac, "s-", label=mname)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_title(f"Class {cls}")
        ax.set_xlabel("Mean Predicted Prob")
        ax.set_ylabel("Fraction Positive")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Reliability Curves (Calibration Set, 1998-2013)", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def baseline_elo_wins(df_eval, label_order):
    """
    基线a: Elo高者获胜（含主场+100修正），不预测平局
    同时返回 pseudo-probs（预测类=0.98，其余各0.01），用于Brier/LL计算
    """
    home_eff = df_eval["elo_home_pre"] + 100 * (1 - df_eval["neutral"].fillna(0))
    preds = np.where(home_eff > df_eval["elo_away_pre"], "H", "A")
    tiny  = 0.01
    ih, id_, ia = label_order.index("H"), label_order.index("D"), label_order.index("A")
    pseudo = np.full((len(preds), 3), tiny)
    for k, p in enumerate(preds):
        pseudo[k, label_order.index(p)] = 1.0 - 2 * tiny
    return preds, pseudo


def baseline_random_freq(y_train, n_pred, label_order, seed=42):
    """
    基线b: 按训练期历史频率随机猜（固定seed），
    概率向量 = 历史频率，每场相同
    """
    rng   = np.random.RandomState(seed)
    vals, cnts = np.unique(y_train, return_counts=True)
    freq  = {v: c / cnts.sum() for v, c in zip(vals, cnts)}
    p_arr = np.array([freq.get(c, 0.0) for c in label_order])
    p_arr = p_arr / p_arr.sum()
    preds = rng.choice(label_order, size=n_pred, p=p_arr)
    # 每场的概率向量就是历史频率（不依赖具体比赛）
    probs = np.tile(p_arr, (n_pred, 1))
    return preds, probs


def print_comparison_table(rows, title=""):
    """格式化打印对比表"""
    if title:
        print(f"\n{'='*68}")
        print(f"  {title}")
        print(f"{'='*68}")
    hdr = f"{'方法':<22} {'准确率':>7} {'macroF1':>8} {'Brier':>7} {'LogLoss':>8}"
    print(hdr)
    print("-" * 68)
    for row in rows:
        print(f"{row['name']:<22} {row['accuracy']:>7.4f} {row['macro_f1']:>8.4f} "
              f"{row['brier']:>7.4f} {row['log_loss']:>8.4f}")
    print("-" * 68)


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

if __name__ == "__main__":

    print("读取特征数据...")
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    FEAT_COLS = get_feature_cols(df)
    print(f"特征列数: {len(FEAT_COLS)}")

    df_tune = df[(df["date"] >= TRAIN_START) & (df["date"] <= TUNE_END)].copy()
    draw_base_rate = (df_tune["result"] == "D").mean()
    print(f"调参数据: {len(df_tune)} 行，平局基率: {draw_base_rate:.4f} ({draw_base_rate*100:.1f}%)")

    # ── LR 超参搜索 ────────────────────────────
    print("\n=== 模型A: 逻辑回归 超参搜索 ===")
    lr_grid = {"C": [0.01, 0.1, 1.0], "lam": [0.0, 0.05, 0.10]}
    lr_best, lr_cv = cv_search(df_tune, "lr", lr_grid, FEAT_COLS)

    # ── XGB 超参搜索 ──────────────────────────
    print("\n=== 模型B: XGBoost 超参搜索 ===")
    xgb_grid = {"max_depth": [3, 5, 7], "lr": [0.05, 0.1, 0.2], "n_est": [100, 200, 300]}
    xgb_best, xgb_cv = cv_search(df_tune, "xgb", xgb_grid, FEAT_COLS)
    xgb_best["lam"] = lr_best.get("lam", 0.0)

    # ── 最终模型训练 ──────────────────────────
    print("\n=== 最终模型训练 ===")
    df_train_full = df_tune.copy()
    pkg_lr  = fit_pipeline(df_train_full, "lr",
                           lr_C=lr_best["C"], lam=lr_best["lam"])
    pkg_xgb = fit_pipeline(df_train_full, "xgb",
                           xgb_max_depth=xgb_best["max_depth"],
                           xgb_lr=xgb_best["lr"], xgb_n_est=xgb_best["n_est"],
                           lam=xgb_best["lam"])

    # ── 评估集（末尾15%，时间切割，严禁接触测试届）──────
    n_eval  = max(int(len(df_train_full) * 0.15), 50)
    df_eval = df_train_full.iloc[-n_eval:].copy()
    X_eval  = df_eval[FEAT_COLS].values.astype(np.float32)
    y_eval  = df_eval["result"].values
    print(f"评估集: {len(df_eval)} 行 ({df_eval['date'].min().date()} ~ {df_eval['date'].max().date()})")
    print(f"评估集平局基率: {(y_eval=='D').mean():.4f}")

    lo_lr  = pkg_lr["label_order"]
    lo_xgb = pkg_xgb["label_order"]

    probs_lr  = pkg_lr["calibrated_model"].predict_proba(
                    pkg_lr["scaler"].transform(X_eval))
    probs_xgb = pkg_xgb["calibrated_model"].predict_proba(X_eval)

    # ══════════════════════════════════════════
    # 1. 重调平局阈值（约束版：预测平局比率 ≈ 训练期基率 ±3ppt）
    # ══════════════════════════════════════════
    print(f"\n=== 平局阈值重调（约束: 预测平局率 ∈ [{draw_base_rate-0.03:.3f}, {draw_base_rate+0.03:.3f}]）===")

    delta_lr, dthr_lr = tune_draw_threshold(probs_lr, y_eval, lo_lr, draw_base_rate)
    delta_xgb, dthr_xgb = tune_draw_threshold(probs_xgb, y_eval, lo_xgb, draw_base_rate)

    preds_lr_adj  = predict_with_draw_adj(probs_lr,  delta_lr,  dthr_lr,  lo_lr)
    preds_xgb_adj = predict_with_draw_adj(probs_xgb, delta_xgb, dthr_xgb, lo_xgb)
    preds_lr_raw  = np.array([lo_lr[i]  for i in np.argmax(probs_lr,  axis=1)])
    preds_xgb_raw = np.array([lo_xgb[i] for i in np.argmax(probs_xgb, axis=1)])

    print(f"LR  阈值: δ={delta_lr:.2f}, draw_thr={dthr_lr:.2f}  "
          f"预测D率={( preds_lr_adj=='D').mean():.3f}")
    print(f"XGB 阈值: δ={delta_xgb:.2f}, draw_thr={dthr_xgb:.2f}  "
          f"预测D率={(preds_xgb_adj=='D').mean():.3f}")
    print(f"实际平局率: {(y_eval=='D').mean():.3f}")

    print(f"\n--- 平局预测前后对比 ---")
    print(f"{'模型':<12} {'argmax预D':>9} {'+adj预D':>7} {'实际D':>6}")
    print(f"{'LR':<12} {(preds_lr_raw =='D').sum():>9} {(preds_lr_adj =='D').sum():>7} {(y_eval=='D').sum():>6}")
    print(f"{'XGB':<12} {(preds_xgb_raw=='D').sum():>9} {(preds_xgb_adj=='D').sum():>7} {(y_eval=='D').sum():>6}")

    print(f"\nXGB argmax 混淆矩阵:")
    cm0 = confusion_matrix(y_eval, preds_xgb_raw, labels=CLASSES)
    print(pd.DataFrame(cm0, index=[f"真{c}" for c in CLASSES],
                       columns=[f"预{c}" for c in CLASSES]).to_string())
    print(f"\nXGB +draw_adj 混淆矩阵:")
    cm1 = confusion_matrix(y_eval, preds_xgb_adj, labels=CLASSES)
    print(pd.DataFrame(cm1, index=[f"真{c}" for c in CLASSES],
                       columns=[f"预{c}" for c in CLASSES]).to_string())

    # 更新包中的阈值
    pkg_lr["delta"]   = delta_lr
    pkg_lr["draw_thr"] = dthr_lr
    pkg_xgb["delta"]  = delta_xgb
    pkg_xgb["draw_thr"] = dthr_xgb

    # ══════════════════════════════════════════
    # 2. 基线对比表
    # ══════════════════════════════════════════
    print(f"\n=== 基线对比 ===")
    # 训练期标签分布（基线b用）
    y_train_full = df_train_full["result"].values

    # 基线a: Elo高者胜
    preds_elo, probs_elo = baseline_elo_wins(df_eval, lo_xgb)
    # 基线b: 按历史频率随机猜（seed=42）
    preds_rnd, probs_rnd = baseline_random_freq(y_train_full, len(y_eval), lo_xgb)

    rows = [
        {"name": "基线a Elo高者胜",
         **compute_metrics(y_eval, preds_elo, probs_elo, lo_xgb)},
        {"name": "基线b 历史频率随机",
         **compute_metrics(y_eval, preds_rnd, probs_rnd, lo_xgb)},
        {"name": "LR (argmax)",
         **compute_metrics(y_eval, preds_lr_raw,  probs_lr,  lo_lr)},
        {"name": "LR (+draw adj)",
         **compute_metrics(y_eval, preds_lr_adj,  probs_lr,  lo_lr)},
        {"name": "XGB (argmax)",
         **compute_metrics(y_eval, preds_xgb_raw, probs_xgb, lo_xgb)},
        {"name": "XGB (+draw adj)",
         **compute_metrics(y_eval, preds_xgb_adj, probs_xgb, lo_xgb)},
    ]
    print_comparison_table(rows, "评估集四方对比表（校准集，1998-2013末15%）")

    # ══════════════════════════════════════════
    # 3a. 校准曲线 + ECE
    # ══════════════════════════════════════════
    print(f"\n=== 校准诊断 ===")
    ece_lr  = compute_ece(y_eval, probs_lr,  lo_lr)
    ece_xgb = compute_ece(y_eval, probs_xgb, lo_xgb)
    print(f"ECE (LR ):  {ece_lr:.4f}")
    print(f"ECE (XGB):  {ece_xgb:.4f}")
    print(f"（ECE越接近0越好，<0.05为优秀，<0.10可接受）")

    try:
        cal_img = OUTPUTS_DIR / "calibration_curve.png"
        plot_calibration_curve(
            y_eval,
            {"LR": probs_lr, "XGB": probs_xgb},
            lo_xgb,
            cal_img,
        )
        print(f"校准曲线已保存: {cal_img}")
    except Exception as e:
        print(f"绘图跳过: {e}")

    # ══════════════════════════════════════════
    # 3b. 按赛事类型分组准确率（XGB argmax）
    # ══════════════════════════════════════════
    print(f"\n=== 按赛事类型分组准确率（XGB argmax）===")
    df_eval = df_eval.copy()
    df_eval["pred_xgb"] = preds_xgb_raw

    def type_acc(mask, name):
        sub = df_eval[mask]
        if len(sub) == 0:
            return
        acc = accuracy_score(sub["result"], sub["pred_xgb"])
        dr  = (sub["result"] == "D").mean()
        print(f"  {name:<20}: n={len(sub):5d}  acc={acc:.4f}  平局率={dr:.3f}")

    type_acc(df_eval["is_world_cup"].astype(bool), "世界杯正赛")
    type_acc(df_eval["is_qualifier"].astype(bool), "世界杯预选赛")
    type_acc(df_eval["is_friendly"].astype(bool), "友谊赛")
    other_mask = (~df_eval["is_world_cup"].astype(bool) &
                  ~df_eval["is_qualifier"].astype(bool) &
                  ~df_eval["is_friendly"].astype(bool))
    type_acc(other_mask, "其他（联合会杯等）")

    # ── 时间衰减对比 ──────────────────────────
    print(f"\n=== 时间衰减对比（LR，校准集）===")
    if lr_best["lam"] != 0.0:
        pkg_nodecay = fit_pipeline(df_train_full, "lr", lr_C=lr_best["C"], lam=0.0)
        probs_nd    = pkg_nodecay["calibrated_model"].predict_proba(
                          pkg_nodecay["scaler"].transform(X_eval))
        preds_nd    = np.array([lo_lr[i] for i in np.argmax(probs_nd, axis=1)])
        print(f"  有衰减(λ={lr_best['lam']:.2f}) acc={accuracy_score(y_eval, preds_lr_raw):.4f}")
        print(f"  无衰减(λ=0.00)       acc={accuracy_score(y_eval, preds_nd):.4f}")
    else:
        print(f"  最佳λ={lr_best['lam']:.2f}（无衰减），两者相同，无需对比")

    # ── 特征重要性 ────────────────────────────
    print("\n=== XGB 特征重要性 TOP 15 ===")
    raw_xgb = pkg_xgb["calibrated_model"].xgb_model
    importances = pd.Series(raw_xgb.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    print(importances.head(15).to_string())

    # ── 保存 ──────────────────────────────────
    print("\n=== 保存模型与参数 ===")
    with open(OUTPUTS_DIR / "model_lr.pkl",  "wb") as f:
        pickle.dump(pkg_lr,  f)
    with open(OUTPUTS_DIR / "model_xgb.pkl", "wb") as f:
        pickle.dump(pkg_xgb, f)

    best_params = {
        "lr":  {**lr_best,  "cv_acc": round(lr_cv,  4)},
        "xgb": {**xgb_best, "cv_acc": round(xgb_cv, 4)},
        "feat_cols": FEAT_COLS,
        "train_start": TRAIN_START,
        "tune_end": TUNE_END,
        "draw_base_rate": round(draw_base_rate, 4),
        "delta_lr": delta_lr,   "draw_thr_lr":  dthr_lr,
        "delta_xgb": delta_xgb, "draw_thr_xgb": dthr_xgb,
    }
    with open(OUTPUTS_DIR / "train_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(f"模型已保存: outputs/model_lr.pkl  outputs/model_xgb.pkl")
    print(f"参数已保存: outputs/train_params.json")
