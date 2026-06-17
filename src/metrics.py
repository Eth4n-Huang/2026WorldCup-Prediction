"""
metrics.py — 项目唯一指标模块 v1.0 (2026-06-12)

═══════════════════════════════════════════════════════════════
定义
═══════════════════════════════════════════════════════════════

1. multiclass_brier(y_true, probs, label_order)
   = mean_over_N [ sum_{c∈{A,D,H}} (p_c - y_c)^2 ]
   单场范围 [0, 2]；整体值为平均。
   与 step4_train.py / step5c_devset.py 旧版完全一致，无改变。

2. bla_probs_canonical(matches, df_train, label_order) ← 规范版 BLa 概率
   来源：Elo 期望胜率 We 拆分法

   We = 1 / (1 + 10^(-（elo_h_eff − elo_away_pre）/ 400))
   elo_h_eff = elo_home_pre + H_adv * (1 − neutral)
   H_adv = 100（主场优势系数，可通过参数覆盖）

   draw_rate = 训练期平局占比（仅用开赛前数据）

   P(H) = We * (1 − draw_rate)
   P(D) = draw_rate
   P(A) = (1 − We) * (1 − draw_rate)

   ▶ 与旧版 step5_backtest.py 的差异：
     旧版 BLa 使用确定性概率 (0.98 / 0.01 / 0.01)。
     本版改为软概率。
     影响：Brier 值从 ~0.268（WC）变为 ~0.57（WC）——不可直接比较。
     准确率不受影响（argmax 决策相同）。

3. paired_bootstrap_diff(y_true, p1, p2, label_order, metric, n_boot, seed)
   对同一场次集做逐场配对 bootstrap，返回
   (obs_diff, (ci_lo, ci_hi), p_two_sided)

   metric="log_loss": diff = LL(p1) − LL(p2)，负值代表 p1 更好
   metric="accuracy": diff = ACC(p1) − ACC(p2)，正值代表 p1 更好

═══════════════════════════════════════════════════════════════
选用准则（见 final_model_spec.md）
═══════════════════════════════════════════════════════════════
主准则 : 全dev集配对 LogLoss（本模块 paired_bootstrap_diff）
决胜准则: WC+Euro+Copa 子集（342场）准确率与 LogLoss
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.preprocessing import label_binarize

LABEL_ORDER = ["A", "D", "H"]


# ══════════════════════════════════════════════════════════════
#  1. Brier Score
# ══════════════════════════════════════════════════════════════

def multiclass_brier(y_true, probs, label_order=LABEL_ORDER):
    """
    Multiclass Brier = mean( sum_{c} (p_c - y_c)^2 )
    单场范围 [0, 2]，全局值为平均。
    """
    yb = label_binarize(y_true, classes=label_order)
    if yb.shape[1] != len(label_order):
        raise ValueError(f"y_true 中含未知类别；期望 {label_order}")
    return float(np.mean(np.sum((probs - yb) ** 2, axis=1)))


# ══════════════════════════════════════════════════════════════
#  2. BLa 规范概率
# ══════════════════════════════════════════════════════════════

def bla_probs_canonical(matches: pd.DataFrame,
                        df_train: pd.DataFrame,
                        label_order=LABEL_ORDER,
                        h_adv: float = 100.0) -> np.ndarray:
    """
    BLa Elo 软概率。返回 (N, 3) 数组，列顺序 = label_order。

    参数
    ----
    matches : 含 elo_home_pre, elo_away_pre, neutral 列
    df_train: 训练期数据（用于计算历史平局率）
    h_adv   : 主场优势系数（默认100）
    """
    draw_rate = float((df_train["result"] == "D").mean())
    probs = []
    for _, r in matches.iterrows():
        elo_h_eff = r["elo_home_pre"] + h_adv * (1.0 - float(r.get("neutral", 0)))
        elo_a     = r["elo_away_pre"]
        we = 1.0 / (1.0 + 10.0 ** (-(elo_h_eff - elo_a) / 400.0))
        ph = we        * (1.0 - draw_rate)
        pa = (1.0 - we) * (1.0 - draw_rate)
        pd_ = draw_rate
        p_map = {"A": pa, "D": pd_, "H": ph}
        probs.append([p_map[c] for c in label_order])
    return np.array(probs)


def blb_probs_canonical(df_train: pd.DataFrame, n: int,
                        label_order=LABEL_ORDER) -> np.ndarray:
    """BLb: 所有比赛赋相同历史频率概率。"""
    rates = [(df_train["result"] == c).mean() for c in label_order]
    return np.tile(np.array(rates), (n, 1))


# ══════════════════════════════════════════════════════════════
#  3. 模型评估
# ══════════════════════════════════════════════════════════════

def evaluate(y_true, probs, label_order=LABEL_ORDER,
             delta=None, draw_thr=None) -> dict:
    """
    返回综合指标字典:
      accuracy, accuracy_adj, macro_f1, brier, log_loss
    """
    preds = np.array([label_order[i] for i in np.argmax(probs, axis=1)])

    if delta is not None:
        from step4_train import predict_with_draw_adj
        preds_adj = predict_with_draw_adj(probs, delta, draw_thr, label_order)
    else:
        preds_adj = preds

    return {
        "accuracy":     float(accuracy_score(y_true, preds)),
        "accuracy_adj": float(accuracy_score(y_true, preds_adj)),
        "macro_f1":     float(f1_score(y_true, preds, average="macro",
                                        zero_division=0)),
        "brier":        multiclass_brier(y_true, probs, label_order),
        "log_loss":     float(log_loss(y_true, probs, labels=label_order)),
    }


# ══════════════════════════════════════════════════════════════
#  4. 配对 Bootstrap
# ══════════════════════════════════════════════════════════════

def paired_bootstrap_diff(y_true, probs1: np.ndarray, probs2: np.ndarray,
                           label_order=LABEL_ORDER,
                           metric: str = "log_loss",
                           n_boot: int = 10_000,
                           seed: int = 42):
    """
    model1 vs model2 的配对 bootstrap 差值检验。

    参数
    ----
    metric : "log_loss" (负好) 或 "accuracy" (正好)

    返回
    ----
    obs_diff  : 观测差值 metric(p1) − metric(p2)
    (ci_lo, ci_hi) : 95% 置信区间
    p_value   : 双侧 p 值（H0: 差值 = 0）
    """
    y_arr = np.array(y_true)
    n     = len(y_arr)
    rng   = np.random.default_rng(seed)

    def _metric(yt, pp):
        if metric == "log_loss":
            return float(log_loss(yt, pp, labels=label_order))
        elif metric == "accuracy":
            preds = np.array([label_order[i] for i in np.argmax(pp, axis=1)])
            return float(accuracy_score(yt, preds))
        elif metric == "brier":
            return multiclass_brier(yt, pp, label_order)
        else:
            raise ValueError(f"未知 metric: {metric}")

    obs_diff = _metric(y_arr, probs1) - _metric(y_arr, probs2)

    diffs = []
    for _ in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        yt_b   = y_arr[idx]
        p1_b   = probs1[idx]
        p2_b   = probs2[idx]
        diffs.append(_metric(yt_b, p1_b) - _metric(yt_b, p2_b))

    diffs  = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_val  = float(min(np.mean(diffs >= 0), np.mean(diffs <= 0)) * 2)

    return obs_diff, (ci_lo, ci_hi), p_val


def pairwise_comparison_table(y_true, model_probs: dict,
                               label_order=LABEL_ORDER,
                               metric: str = "log_loss",
                               n_boot: int = 5_000):
    """
    对字典中所有模型对做配对比较，返回 DataFrame。
    model_probs: {"模型名": probs_array, ...}
    """
    names  = list(model_probs.keys())
    rows   = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            obs, ci, p = paired_bootstrap_diff(
                y_true, model_probs[n1], model_probs[n2],
                label_order=label_order, metric=metric,
                n_boot=n_boot)
            sig = "✓ 显著" if p < 0.05 else "— 不显著"
            rows.append({
                "模型A": n1, "模型B": n2,
                "差值(A-B)": round(obs, 5),
                "CI_lo": round(ci[0], 5), "CI_hi": round(ci[1], 5),
                "p值": round(p, 3), "结论": sig,
            })
    return pd.DataFrame(rows)
