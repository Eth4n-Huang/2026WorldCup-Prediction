"""
步骤3: Elo 超参数优化（合规版 v2）
==============================================
选参准则: 1998-01-01 至 2014-06-11（WC2014开赛前）训练期 BLa rolling LogLoss
  — 严禁在此步骤接触任何 dev 赛事结果
  — dev 集只在参数确认后看一次（由 step6b_elo_update.py 执行）

超参空间: H_adv, K_wc, K_major, K_qual, K_friendly, G_func

附加报告:
  - 默认 vs 最优在训练期的 LogLoss 变化
  - Top10 参数组合
  - 各维度影响分析（仅在训练期上计算）
  - AFCON 2015 分析说明（在 step6b 中报告 dev 效果）

输出: outputs/elo_best_params.json
"""
from __future__ import annotations
import sys, json, time, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent))
from metrics import LABEL_ORDER

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR  = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── 合规边界 ─────────────────────────────────────────────
# 训练期评估范围：热身期结束(1998)至第一个dev赛事(WC2014)前一天
EVAL_START = pd.Timestamp("1998-01-01")
EVAL_END   = pd.Timestamp("2014-06-11")   # WC2014 前一天

DEFAULT_PARAMS = {
    "H_adv": 100, "K_wc": 60, "K_major": 50,
    "K_qual": 40, "K_friendly": 20, "G_func": "original",
}

PARAM_GRID = {
    "H_adv":      [75, 100, 125],
    "K_wc":       [50, 60, 70],
    "K_major":    [40, 50, 60],
    "K_qual":     [30, 40],
    "K_friendly": [15, 20],
    "G_func":     ["original", "none"],
}
# 固定: K_other=30（分类型赛事）


def _compute_g(diff_arr: np.ndarray, g_func: str) -> np.ndarray:
    diff = diff_arr.astype(int)
    if g_func == "none":
        return np.ones(len(diff), dtype=float)
    g = np.ones(len(diff), dtype=float)
    g[diff == 2] = 1.5
    mask3 = diff >= 3
    g[mask3] = (11 + diff[mask3]) / 8.0
    return g


def compute_elo_series(df_clean: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    给定参数重新演算 Elo（Python dict 循环，比 iterrows 快约10x）。
    返回含 elo_home_pre/away_pre/home_post/away_post 的 DataFrame。
    """
    n           = len(df_clean)
    home_arr    = df_clean["home_team"].values
    away_arr    = df_clean["away_team"].values
    result_arr  = df_clean["result"].values
    neutral_arr = df_clean["neutral"].values.astype(bool)
    hs_arr      = df_clean["home_score"].values.astype(int)
    as_arr      = df_clean["away_score"].values.astype(int)
    orig_k_arr  = df_clean["k_factor"].values

    k_arr = np.full(n, 30.0)
    k_arr[orig_k_arr == 60] = params["K_wc"]
    k_arr[orig_k_arr == 50] = params["K_major"]
    k_arr[orig_k_arr == 40] = params["K_qual"]
    k_arr[orig_k_arr == 20] = params["K_friendly"]

    diff_arr = np.abs(hs_arr - as_arr)
    g_arr    = _compute_g(diff_arr, params["G_func"])
    W_MAP    = {"H": 1.0, "D": 0.5, "A": 0.0}
    w_arr    = np.array([W_MAP[r] for r in result_arr])

    H_adv = float(params["H_adv"])
    INIT  = 1500.0
    elo_now = {}
    elo_home_pre  = np.empty(n)
    elo_away_pre  = np.empty(n)
    elo_home_post = np.empty(n)
    elo_away_post = np.empty(n)

    for i in range(n):
        home = home_arr[i]; away = away_arr[i]
        eh = elo_now.get(home, INIT)
        ea = elo_now.get(away, INIT)
        H  = 0.0 if neutral_arr[i] else H_adv
        dr = (eh + H) - ea
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        delta = k_arr[i] * g_arr[i] * (w_arr[i] - we)
        elo_home_pre[i]  = eh
        elo_away_pre[i]  = ea
        elo_now[home]    = eh + delta
        elo_now[away]    = ea - delta
        elo_home_post[i] = elo_now[home]
        elo_away_post[i] = elo_now[away]

    df_out = df_clean.copy()
    df_out["elo_home_pre"]  = elo_home_pre
    df_out["elo_away_pre"]  = elo_away_pre
    df_out["elo_home_post"] = elo_home_post
    df_out["elo_away_post"] = elo_away_post
    return df_out


def eval_train_period_ll(df_clean: pd.DataFrame, params: dict) -> float:
    """
    在训练期（EVAL_START ~ EVAL_END）计算 BLa rolling LogLoss。
    draw_rate 用训练期整体平局率（恒定，对早期比赛是轻微近似，可接受）。
    """
    df_elo = compute_elo_series(df_clean, params)
    mask = (df_elo["date"] >= EVAL_START) & (df_elo["date"] <= EVAL_END)
    df_eval = df_elo[mask]
    if len(df_eval) < 100:
        return np.nan

    draw_rate = float((df_eval["result"] == "D").mean())
    H_adv = float(params["H_adv"])
    neutral_arr = df_eval["neutral"].values.astype(bool)
    H_arr = np.where(neutral_arr, 0.0, H_adv)
    dr_arr = (df_eval["elo_home_pre"].values + H_arr) - df_eval["elo_away_pre"].values
    we_arr = 1.0 / (1.0 + 10.0 ** (-dr_arr / 400.0))

    ph  = we_arr * (1 - draw_rate)
    pa  = (1 - we_arr) * (1 - draw_rate)
    pdd = np.full_like(we_arr, draw_rate)
    probs  = np.stack([pa, pdd, ph], axis=1)   # A, D, H
    y_true = df_eval["result"].values
    return float(log_loss(y_true, probs, labels=LABEL_ORDER))


def run_grid_search(df_clean: pd.DataFrame):
    """
    在训练期 LogLoss 上做网格搜索，不接触 dev 集。
    返回 (df_results, best_params)
    """
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(product(*values))
    total  = len(combos)
    print(f"\n网格搜索: {total} 组参数 × 训练期评估")
    print(f"评估范围: {EVAL_START.date()} ~ {EVAL_END.date()}")
    print("(仅使用训练期数据，不接触任何 dev 赛事)\n")

    t0 = time.time()
    rows = []
    best_ll = np.inf; best_params = None

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        ll = eval_train_period_ll(df_clean, params)
        rows.append({**params, "train_ll": ll})
        if ll < best_ll:
            best_ll = ll; best_params = params
            print(f"  [{idx+1:3d}/{total}] 新最优 LL={ll:.5f}  {params}")
        elif (idx + 1) % 50 == 0:
            eta = (time.time() - t0) / (idx + 1) * (total - idx - 1)
            print(f"  [{idx+1:3d}/{total}] 当前最优 LL={best_ll:.5f}  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成. 耗时 {elapsed:.1f}s")
    return pd.DataFrame(rows).sort_values("train_ll"), best_params


def main():
    print("=" * 60)
    print("  步骤3: Elo 超参优化 — 合规版（训练期选参）")
    print("=" * 60)
    print("\n读取 matches_clean.csv ...")
    df_clean = pd.read_csv(PROC_DIR / "matches_clean.csv", parse_dates=["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)
    n_eval = ((df_clean["date"] >= EVAL_START) &
              (df_clean["date"] <= EVAL_END)).sum()
    print(f"总行数: {len(df_clean)}, 训练期评估行数: {n_eval}")

    # ── 默认参数基线（训练期）──────────────────────────────
    print("\n--- 默认参数 训练期 LogLoss ---")
    ll_default = eval_train_period_ll(df_clean, DEFAULT_PARAMS)
    print(f"  H=100, K_wc=60, K_major=50, K_qual=40, K_friendly=20, G=original")
    print(f"  训练期 LL = {ll_default:.5f}")

    # ── 网格搜索（仅训练期）───────────────────────────────
    df_results, best_params = run_grid_search(df_clean)
    ll_best = df_results.iloc[0]["train_ll"]

    # ── 训练期结果汇报 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  训练期搜索结果（不含任何 dev 信息）")
    print("=" * 60)

    print(f"\n默认参数 训练期 LL: {ll_default:.5f}")
    print(f"最优参数 训练期 LL: {ll_best:.5f}  (Δ={ll_best-ll_default:+.5f})")

    print(f"\n最优参数: {best_params}")

    print("\n--- Top 10 参数组合（按训练期 LL）---")
    print(df_results.head(10)[
        ["H_adv", "K_wc", "K_major", "K_qual", "K_friendly", "G_func", "train_ll"]
    ].to_string(index=False))

    # ── 各维度影响（仅训练期）────────────────────────────
    print("\n--- 各超参维度影响（均值/最优 均为训练期 LL）---")
    for param in PARAM_GRID:
        print(f"\n  {param}:")
        for v in PARAM_GRID[param]:
            sub = df_results[df_results[param] == v]["train_ll"]
            print(f"    {str(v):12s}  mean={sub.mean():.5f}  best={sub.min():.5f}")

    # ── 保存最优参数 ────────────────────────────────────
    out = {
        **best_params,
        "K_other": 30,
        "train_ll_default": round(ll_default, 5),
        "train_ll_best":    round(float(ll_best), 5),
        "delta_train_ll":   round(float(ll_best) - ll_default, 5),
        "note": "合规版: 仅依据1998-2014-06-11训练期LL选参; dev集由step6b_elo_update.py评估"
    }
    with open(OUT_DIR / "elo_best_params.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    df_results.to_csv(OUT_DIR / "elo_opt_results.csv", index=False)
    print(f"\n已保存: outputs/elo_best_params.json, outputs/elo_opt_results.csv")

    print("\n[步骤3 Phase1 完成]")
    print("  → 最优 Elo 参数已由训练期合规选出")
    print("  → 运行 step6b_elo_update.py 以执行: 重放历史/重训XGB/dev配对bootstrap")


if __name__ == "__main__":
    main()
