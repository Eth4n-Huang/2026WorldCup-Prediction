"""
src/gen3/dc_shrinkage.py
========================
第三代 DC 模型：James-Stein 式分层收缩实验

收缩公式：shrunk = (n/(n+k)) × own + (k/(n+k)) × target
  n = 该队在训练集中的出场场次
  k = 收缩强度超参数（网格搜索）
  target = 方案A: 全局均值 / 方案B: 大洲均值（不足4队时退回全局）

严格隔离：
  - 代码在 src/gen3/，输出在 outputs/gen3/
  - 不修改 src/ 或 outputs/ 下任何一代文件
  - k 只在 1998-2013 数据上选取，三届回测共用同一 k
  - 每届回测训练集截止至该届开赛前一天（1998-01-01 起）
"""
from __future__ import annotations
import copy, sys
from pathlib import Path

# ── 路径：只读一代代码 ────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent.parent   # e.g. worldcup/
SRC1  = ROOT / "src"
sys.path.insert(0, str(SRC1))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize

from step5b_improve import DixonColesModel, LABEL_ORDER   # ["A","D","H"]
from step5c_devset   import dc_probs_for_matches, FIXED_HALF_LIFE

PROCESSED = ROOT / "data" / "processed"
OUT_G3    = ROOT / "outputs" / "gen3"
OUT_G3.mkdir(parents=True, exist_ok=True)

TRAIN_START      = pd.Timestamp("1998-01-01")
TUNE_CUTOFF      = pd.Timestamp("2014-01-01")   # k 选取的硬截止
MIN_CONFED_TEAMS = 4                             # 大洲均值最少球队数，不足则退回全局
SPARSE_THRESH    = 10                            # 稀疏球队判定阈值（训练场次 < N）
K_GRID           = [1, 2, 3, 5, 8, 13, 20, 30, 50]   # 收缩强度候选

# 大洲标签（含 OTHER 用于未知归属）
CONFEDS = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "OTHER", "UEFA"]

WC_DATES = {
    2014: (pd.Timestamp("2014-06-12"), pd.Timestamp("2014-07-13")),
    2018: (pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-15")),
    2022: (pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18")),
}


# ══════════════════════════════════════════════════════════════════════════════
#  1. 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_matches() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "matches_clean.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  matches_clean: {len(df)} 行  ({df['date'].min().date()} ~ {df['date'].max().date()})")
    return df


def load_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "features.csv", parse_dates=["date"])


# ══════════════════════════════════════════════════════════════════════════════
#  2. 大洲映射（从 features.csv one-hot 列重建）
# ══════════════════════════════════════════════════════════════════════════════

def build_confed_map(df_features: pd.DataFrame) -> dict[str, str]:
    """
    从 features.csv 的 confed_home_XXX / confed_away_XXX 列
    重建 team → confederation 字符串映射。
    """
    confed_map: dict[str, str] = {}
    for cf in CONFEDS:
        h_col = f"confed_home_{cf}"
        a_col = f"confed_away_{cf}"
        if h_col in df_features.columns:
            mask = df_features[h_col].fillna(0).astype(int) == 1
            for t in df_features.loc[mask, "home_team"].unique():
                confed_map[str(t)] = cf
        if a_col in df_features.columns:
            mask = df_features[a_col].fillna(0).astype(int) == 1
            for t in df_features.loc[mask, "away_team"].unique():
                confed_map[str(t)] = cf

    dist = pd.Series(list(confed_map.values())).value_counts().to_dict()
    print(f"  confed_map: {len(confed_map)} 支球队  分布={dist}")
    return confed_map


# ══════════════════════════════════════════════════════════════════════════════
#  3. 场次统计（用于计算收缩 n）
# ══════════════════════════════════════════════════════════════════════════════

def count_matches_per_team(df: pd.DataFrame) -> dict[str, int]:
    """返回每队在 df 中出现的总场次（主场 + 客场）"""
    counts: dict[str, int] = {}
    for t in df["home_team"].astype(str).values:
        counts[t] = counts.get(t, 0) + 1
    for t in df["away_team"].astype(str).values:
        counts[t] = counts.get(t, 0) + 1
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  4. 收缩核心
# ══════════════════════════════════════════════════════════════════════════════

def apply_shrinkage(
    dc_base: DixonColesModel,
    match_counts: dict[str, int],
    k: float,
    plan: str,                            # "A"(全局) 或 "B"(大洲)
    confed_map: dict[str, str] | None,
) -> DixonColesModel:
    """
    返回新的 DixonColesModel，log_a / log_d 已收缩：
      shrunk = (n/(n+k)) × own_mle + (k/(n+k)) × target_mean
    不修改 dc_base。
    """
    dc_s   = copy.deepcopy(dc_base)
    teams  = list(dc_base.log_a.keys())

    log_a_arr = np.array([dc_base.log_a[t] for t in teams])
    log_d_arr = np.array([dc_base.log_d[t] for t in teams])
    global_mean_a = float(np.mean(log_a_arr))
    global_mean_d = float(np.mean(log_d_arr))

    # 预计算大洲均值（Plan B）
    confed_mean_a: dict[str, float] = {}
    confed_mean_d: dict[str, float] = {}
    if plan == "B" and confed_map:
        all_cf = set(confed_map.get(t, "__none__") for t in teams)
        for cf in all_cf:
            cf_teams = [t for t in teams if confed_map.get(t) == cf]
            if len(cf_teams) >= MIN_CONFED_TEAMS:
                confed_mean_a[cf] = float(np.mean([dc_base.log_a[t] for t in cf_teams]))
                confed_mean_d[cf] = float(np.mean([dc_base.log_d[t] for t in cf_teams]))

    for t in teams:
        n        = float(match_counts.get(t, 0))
        w_own    = n / (n + k)
        w_target = k / (n + k)

        if plan == "A" or not confed_map:
            tgt_a, tgt_d = global_mean_a, global_mean_d
        else:
            cf = confed_map.get(t, "__none__")
            if cf in confed_mean_a:
                tgt_a = confed_mean_a[cf]
                tgt_d = confed_mean_d[cf]
            else:
                tgt_a, tgt_d = global_mean_a, global_mean_d

        dc_s.log_a[t] = w_own * dc_base.log_a[t] + w_target * tgt_a
        dc_s.log_d[t] = w_own * dc_base.log_d[t] + w_target * tgt_d

    # 更新内置均值（供未知球队回退用；取收缩后均值）
    dc_s.avg_log_a = float(np.mean([dc_s.log_a[t] for t in teams]))
    dc_s.avg_log_d = float(np.mean([dc_s.log_d[t] for t in teams]))
    return dc_s


# ══════════════════════════════════════════════════════════════════════════════
#  5. 指标计算
# ══════════════════════════════════════════════════════════════════════════════

def eval_probs(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """probs shape=(N,3) 顺序=[A,D,H]"""
    preds = [LABEL_ORDER[i] for i in np.argmax(probs, axis=1)]
    acc   = accuracy_score(y_true, preds)
    ll    = log_loss(y_true, probs, labels=LABEL_ORDER)
    yb    = label_binarize(y_true, classes=LABEL_ORDER)
    brier = float(np.mean(np.sum((probs - yb) ** 2, axis=1)))
    return {"acc": acc, "log_loss": ll, "brier": brier}


# ══════════════════════════════════════════════════════════════════════════════
#  6. Phase 1: k 调参（只用 TRAIN_START ~ TUNE_CUTOFF）
# ══════════════════════════════════════════════════════════════════════════════

def tune_k(df_matches: pd.DataFrame, confed_map: dict) -> tuple[float, float]:
    """
    在 1998-01-01 ~ 2014-01-01 内，85/15 时间切割选最优 k。
    返回 (best_k_A, best_k_B)，依据：验证集 LogLoss 最小。
    """
    df_tune = df_matches[
        (df_matches["date"] >= TRAIN_START) &
        (df_matches["date"] < TUNE_CUTOFF)
    ].copy()

    n      = len(df_tune)
    n_val  = max(int(n * 0.15), 100)
    df_fit = df_tune.iloc[:n - n_val].copy()
    df_val = df_tune.iloc[n - n_val:].copy()
    ref_date = df_fit["date"].max()

    print(f"\n  k 调参集: fit={len(df_fit)} 场 | val={len(df_val)} 场")
    print(f"  fit 截止: {ref_date.date()}  val 截止: {df_val['date'].max().date()}")

    # 训练基础 DC（在 fit 子集）
    dc_base = DixonColesModel()
    dc_base.fit(df_fit, ref_date, half_life_days=FIXED_HALF_LIFE)
    y_val  = df_val["result"].values
    mc_fit = count_matches_per_team(df_fit)

    best_k_A, best_ll_A = K_GRID[0], float("inf")
    best_k_B, best_ll_B = K_GRID[0], float("inf")

    print(f"\n  {'k':>4}  {'Plan-A LL':>12}  {'Plan-B LL':>12}")
    print("  " + "-" * 35)

    for k in K_GRID:
        dc_A    = apply_shrinkage(dc_base, mc_fit, k, "A", confed_map)
        probs_A = dc_probs_for_matches(dc_A, df_val)
        ll_A    = log_loss(y_val, probs_A, labels=LABEL_ORDER)

        dc_B    = apply_shrinkage(dc_base, mc_fit, k, "B", confed_map)
        probs_B = dc_probs_for_matches(dc_B, df_val)
        ll_B    = log_loss(y_val, probs_B, labels=LABEL_ORDER)

        tag_a = " ←" if ll_A < best_ll_A else ""
        tag_b = " ←" if ll_B < best_ll_B else ""
        print(f"  {k:>4}  {ll_A:>12.6f}{tag_a:<3}  {ll_B:>12.6f}{tag_b}")

        if ll_A < best_ll_A: best_ll_A, best_k_A = ll_A, k
        if ll_B < best_ll_B: best_ll_B, best_k_B = ll_B, k

    print(f"\n  最优 k_A={best_k_A}  (val LL={best_ll_A:.6f})")
    print(f"  最优 k_B={best_k_B}  (val LL={best_ll_B:.6f})")
    return float(best_k_A), float(best_k_B)


# ══════════════════════════════════════════════════════════════════════════════
#  7. Phase 2: 三届 WC 回测
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df_matches: pd.DataFrame,
    confed_map: dict,
    k_A: float,
    k_B: float,
) -> pd.DataFrame:
    """
    对 2014/2018/2022 三届各自：
      1. 训练集 = 1998-01-01 ~ 该届开赛前一天
      2. 测试集 = 该届 FIFA World Cup 场次
      3. 对比 Base DC / Shrunk_A / Shrunk_B
      4. 额外报告稀疏球队子集（训练场次 < SPARSE_THRESH 的队参与的比赛）
    """
    all_rows  = []
    all_dets  = []

    for year, (wc_start, wc_end) in sorted(WC_DATES.items()):
        print(f"\n{'═'*62}")
        print(f"  WC{year}  训练截止={wc_start.date()} - 1天  "
              f"测试={wc_start.date()} ~ {wc_end.date()}")
        print(f"{'═'*62}")

        df_train = df_matches[
            (df_matches["date"] >= TRAIN_START) &
            (df_matches["date"] < wc_start)
        ].copy()
        df_test  = df_matches[
            (df_matches["date"] >= wc_start) &
            (df_matches["date"] <= wc_end) &
            df_matches["tournament"].str.contains("FIFA World Cup", na=False)
        ].sort_values("date").copy()

        if df_test.empty:
            print(f"  ⚠ 未找到 {year} WC 数据，跳过")
            continue

        y_true = df_test["result"].values
        n_test = len(df_test)
        print(f"  训练: {len(df_train)} 场  测试: {n_test} 场")

        mc = count_matches_per_team(df_train)

        # 训练基础 DC
        dc_base = DixonColesModel()
        dc_base.fit(df_train, wc_start, half_life_days=FIXED_HALF_LIFE)
        print(f"  DC fit: γ={np.exp(dc_base.log_gamma):.3f}  ρ={dc_base.rho:.3f}  "
              f"已知球队={len(dc_base.log_a)}")

        # 三种模型预测
        probs_base = dc_probs_for_matches(dc_base,                              df_test)
        probs_A    = dc_probs_for_matches(apply_shrinkage(dc_base, mc, k_A, "A", confed_map), df_test)
        probs_B    = dc_probs_for_matches(apply_shrinkage(dc_base, mc, k_B, "B", confed_map), df_test)

        m_base = eval_probs(probs_base, y_true)
        m_A    = eval_probs(probs_A,    y_true)
        m_B    = eval_probs(probs_B,    y_true)

        # ── 全体结果 ───────────────────────────────────────────────────────
        print(f"\n  全体({n_test}场):")
        print(f"  {'模型':<12} {'ACC':>8} {'Brier':>8} {'LogLoss':>10}")
        print(f"  {'-'*42}")
        for label, m in [("Base_DC", m_base), (f"Shrunk_A(k={k_A})", m_A),
                          (f"Shrunk_B(k={k_B})", m_B)]:
            flag = ""
            if "Shrunk" in label:
                da = m["acc"]   - m_base["acc"]
                db = m["brier"] - m_base["brier"]
                flag = f"  (acc {da:+.4f} / brier {db:+.4f})"
            print(f"  {label:<18} {m['acc']:>8.4f} {m['brier']:>8.4f} {m['log_loss']:>10.4f}{flag}")

        # ── 稀疏球队子集 ───────────────────────────────────────────────────
        sparse_mask = np.array([
            mc.get(r["home_team"], 0) < SPARSE_THRESH or
            mc.get(r["away_team"], 0) < SPARSE_THRESH
            for _, r in df_test.iterrows()
        ])
        n_sparse = int(sparse_mask.sum())
        m_sp_base = m_sp_A = m_sp_B = None
        if n_sparse > 0:
            y_sp      = y_true[sparse_mask]
            m_sp_base = eval_probs(probs_base[sparse_mask], y_sp)
            m_sp_A    = eval_probs(probs_A[sparse_mask],    y_sp)
            m_sp_B    = eval_probs(probs_B[sparse_mask],    y_sp)
            print(f"\n  稀疏子集(n={n_sparse}, 任意一队训练场次<{SPARSE_THRESH}):")
            for label, m in [("Base_DC", m_sp_base), (f"Shrunk_A", m_sp_A),
                              (f"Shrunk_B", m_sp_B)]:
                flag = ""
                if "Shrunk" in label:
                    da = m["acc"]   - m_sp_base["acc"]
                    db = m["brier"] - m_sp_base["brier"]
                    flag = f"  ({da:+.4f} / {db:+.4f})"
                print(f"    {label:<12} ACC={m['acc']:.4f}  Brier={m['brier']:.4f}  LL={m['log_loss']:.4f}{flag}")
        else:
            print(f"\n  [稀疏子集] 本届无训练场次 < {SPARSE_THRESH} 的队")

        # ── 保存明细 CSV ───────────────────────────────────────────────────
        dets = []
        for i, (_, r) in enumerate(df_test.iterrows()):
            n_h = mc.get(r["home_team"], 0)
            n_a = mc.get(r["away_team"], 0)
            dets.append({
                "year": year,
                "date": str(r["date"].date()),
                "home": r["home_team"], "away": r["away_team"],
                "result": r["result"],
                "n_home": n_h, "n_away": n_a,
                "is_sparse": int(n_h < SPARSE_THRESH or n_a < SPARSE_THRESH),
                "base_pa": round(float(probs_base[i,0]), 4),
                "base_pd": round(float(probs_base[i,1]), 4),
                "base_ph": round(float(probs_base[i,2]), 4),
                "base_pred": LABEL_ORDER[int(np.argmax(probs_base[i]))],
                "shrA_pa":  round(float(probs_A[i,0]), 4),
                "shrA_pd":  round(float(probs_A[i,1]), 4),
                "shrA_ph":  round(float(probs_A[i,2]), 4),
                "shrA_pred": LABEL_ORDER[int(np.argmax(probs_A[i]))],
                "shrB_pa":  round(float(probs_B[i,0]), 4),
                "shrB_pd":  round(float(probs_B[i,1]), 4),
                "shrB_ph":  round(float(probs_B[i,2]), 4),
                "shrB_pred": LABEL_ORDER[int(np.argmax(probs_B[i]))],
            })
        all_dets.extend(dets)
        pd.DataFrame(dets).to_csv(
            OUT_G3 / f"backtest_{year}.csv", index=False, encoding="utf-8-sig"
        )

        # ── 汇总行 ────────────────────────────────────────────────────────
        for label, m, m_sp in [
            ("Base_DC",  m_base, m_sp_base),
            ("Shrunk_A", m_A,    m_sp_A),
            ("Shrunk_B", m_B,    m_sp_B),
        ]:
            row = {
                "year": year, "model": label, "n": n_test,
                "acc": m["acc"], "brier": m["brier"], "log_loss": m["log_loss"],
                "n_sparse": n_sparse,
                "acc_sparse":   m_sp["acc"]   if m_sp else np.nan,
                "brier_sparse": m_sp["brier"] if m_sp else np.nan,
            }
            all_rows.append(row)

    # 保存合并明细
    pd.DataFrame(all_dets).to_csv(
        OUT_G3 / "backtest_all.csv", index=False, encoding="utf-8-sig"
    )
    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
#  8. 汇总打印
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, k_A: float, k_B: float):
    models = ["Base_DC", "Shrunk_A", "Shrunk_B"]
    years  = sorted(df["year"].unique().tolist())

    print("\n" + "█" * 68)
    print("  Gen3 DC 分层收缩实验 — 汇总表")
    print(f"  Plan A: 全局均值收缩 k={k_A}")
    print(f"  Plan B: 大洲均值收缩 k={k_B} (< {MIN_CONFED_TEAMS} 队退回全局)")
    print("█" * 68)

    # 逐届
    for y in years:
        sub = df[df["year"] == y]
        n   = int(sub.iloc[0]["n"]) if not sub.empty else 0
        print(f"\n  WC{y} ({n}场):")
        print(f"  {'模型':<12} {'ACC':>8} {'Brier':>8} {'LogLoss':>10}  {'Δ-ACC':>7}  {'Δ-Brier':>9}")
        base_acc   = float(sub[sub["model"]=="Base_DC"]["acc"].values[0])
        base_brier = float(sub[sub["model"]=="Base_DC"]["brier"].values[0])
        for m in models:
            r = sub[sub["model"] == m]
            if r.empty: continue
            r = r.iloc[0]
            da = r["acc"]   - base_acc
            db = r["brier"] - base_brier
            da_s = f"{da:+.4f}" if m != "Base_DC" else "     —"
            db_s = f"{db:+.4f}" if m != "Base_DC" else "       —"
            print(f"  {m:<12} {r['acc']:>8.4f} {r['brier']:>8.4f} {r['log_loss']:>10.4f}  {da_s:>7}  {db_s:>9}")

    # 三届平均
    print(f"\n  三届平均 (共{int(df[df['model']=='Base_DC']['n'].sum())}场):")
    print(f"  {'模型':<12} {'AVG-ACC':>8} {'AVG-Brier':>10} {'AVG-LL':>9}  {'Δ-ACC':>7}  {'Δ-Brier':>9}")
    base_avg_acc   = df[df["model"]=="Base_DC"]["acc"].mean()
    base_avg_brier = df[df["model"]=="Base_DC"]["brier"].mean()
    for m in models:
        sub = df[df["model"] == m]
        a   = sub["acc"].mean()
        b   = sub["brier"].mean()
        ll  = sub["log_loss"].mean()
        da  = a - base_avg_acc
        db  = b - base_avg_brier
        da_s = f"{da:+.4f}" if m != "Base_DC" else "     —"
        db_s = f"{db:+.4f}" if m != "Base_DC" else "       —"
        print(f"  {m:<12} {a:>8.4f} {b:>10.4f} {ll:>9.4f}  {da_s:>7}  {db_s:>9}")

    # 稀疏子集均值
    df_sp = df[df["n_sparse"] > 0].copy()
    if not df_sp.empty:
        print(f"\n  稀疏球队子集平均 (训练场次<{SPARSE_THRESH}):")
        print(f"  {'模型':<12} {'ACC':>8} {'Brier':>8}  {'Δ-ACC':>7}  {'Δ-Brier':>9}")
        b_acc   = df_sp[df_sp["model"]=="Base_DC"]["acc_sparse"].mean()
        b_brier = df_sp[df_sp["model"]=="Base_DC"]["brier_sparse"].mean()
        for m in models:
            sub = df_sp[df_sp["model"] == m]
            if sub.empty or sub["acc_sparse"].isna().all(): continue
            a   = sub["acc_sparse"].mean()
            b   = sub["brier_sparse"].mean()
            da  = a - b_acc
            db  = b - b_brier
            da_s = f"{da:+.4f}" if m != "Base_DC" else "     —"
            db_s = f"{db:+.4f}" if m != "Base_DC" else "       —"
            print(f"  {m:<12} {a:>8.4f} {b:>8.4f}  {da_s:>7}  {db_s:>9}")
    else:
        print(f"\n  [稀疏子集] 三届均无训练场次<{SPARSE_THRESH}的球队")

    print("\n  输出文件:")
    for f in sorted(OUT_G3.iterdir()):
        print(f"    {f.name}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  Gen3 Dixon-Coles 分层收缩实验")
    print(f"  half_life={FIXED_HALF_LIFE}d  k 候选={K_GRID}")
    print("=" * 68)

    df_matches = load_matches()
    df_feats   = load_features()
    confed_map = build_confed_map(df_feats)

    # Phase 1: 调参
    print(f"\n{'='*68}")
    print(f"  Phase 1 — k 调参 ({TRAIN_START.date()} ~ {TUNE_CUTOFF.date()})")
    print(f"{'='*68}")
    k_A, k_B = tune_k(df_matches, confed_map)

    # Phase 2: 回测
    print(f"\n{'='*68}")
    print(f"  Phase 2 — 三届回测 (k_A={k_A}, k_B={k_B})")
    print(f"{'='*68}")
    df_result = run_backtest(df_matches, confed_map, k_A, k_B)

    # 打印汇总
    print_summary(df_result, k_A, k_B)

    # 保存汇总 CSV
    df_result.to_csv(OUT_G3 / "shrinkage_summary.csv", index=False, encoding="utf-8-sig")
    print(f"  汇总 CSV → {OUT_G3 / 'shrinkage_summary.csv'}")


if __name__ == "__main__":
    main()
