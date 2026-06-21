"""
src/gen3/draw_calibration.py
============================
平局校准诊断 — 只用三届回测数据(192场)，不改任何模型

诊断项：
  A. 整体：mean(P(D)) vs 实际平局率
  B. P(D) 可靠性曲线：按 P(D) 分桶，对比桶内实际平局率
  C. λ 区间分析：按期望总进球(λ_home+λ_away)分组，预测 vs 实际平局率
  D. 高分平局：ρ 只修正(0-0/1-0/0-1/1-1),检查 >=2-2 平局是否被系统低估
  E. 结论：δ 是"可删除的决策技巧"还是"必要的残差修正"
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC1 = ROOT / "src"
sys.path.insert(0, str(SRC1))

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.preprocessing import label_binarize

from step5b_improve import DixonColesModel, LABEL_ORDER
from step5c_devset   import dc_probs_for_matches, FIXED_HALF_LIFE

PROCESSED = ROOT / "data" / "processed"
OUT_G3    = ROOT / "outputs" / "gen3"
OUT_G3.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("1998-01-01")
WC_DATES = {
    2014: (pd.Timestamp("2014-06-12"), pd.Timestamp("2014-07-13")),
    2018: (pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-15")),
    2022: (pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18")),
}
# 期望总进球分桶边界
LAMBDA_BINS = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 99.0]
LAMBDA_LABELS = ["<1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", ">3.5"]


# ══════════════════════════════════════════════════════════════════════════════
#  λ 向量化计算（与 dc_probs_for_matches 使用相同参数）
# ══════════════════════════════════════════════════════════════════════════════

def compute_lambda_mu(dc: DixonColesModel, df: pd.DataFrame):
    """
    返回 (lam_arr, mu_arr) shape (N,)
    lam = exp(log_a[home] + log_d[away]) * gamma  (非中立场)
    mu  = exp(log_a[away] + log_d[home])
    """
    try:
        from team_names import norm_team
    except ImportError:
        def norm_team(x): return x

    ht  = [norm_team(t) for t in df["home_team"].values]
    at  = [norm_team(t) for t in df["away_team"].values]
    neu = df["neutral"].fillna(0).values.astype(float)

    avg_la, avg_ld = dc.avg_log_a, dc.avg_log_d
    la_h = np.array([dc.log_a.get(t, avg_la) for t in ht])
    ld_h = np.array([dc.log_d.get(t, avg_ld) for t in ht])
    la_a = np.array([dc.log_a.get(t, avg_la) for t in at])
    ld_a = np.array([dc.log_d.get(t, avg_ld) for t in at])

    gamma = np.exp(dc.log_gamma)
    lam   = np.exp(la_h + ld_a) * np.where(neu == 0, gamma, 1.0)
    mu    = np.exp(la_a + ld_h)
    return lam, mu


def dc_prob_score(dc: DixonColesModel, df: pd.DataFrame, max_goals=10):
    """
    返回每场每个比分的概率字典列表，仅用于高分平局分析。
    shape: list of dicts {(hg, ag): prob}
    """
    try:
        from team_names import norm_team
    except ImportError:
        def norm_team(x): return x

    ht  = [norm_team(t) for t in df["home_team"].values]
    at  = [norm_team(t) for t in df["away_team"].values]
    neu = df["neutral"].fillna(0).values.astype(float)

    avg_la, avg_ld = dc.avg_log_a, dc.avg_log_d
    la_h = np.array([dc.log_a.get(t, avg_la) for t in ht])
    ld_h = np.array([dc.log_d.get(t, avg_ld) for t in ht])
    la_a = np.array([dc.log_a.get(t, avg_la) for t in at])
    ld_a = np.array([dc.log_d.get(t, avg_ld) for t in at])

    gamma = np.exp(dc.log_gamma); rho = dc.rho
    lam   = np.exp(la_h + ld_a) * np.where(neu == 0, gamma, 1.0)
    mu    = np.exp(la_a + ld_h)

    goals = np.arange(max_goals + 1, dtype=float)
    lgf   = gammaln(goals + 1)
    p_hg  = np.exp(goals[None,:] * np.log(np.maximum(lam[:,None], 1e-10)) - lam[:,None] - lgf[None,:])
    p_ag  = np.exp(goals[None,:] * np.log(np.maximum(mu[:,None],  1e-10)) - mu[:,None]  - lgf[None,:])

    joint = p_hg[:,:,None] * p_ag[:,None,:]
    joint[:,0,0] *= np.maximum(1 - lam * mu * rho, 0)
    joint[:,1,0] *= np.maximum(1 + mu * rho, 0)
    joint[:,0,1] *= np.maximum(1 + lam * rho, 0)
    joint[:,1,1] *= np.maximum(1 - rho, 0)
    joint  = np.maximum(joint, 0)
    norm   = joint.sum(axis=(1,2), keepdims=True).clip(min=1e-10)
    joint /= norm

    n = len(lam)
    out = []
    for i in range(n):
        d = {}
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                d[(hg, ag)] = float(joint[i, hg, ag])
        out.append(d)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  收集三届回测数据
# ══════════════════════════════════════════════════════════════════════════════

def collect_backtest_data() -> pd.DataFrame:
    """
    重跑三届 DC 预测（只取 base DC），拿到:
      pd_pred  = P(D)
      lam_tot  = λ_home + λ_away
      result   = H/D/A
      home_score, away_score
      年份
    """
    df_hist = pd.read_csv(PROCESSED / "matches_clean.csv", parse_dates=["date"])
    df_hist = df_hist.sort_values("date").reset_index(drop=True)

    records = []
    for year, (wc_start, wc_end) in sorted(WC_DATES.items()):
        print(f"  训练 WC{year} DC ...", end=" ", flush=True)

        df_train = df_hist[
            (df_hist["date"] >= TRAIN_START) &
            (df_hist["date"] < wc_start)
        ].copy()
        df_test  = df_hist[
            (df_hist["date"] >= wc_start) &
            (df_hist["date"] <= wc_end) &
            df_hist["tournament"].str.contains("FIFA World Cup", na=False)
        ].copy().sort_values("date").reset_index(drop=True)

        dc = DixonColesModel()
        dc.fit(df_train, wc_start, half_life_days=FIXED_HALF_LIFE)
        print(f"ρ={dc.rho:.4f}  {len(df_test)}场")

        probs      = dc_probs_for_matches(dc, df_test)   # (N,3) [A,D,H]
        lam, mu    = compute_lambda_mu(dc, df_test)
        score_dists = dc_prob_score(dc, df_test)

        for i, (_, r) in enumerate(df_test.iterrows()):
            records.append({
                "year":       year,
                "date":       r["date"],
                "home":       r["home_team"],
                "away":       r["away_team"],
                "home_score": int(r["home_score"]),
                "away_score": int(r["away_score"]),
                "result":     r["result"],
                "is_draw":    int(r["result"] == "D"),
                "pa":         float(probs[i, 0]),
                "pd":         float(probs[i, 1]),
                "ph":         float(probs[i, 2]),
                "lam":        float(lam[i]),
                "mu":         float(mu[i]),
                "lam_tot":    float(lam[i] + mu[i]),
                "score_dist": score_dists[i],
                # 罗列关键比分的预测概率
                "p_00": score_dists[i].get((0,0), 0),
                "p_11": score_dists[i].get((1,1), 0),
                "p_22": score_dists[i].get((2,2), 0),
                "p_33": score_dists[i].get((3,3), 0),
                "p_draw_hi": sum(
                    score_dists[i].get((k,k), 0)
                    for k in range(2, 11)
                ),  # >= 2-2 的平局概率之和
            })

    df = pd.DataFrame(records)
    df.drop(columns=["score_dist"], inplace=True)   # 不序列化 dict
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  诊断 A: 整体平局校准
# ══════════════════════════════════════════════════════════════════════════════

def diag_overall(df: pd.DataFrame):
    print("\n" + "=" * 68)
    print("  A. 整体平局校准")
    print("=" * 68)

    n = len(df)
    mean_pd     = df["pd"].mean()
    actual_dr   = df["is_draw"].mean()
    gap         = mean_pd - actual_dr

    print(f"  场次: {n}  (三届合并)")
    print(f"  DC 预测平局率 (mean P(D)): {mean_pd:.4f}  ({mean_pd*100:.2f}%)")
    print(f"  实际平局率:                {actual_dr:.4f}  ({actual_dr*100:.2f}%)")
    print(f"  差值 (预测-实际):          {gap:+.4f}  ({gap*100:+.2f}%)")

    by_year = df.groupby("year").agg(
        n=("is_draw","count"),
        pred_dr=("pd","mean"),
        real_dr=("is_draw","mean"),
    ).assign(gap=lambda x: x["pred_dr"] - x["real_dr"])

    print(f"\n  逐届:")
    print(f"  {'届次':>8} {'场次':>5} {'预测DR':>9} {'实际DR':>9} {'差值':>9}")
    for y, row in by_year.iterrows():
        print(f"  {'WC'+str(y):>8} {int(row['n']):>5} {row['pred_dr']:>9.4f} {row['real_dr']:>9.4f} {row['gap']:>+9.4f}")

    if abs(gap) < 0.01:
        verdict = "整体基本校准 (差距 < 1 ppt)"
    elif gap > 0:
        verdict = f"DC 系统性高估平局率 (高估 {gap*100:.1f} ppt)"
    else:
        verdict = f"DC 系统性低估平局率 (低估 {abs(gap)*100:.1f} ppt)"
    print(f"\n  --> {verdict}")
    return mean_pd, actual_dr


# ══════════════════════════════════════════════════════════════════════════════
#  诊断 B: P(D) 可靠性曲线
# ══════════════════════════════════════════════════════════════════════════════

def diag_reliability(df: pd.DataFrame):
    print("\n" + "=" * 68)
    print("  B. P(D) 可靠性曲线 (按 P(D) 分桶)")
    print("=" * 68)

    bins   = np.arange(0.0, 0.55, 0.05)   # [0,0.05), [0.05,0.10), ..., [0.50+)
    labels = [f"{b:.2f}-{b+0.05:.2f}" for b in bins]
    df2    = df.copy()
    df2["pd_bin"] = pd.cut(df2["pd"], bins=list(bins) + [1.0],
                            labels=labels, right=False)

    print(f"  {'P(D)桶':>12} {'场次':>5} {'预测均值':>10} {'实际平局率':>10} {'差值':>9}")
    print(f"  {'-'*52}")
    rows_rel = []
    for lab in labels:
        sub = df2[df2["pd_bin"] == lab]
        if len(sub) == 0:
            continue
        pred_mean = sub["pd"].mean()
        real_mean = sub["is_draw"].mean()
        gap       = pred_mean - real_mean
        flag      = " <--" if abs(gap) > 0.05 else ""
        print(f"  {lab:>12} {len(sub):>5} {pred_mean:>10.4f} {real_mean:>10.4f} {gap:>+9.4f}{flag}")
        rows_rel.append({"bin": lab, "n": len(sub), "pred": pred_mean, "actual": real_mean})

    rel_df = pd.DataFrame(rows_rel)
    # 整体偏差：加权 MAE
    weights  = rel_df["n"].values.astype(float)
    mae_rel  = float(np.average(np.abs(rel_df["pred"] - rel_df["actual"]), weights=weights))
    print(f"\n  加权 MAE (校准误差): {mae_rel:.4f}")

    if mae_rel < 0.03:
        verdict = "P(D) 校准良好 (MAE < 3%)"
    elif mae_rel < 0.07:
        verdict = "P(D) 存在中等校准偏差 (MAE 3-7%)"
    else:
        verdict = "P(D) 校准偏差较大 (MAE > 7%)"
    print(f"  --> {verdict}")
    return rel_df


# ══════════════════════════════════════════════════════════════════════════════
#  诊断 C: λ 区间分析
# ══════════════════════════════════════════════════════════════════════════════

def diag_lambda_buckets(df: pd.DataFrame):
    print("\n" + "=" * 68)
    print("  C. 按期望总进球 (λ_home + λ_away) 分组")
    print("=" * 68)

    df2 = df.copy()
    df2["lam_bucket"] = pd.cut(df2["lam_tot"], bins=LAMBDA_BINS,
                                labels=LAMBDA_LABELS, right=False)

    print(f"  {'λ桶':>10} {'场次':>5} {'λ均值':>8} {'均P(D)':>8} {'实平率':>8} {'差值':>8}")
    print(f"  {'-'*54}")
    rows_lam = []
    for lab in LAMBDA_LABELS:
        sub = df2[df2["lam_bucket"] == lab]
        if len(sub) == 0:
            continue
        avg_lam   = sub["lam_tot"].mean()
        mean_pd   = sub["pd"].mean()
        real_dr   = sub["is_draw"].mean()
        gap       = mean_pd - real_dr
        flag      = " <--" if gap < -0.04 else ("" if gap > -0.02 else " .")
        n_act_d   = sub["is_draw"].sum()
        print(f"  {lab:>10} {len(sub):>5} {avg_lam:>8.3f} {mean_pd:>8.4f} {real_dr:>8.4f} {gap:>+8.4f}{flag}"
              f"  [{n_act_d}场实平]")
        rows_lam.append({"bucket": lab, "n": len(sub),
                          "lam_avg": avg_lam, "pred": mean_pd, "actual": real_dr})

    lam_df = pd.DataFrame(rows_lam)
    # 找低估最严重的区间
    lam_df["gap"] = lam_df["pred"] - lam_df["actual"]
    worst = lam_df.sort_values("gap").iloc[0]
    print(f"\n  低估最大区间: λ={worst['bucket']}  ΔDR={worst['gap']:+.4f}  (n={worst['n']})")
    return lam_df


# ══════════════════════════════════════════════════════════════════════════════
#  诊断 D: 高分平局分析 (ρ 覆盖范围外)
# ══════════════════════════════════════════════════════════════════════════════

def diag_high_score_draws(df: pd.DataFrame):
    print("\n" + "=" * 68)
    print("  D. 高分平局分析 (ρ 只修正 0-0/1-0/0-1/1-1)")
    print("=" * 68)

    # 各比分平局统计
    print(f"\n  实际平局比分分布 (三届192场):")
    draws = df[df["is_draw"] == 1].copy()
    draws["score"] = draws["home_score"].astype(str) + "-" + draws["away_score"].astype(str)
    score_counts = draws["score"].value_counts().sort_index()
    n_draws = len(draws)
    print(f"  共 {n_draws} 场平局  (实际平局率 {n_draws/len(df)*100:.1f}%)")
    print(f"\n  {'比分':>8}  {'场次':>5}  {'占总平局%':>10}  {'DC预测P(该比分)均值':>20}  ρ覆盖")
    for score, cnt in score_counts.items():
        h, a = score.split("-")
        h, a = int(h), int(a)
        rho_covered = "(yes)" if (h <= 1 and a <= 1) else "(no) "
        sub_matches = df[(df["home_score"] == h) & (df["away_score"] == a)]
        avg_pred_p  = sub_matches[f"p_{h}{a}"].mean() if f"p_{h}{a}" in df.columns else np.nan
        # fallback if not in preset columns
        if f"p_{h}{a}" not in df.columns or np.isnan(avg_pred_p):
            avg_pred_p = np.nan
        print(f"  {score:>8}  {cnt:>5}  {cnt/n_draws*100:>10.1f}%"
              f"  {avg_pred_p:>20.4f}  {rho_covered}")

    # 低分平局 vs 高分平局
    low_draw_actual   = len(df[(df["is_draw"]==1) & (df["home_score"] <= 1)]) / len(df)
    high_draw_actual  = len(df[(df["is_draw"]==1) & (df["home_score"] >= 2)]) / len(df)
    low_draw_pred     = (df["p_00"] + df["p_11"]).mean()
    high_draw_pred    = df["p_draw_hi"].mean()

    print(f"\n  低分平局 (0-0 + 1-1):  预测总率={low_draw_pred:.4f}  实际总率={low_draw_actual:.4f}"
          f"  差={low_draw_pred-low_draw_actual:+.4f}  (ρ 可修正)")
    print(f"  高分平局 (>=2-2):      预测总率={high_draw_pred:.4f}  实际总率={high_draw_actual:.4f}"
          f"  差={high_draw_pred-high_draw_actual:+.4f}  (ρ 无法修正)")

    gap_low  = low_draw_pred  - low_draw_actual
    gap_high = high_draw_pred - high_draw_actual

    if gap_high < -0.01:
        hi_verdict = f"ρ 覆盖范围外(>=2-2)存在系统性低估 ({gap_high*100:.1f} ppt)"
    elif gap_high > 0.01:
        hi_verdict = f"ρ 覆盖范围外(>=2-2)略有高估 (+{gap_high*100:.1f} ppt)"
    else:
        hi_verdict = f"ρ 覆盖范围外(>=2-2)基本校准 (差距 < 1 ppt)"
    print(f"\n  --> 高分平局: {hi_verdict}")
    print(f"  --> 低分平局: 差值 {gap_low*100:+.1f} ppt，ρ 在起作用但{'略有余量' if gap_low > 0.005 else '已基本对齐'}")

    return gap_low, gap_high


# ══════════════════════════════════════════════════════════════════════════════
#  诊断 E: ρ 的实际贡献估算
# ══════════════════════════════════════════════════════════════════════════════

def diag_rho_contribution(df: pd.DataFrame):
    """
    显示 ρ 在 0-0/1-1 的具体提升量。
    由于无法轻松运行"ρ=0的模型"，这里用解析近似估算。
    近似：P_nocorrect(0,0) ≈ p_00 / (1 - lam*mu*rho)，等等。
    """
    print("\n" + "=" * 68)
    print("  E. ρ 对平局概率的实际贡献估算")
    print("=" * 68)

    # 这届各比赛 ρ 值已知（从 dc.rho），这里只有概率，无法还原 lam/mu
    # 改为：显示 ρ < 0 对整体 P(D) 的理论方向
    # p(0-0) 被提升 (ρ<0 → tau00 > 1)
    # p(1-1) 被提升
    # p(1-0), p(0-1) 被压低
    # 净效应对 P(D) 取决于具体 lam/mu

    # 用三届回测的 λ 统计来估算 ρ 贡献
    # tau_00 = 1 - lam*mu*rho, 典型值: lam≈1.5,mu≈1.0,rho≈-0.06 → tau_00=1+0.09=1.09
    # 所以 p(0-0) 被提升约 9%（相对比例）
    # tau_11 = 1 - rho = 1+0.06 → p(1-1) 被提升约 6%

    avg_lam = df["lam"].mean()
    avg_mu  = df["mu"].mean()

    # 从回测数据中读 ρ：这里没有保存，只能说"典型值约-0.06"
    # 但我们可以从各届 DC fit 中读到, 这里简化报告
    p_00_mean = df["p_00"].mean()
    p_11_mean = df["p_11"].mean()
    p_22_mean = df["p_22"].mean()
    p_33_mean = df["p_33"].mean()

    print(f"  三届平均 λ_home={avg_lam:.3f}  μ_away={avg_mu:.3f}")
    print(f"  (提示: 各届 ρ ≈ -0.06 ~ -0.07, 即 ρ<0 → τ(0-0)>1, τ(1-1)>1)")
    print()
    print(f"  各关键比分的DC预测概率均值:")
    print(f"  {'比分':>6}  {'预测P均值':>10}  {'ρ覆盖':>8}  {'理论效应'}")
    print(f"  {'─'*54}")
    for score, p_col, rho_tag, effect in [
        ("0-0", "p_00", "yes (+)",  "τ=1-λμρ>1 → 提升"),
        ("1-1", "p_11", "yes (+)",  "τ=1-ρ>1   → 提升"),
        ("2-2", "p_22", "no",       "τ=1 (无修正)"),
        ("3-3", "p_33", "no",       "τ=1 (无修正)"),
    ]:
        pv = df[p_col].mean()
        print(f"  {score:>6}  {pv:>10.4f}  {rho_tag:>8}  {effect}")

    print(f"\n  ρ 修正的 4 个格子合计 P(0-0)+P(1-1)  均值: {p_00_mean+p_11_mean:.4f}")
    print(f"  ρ 未修正的 >=2-2 合计 P_draw_hi       均值: {df['p_draw_hi'].mean():.4f}")
    pct_hi = df["p_draw_hi"].mean() / df["pd"].mean() * 100
    print(f"  高分平局占 P(D) 的比例: {pct_hi:.1f}%")
    if pct_hi > 25:
        print(f"  --> 高分平局贡献了 P(D) 的 {pct_hi:.0f}%，是不可忽视的组成部分")
    else:
        print(f"  --> 高分平局贡献相对小，主要平局校准来自低分格子")


# ══════════════════════════════════════════════════════════════════════════════
#  保存诊断数据 & 最终结论
# ══════════════════════════════════════════════════════════════════════════════

def final_verdict(mean_pd, actual_dr, gap_low, gap_high, lam_df: pd.DataFrame):
    worst_lam_gap = float(lam_df["gap"].min())

    print("\n" + "#" * 68)
    print("  最终结论：δ 是「决策技巧」还是「必要残差修正」")
    print("#" * 68)

    overall_gap = mean_pd - actual_dr
    overall_ok  = abs(overall_gap) < 0.01
    high_ok     = abs(gap_high)    < 0.01
    lam_ok      = worst_lam_gap    > -0.05

    print(f"\n  1. 整体平局率: 预测 {mean_pd:.4f} vs 实际 {actual_dr:.4f} → {'+' if overall_gap>0 else ''}{overall_gap:.4f}")
    print(f"     {'[校准良好]' if overall_ok else '[存在偏差]'}")
    print(f"\n  2. 高分平局(>=2-2): 预测-实际 = {gap_high:+.4f}")
    print(f"     {'[ρ范围外基本校准]' if high_ok else '[ρ范围外有偏差，ρ不能解决]'}")
    print(f"\n  3. λ 最差区间差值: {worst_lam_gap:+.4f}")
    print(f"     {'[各λ区间无系统性偏差]' if lam_ok else '[存在λ区间级别的系统性低估]'}")

    print(f"\n  综合判断:")
    if overall_ok and high_ok and lam_ok:
        print("  ★ DC 平局概率基本校准，δ 更接近「可选的决策调优技巧」")
        print("    理由：P(D) 的均值和分布与实际相符，argmax 不输出 D 是因为")
        print("    平局很少是三类中的最高概率（结构问题），而不是 P(D) 被低估。")
        print("    → 若去掉 δ，概率质量损失不大，准确率可能因少预测假阳性平局而上升。")
    elif not overall_ok and overall_gap < 0:
        print("  ★ DC 系统性低估 P(D)，δ 是必要的残差修正")
        print("    理由：mean P(D) 明显低于实际平局率，argmax 几乎不输出 D 是因为")
        print("    模型概率本身就低，修正 δ 有实质意义。")
    elif not high_ok and gap_high < 0:
        print("  ★ ρ 修正范围外(>=2-2)存在低估，这部分 δ 有补残差价值")
        print("    但整体偏差不大，δ 的收益有限——主要应在论文中如实说明 ρ 的局限性。")
    else:
        print("  ★ 情况复杂，δ 的必要性依赖于具体使用场景：")
        print("    - 若目标是概率输出(Brier/LL)：δ 贡献有限，可考虑去除")
        print("    - 若目标是点预测ACC：δ 可能通过多预测几场平局略微提升")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  DC 平局校准诊断 (192场 WC 三届回测)")
    print("=" * 68)

    # 1. 数据收集
    print("\n  收集三届回测数据...")
    df = collect_backtest_data()
    print(f"  总场次: {len(df)}  实际平局: {df['is_draw'].sum()} 场")

    # 2. 四项诊断
    mean_pd, actual_dr = diag_overall(df)
    rel_df             = diag_reliability(df)
    lam_df             = diag_lambda_buckets(df)
    gap_low, gap_high  = diag_high_score_draws(df)
    diag_rho_contribution(df)

    # 3. 最终结论
    final_verdict(mean_pd, actual_dr, gap_low, gap_high, lam_df)

    # 4. 保存
    df_save = df.drop(columns=["lam", "mu"], errors="ignore")
    df_save.to_csv(OUT_G3 / "draw_calib_detail.csv", index=False, encoding="utf-8-sig")
    rel_df.to_csv(OUT_G3 / "draw_calib_reliability.csv", index=False, encoding="utf-8-sig")
    lam_df.to_csv(OUT_G3 / "draw_calib_lambda.csv", index=False, encoding="utf-8-sig")
    print(f"  输出: outputs/gen3/draw_calib_*.csv")


if __name__ == "__main__":
    main()
