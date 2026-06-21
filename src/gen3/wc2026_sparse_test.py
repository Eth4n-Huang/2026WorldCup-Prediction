"""
src/gen3/wc2026_sparse_test.py
==============================
2026 WC 稀疏球队诊断测试 — 原始 DC vs Shrunk_B (大洲收缩, k=8)

【防泄漏】
  - DC/Shrunk_B 参数全部由 2026-06-10 之前的历史数据估计
  - wc2026_results.csv 只用于取「真实结果」和「比赛元数据」(home/away/neutral)
  - 2026 已赛比分绝不进入参数估计
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
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize

from step5b_improve import DixonColesModel, LABEL_ORDER
from step5c_devset   import dc_probs_for_matches, FIXED_HALF_LIFE
from team_names      import norm_team          # 统一队名标准化
from gen3.dc_shrinkage import (
    build_confed_map, count_matches_per_team,
    apply_shrinkage, eval_probs,
    PROCESSED, OUT_G3, TRAIN_START
)

# ── 常量 ──────────────────────────────────────────────────────────────────────
CUTOFF_DATE  = pd.Timestamp("2026-06-11")   # 严格截止：开赛前
K_SHRINK     = 8                             # 已在训练期选定
SPARSE_THRESH    = 10    # 严格稀疏阈值
MODERATE_THRESH  = 50    # 宽松阈值：相对历史较少的队

# 2026 WC 球队大洲补充映射（覆盖 features.csv 可能未收录的新队）
CONFED_SUPPLEMENT: dict[str, str] = {
    # CONCACAF
    "Curaçao": "CONCACAF", "Haiti": "CONCACAF",
    "Panama": "CONCACAF", "Jamaica": "CONCACAF",
    "Canada": "CONCACAF", "United States": "CONCACAF",
    "Mexico": "CONCACAF", "Costa Rica": "CONCACAF",
    "Honduras": "CONCACAF", "El Salvador": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF",
    # CAF
    "Cape Verde": "CAF", "DR Congo": "CAF",
    "Ghana": "CAF", "Morocco": "CAF", "Senegal": "CAF",
    "Algeria": "CAF", "Egypt": "CAF", "Ivory Coast": "CAF",
    "Tunisia": "CAF", "South Africa": "CAF", "Nigeria": "CAF",
    "Cameroon": "CAF", "Uganda": "CAF", "Zambia": "CAF",
    "Benin": "CAF", "Guinea": "CAF", "Tanzania": "CAF",
    "Comoros": "CAF", "Kenya": "CAF",
    # AFC
    "Uzbekistan": "AFC", "Jordan": "AFC", "Iraq": "AFC",
    "Iran": "AFC", "Japan": "AFC", "South Korea": "AFC",
    "Saudi Arabia": "AFC", "Australia": "AFC",
    "Indonesia": "AFC", "Qatar": "AFC",
    "China PR": "AFC", "Tajikistan": "AFC", "Oman": "AFC",
    "United Arab Emirates": "AFC", "Kuwait": "AFC",
    "Kyrgyzstan": "AFC", "Vietnam": "AFC", "Thailand": "AFC",
    # UEFA
    "Czech Republic": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Netherlands": "UEFA", "Sweden": "UEFA", "Germany": "UEFA",
    "Belgium": "UEFA", "Spain": "UEFA", "Austria": "UEFA",
    "England": "UEFA", "Croatia": "UEFA", "Portugal": "UEFA",
    "France": "UEFA", "Scotland": "UEFA", "Norway": "UEFA",
    "Switzerland": "UEFA", "Turkey": "UEFA",
    "Serbia": "UEFA", "Hungary": "UEFA", "Romania": "UEFA",
    "Albania": "UEFA", "Slovenia": "UEFA",
    "Georgia": "UEFA", "Slovakia": "UEFA",
    "Denmark": "UEFA", "Poland": "UEFA", "Ukraine": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Peru": "CONMEBOL",
    "Venezuela": "CONMEBOL", "Bolivia": "CONMEBOL",
    # OFC
    "New Zealand": "OFC", "Fiji": "OFC",
    "Papua New Guinea": "OFC", "Solomon Islands": "OFC",
}


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  2026 WC 稀疏球队诊断: 原始 DC vs Shrunk_B (k=8)")
    print(f"  参数截止: {CUTOFF_DATE.date()}  稀疏阈值: <{SPARSE_THRESH} 场")
    print("=" * 68)

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    df_hist  = pd.read_csv(PROCESSED / "matches_clean.csv", parse_dates=["date"])
    df_feats = pd.read_csv(PROCESSED / "features.csv", parse_dates=["date"])
    df_wc26  = pd.read_csv(ROOT / "data" / "processed" / "wc2026_results.csv",
                            parse_dates=["date"])

    df_hist = df_hist.sort_values("date").reset_index(drop=True)

    # 2026 WC 已结算场次（有 result 的行）
    df_settled = df_wc26[
        df_wc26["result"].astype(str).isin(["H", "D", "A"])
    ].copy().sort_values("date").reset_index(drop=True)

    print(f"\n  历史场次(全部): {len(df_hist)} 场")
    print(f"  历史场次(至截止日前): ", end="")
    n_pre = (df_hist["date"] < CUTOFF_DATE).sum()
    print(f"{n_pre} 场")
    print(f"  2026 WC 已结算: {len(df_settled)} 场")

    # ── 2. 训练集（严格截止于开赛前）────────────────────────────────────────
    df_train = df_hist[
        (df_hist["date"] >= TRAIN_START) &
        (df_hist["date"] < CUTOFF_DATE)
    ].copy()
    print(f"  DC 训练集: {len(df_train)} 场  ({df_train['date'].min().date()} ~ {df_train['date'].max().date()})")

    # ── 3. 大洲映射（features.csv + 补充）──────────────────────────────────
    confed_map = build_confed_map(df_feats)
    # 叠加补充映射（不覆盖已有记录，只补空缺）
    for team, cf in CONFED_SUPPLEMENT.items():
        confed_map.setdefault(team, cf)
    print(f"  confed_map: {len(confed_map)} 队（含补充）")

    # ── 4. 统计每队训练场次 ──────────────────────────────────────────────────
    mc = count_matches_per_team(df_train)

    # 2026 WC 所有参赛队
    wc26_teams = sorted(set(df_settled["home_team"]) | set(df_settled["away_team"]))
    print(f"\n  2026 WC 参赛队 ({len(wc26_teams)} 支，截至当前结算):")

    # 用 norm_team 做标准化匹配，显示时保留原始名
    def n_for(t): return mc.get(norm_team(t), 0)

    sparse_teams   = []   # < SPARSE_THRESH
    moderate_teams = []   # SPARSE_THRESH <= n < MODERATE_THRESH
    non_sparse     = []   # >= MODERATE_THRESH
    for t in wc26_teams:
        n = n_for(t)
        if n < SPARSE_THRESH:
            sparse_teams.append(t)
        elif n < MODERATE_THRESH:
            moderate_teams.append(t)
        else:
            non_sparse.append(t)

    print(f"\n  >> 严格稀疏 (训练场次 < {SPARSE_THRESH}):")
    if sparse_teams:
        for t in sparse_teams:
            nt = norm_team(t)
            n  = mc.get(nt, 0)
            cf = confed_map.get(nt, confed_map.get(t, "UNKNOWN"))
            note = f"  [标准名: {nt}]" if nt != t else ""
            print(f"      {t:<28}  n={n:>3}  大洲={cf}{note}")
    else:
        print("      (无) -- 所有参赛队在历史数据中均有 >= 10 场记录")

    print(f"\n  >> 相对稀疏 ({SPARSE_THRESH} <= 场次 < {MODERATE_THRESH}):")
    if moderate_teams:
        for t in sorted(moderate_teams):
            nt = norm_team(t)
            n  = mc.get(nt, 0)
            cf = confed_map.get(nt, confed_map.get(t, "UNKNOWN"))
            note = f"  [标准名: {nt}]" if nt != t else ""
            print(f"      {t:<28}  n={n:>3}  大洲={cf}{note}")
    else:
        print("      (无)")

    print(f"\n  >> 历史充足 (场次 >= {MODERATE_THRESH}, {len(non_sparse)} 支):")
    for t in sorted(non_sparse):
        nt = norm_team(t)
        n  = mc.get(nt, 0)
        cf = confed_map.get(nt, confed_map.get(t, "UNKNOWN"))
        print(f"      {t:<28}  n={n:>3}  大洲={cf}")

    # ── 5. 训练 DC 模型 ──────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  训练 DC 模型 (half_life=730d)...")
    dc_base = DixonColesModel()
    dc_base.fit(df_train, CUTOFF_DATE, half_life_days=FIXED_HALF_LIFE)
    print(f"  DC fit: γ={np.exp(dc_base.log_gamma):.3f}  ρ={dc_base.rho:.3f}  已知队={len(dc_base.log_a)}")

    # Shrunk_B
    dc_shrunk = apply_shrinkage(dc_base, mc, K_SHRINK, "B", confed_map)
    print(f"  Shrunk_B (k={K_SHRINK}) 生成完毕")

    # ── 6. 对全部 32 场生成预测 ──────────────────────────────────────────────
    probs_base   = dc_probs_for_matches(dc_base,   df_settled)   # (N,3) [A,D,H]
    probs_shrunk = dc_probs_for_matches(dc_shrunk, df_settled)
    y_true = df_settled["result"].values

    # ── 7. 全体指标 ──────────────────────────────────────────────────────────
    m_base   = eval_probs(probs_base,   y_true)
    m_shrunk = eval_probs(probs_shrunk, y_true)
    n_total  = len(df_settled)

    print(f"\n{'─'*68}")
    print(f"  全体 {n_total} 场对比:")
    print(f"  {'模型':<14} {'ACC':>8} {'Brier':>8} {'LogLoss':>10}")
    print(f"  {'─'*44}")
    for label, m in [("Base DC", m_base), (f"Shrunk_B(k={K_SHRINK})", m_shrunk)]:
        print(f"  {label:<14} {m['acc']:>8.4f} {m['brier']:>8.4f} {m['log_loss']:>10.4f}")
    da = m_shrunk["acc"]   - m_base["acc"]
    db = m_shrunk["brier"] - m_base["brier"]
    print(f"  {'Δ(收缩-原始)':<14} {da:>+8.4f} {db:>+8.4f}")
    print(f"  （注：样本仅 {n_total} 场，结论以定性参考为主）")

    # ── 8. 含稀疏/中度稀疏球队的比赛逐场对比 ────────────────────────────────
    sparse_set   = set(sparse_teams)
    moderate_set = set(moderate_teams)
    # 主分析用 moderate（更有样本量），同时保留 strict sparse 子集
    analysis_set = sparse_set | moderate_set
    sparse_rows_idx = [
        i for i, (_, r) in enumerate(df_settled.iterrows())
        if r["home_team"] in analysis_set or r["away_team"] in analysis_set
    ]

    label_thresh = SPARSE_THRESH if sparse_teams else MODERATE_THRESH
    label_name   = "严格稀疏" if sparse_teams else "相对稀疏"

    print(f"\n{'='*68}")
    print(f"  含{label_name}球队的比赛 ({len(sparse_rows_idx)} 场, 阈值<{label_thresh}):")
    print(f"{'='*68}")

    detail_sparse = []
    for i in sparse_rows_idx:
        r      = df_settled.iloc[i]
        pb     = probs_base[i]    # [A,D,H]
        ps     = probs_shrunk[i]
        actual = r["result"]
        ht, at = r["home_team"], r["away_team"]

        pred_b = LABEL_ORDER[int(np.argmax(pb))]
        pred_s = LABEL_ORDER[int(np.argmax(ps))]
        ok_b   = "✓" if pred_b == actual else "✗"
        ok_s   = "✓" if pred_s == actual else "✗"

        # Brier 单场（sum-of-squares）
        yb_    = label_binarize([actual], classes=LABEL_ORDER)[0]
        brier_b = float(np.sum((pb - yb_)**2))
        brier_s = float(np.sum((ps - yb_)**2))
        db_     = brier_s - brier_b

        n_h = mc.get(norm_team(ht), 0); n_a = mc.get(norm_team(at), 0)
        which_sparse = []
        if ht in analysis_set: which_sparse.append(f"{ht}(n={n_h})")
        if at in analysis_set: which_sparse.append(f"{at}(n={n_a})")

        print(f"\n  {str(r['date'].date())}  {ht} vs {at}  实际={actual}")
        print(f"  稀疏方: {' / '.join(which_sparse)}")
        print(f"  {'模型':<18} {'P(A)':>6} {'P(D)':>6} {'P(H)':>6}  {'预测':>4}  {'结果':>4}  {'Brier':>7}")
        print(f"  {'─'*60}")
        print(f"  {'Base DC':<18} {pb[0]:>6.3f} {pb[1]:>6.3f} {pb[2]:>6.3f}  {pred_b:>4}  {ok_b:>4}  {brier_b:>7.4f}")
        print(f"  {f'Shrunk_B(k={K_SHRINK})':<18} {ps[0]:>6.3f} {ps[1]:>6.3f} {ps[2]:>6.3f}  {pred_s:>4}  {ok_s:>4}  {brier_s:>7.4f}")
        print(f"  {'Δ(收缩-原始)':<18} {ps[0]-pb[0]:>+6.3f} {ps[1]-pb[1]:>+6.3f} {ps[2]-pb[2]:>+6.3f}              {db_:>+7.4f}")

        detail_sparse.append({
            "date":  str(r["date"].date()),
            "home":  ht, "away": at,
            "actual": actual,
            "sparse_teams": "; ".join(which_sparse),
            "n_home": n_h, "n_away": n_a,
            "base_pa":  round(float(pb[0]),4), "base_pd": round(float(pb[1]),4), "base_ph": round(float(pb[2]),4),
            "base_pred": pred_b, "base_ok": ok_b,
            "shrB_pa":  round(float(ps[0]),4), "shrB_pd": round(float(ps[1]),4), "shrB_ph": round(float(ps[2]),4),
            "shrB_pred": pred_s, "shrB_ok": ok_s,
            "brier_base": round(brier_b,4), "brier_shrB": round(brier_s,4),
            "delta_brier": round(db_,4),
        })

    # ── 9. 稀疏子集汇总 ──────────────────────────────────────────────────────
    if sparse_rows_idx:
        y_sp  = y_true[sparse_rows_idx]
        m_sp_base   = eval_probs(probs_base[sparse_rows_idx],   y_sp)
        m_sp_shrunk = eval_probs(probs_shrunk[sparse_rows_idx], y_sp)
        da_sp = m_sp_shrunk["acc"]   - m_sp_base["acc"]
        db_sp = m_sp_shrunk["brier"] - m_sp_base["brier"]

        print(f"\n{'═'*68}")
        print(f"  稀疏子集汇总 ({len(sparse_rows_idx)} 场):")
        print(f"  {'模型':<14} {'ACC':>8} {'Brier':>8} {'LogLoss':>10}")
        print(f"  {'─'*44}")
        for label, m in [("Base DC", m_sp_base), (f"Shrunk_B", m_sp_shrunk)]:
            print(f"  {label:<14} {m['acc']:>8.4f} {m['brier']:>8.4f} {m['log_loss']:>10.4f}")
        print(f"  {'Δ':<14} {da_sp:>+8.4f} {db_sp:>+8.4f}")

    # ── 10. 逐场收缩效果分析 ─────────────────────────────────────────────────
    if detail_sparse:
        n_better = sum(1 for r in detail_sparse if r["delta_brier"] < -0.001)
        n_worse  = sum(1 for r in detail_sparse if r["delta_brier"] > +0.001)
        n_same   = len(detail_sparse) - n_better - n_worse

        print(f"\n  收缩 Brier 变化分布（含稀疏队的 {len(detail_sparse)} 场）:")
        print(f"    改善 (ΔBrier < -0.001): {n_better} 场")
        print(f"    基本不变 (|Δ| ≤ 0.001): {n_same} 场")
        print(f"    变差 (ΔBrier > +0.001): {n_worse} 场")

        # 稀疏/中度稀疏队参数变化展示
        print(f"\n  {label_name}球队参数收缩对比 (n/(n+k) 是保留自身比例):")
        print(f"  {'球队':<28} {'大洲':>10}  {'n':>4}  {'n/(n+k)':>8}"
              f"  {'log_a原始':>9} {'log_a收缩':>9} {'Δlog_a':>7}"
              f"  {'log_d原始':>9} {'log_d收缩':>9} {'Δlog_d':>7}")
        print(f"  {'─'*115}")
        for t in sorted(analysis_set):
            nt   = norm_team(t)
            cf   = confed_map.get(nt, confed_map.get(t, "UNKNOWN"))
            n    = mc.get(nt, 0)
            la_b = dc_base.log_a.get(nt, dc_base.avg_log_a)
            la_s = dc_shrunk.log_a.get(nt, dc_shrunk.avg_log_a)
            ld_b = dc_base.log_d.get(nt, dc_base.avg_log_d)
            ld_s = dc_shrunk.log_d.get(nt, dc_shrunk.avg_log_d)
            own_weight = n / (n + K_SHRINK) if (n + K_SHRINK) > 0 else 0
            in_model = "(model)" if nt in dc_base.log_a else "(avg)"
            print(f"  {t:<28} {cf:>10}  {n:>4}  {own_weight:>8.3f}"
                  f"  {la_b:>9.4f} {la_s:>9.4f} {la_s-la_b:>+7.4f}"
                  f"  {ld_b:>9.4f} {ld_s:>9.4f} {ld_s-ld_b:>+7.4f}  {in_model}")

    # ── 11. 保存输出 ──────────────────────────────────────────────────────────
    # 全部场次明细
    all_detail = []
    for i, (_, r) in enumerate(df_settled.iterrows()):
        pb = probs_base[i]; ps = probs_shrunk[i]
        actual = r["result"]
        ht, at = r["home_team"], r["away_team"]
        n_h = mc.get(norm_team(ht), 0); n_a = mc.get(norm_team(at), 0)
        yb_ = label_binarize([actual], classes=LABEL_ORDER)[0]
        all_detail.append({
            "date": str(r["date"].date()), "home": ht, "away": at,
            "actual": actual, "n_home": n_h, "n_away": n_a,
            "is_moderate": int(n_h < MODERATE_THRESH or n_a < MODERATE_THRESH),
            "is_sparse": int(n_h < SPARSE_THRESH or n_a < SPARSE_THRESH),
            "neutral": r.get("neutral", True),
            "base_pa": round(float(pb[0]),4), "base_pd": round(float(pb[1]),4),
            "base_ph": round(float(pb[2]),4),
            "base_pred": LABEL_ORDER[int(np.argmax(pb))],
            "base_ok": LABEL_ORDER[int(np.argmax(pb))] == actual,
            "shrB_pa": round(float(ps[0]),4), "shrB_pd": round(float(ps[1]),4),
            "shrB_ph": round(float(ps[2]),4),
            "shrB_pred": LABEL_ORDER[int(np.argmax(ps))],
            "shrB_ok": LABEL_ORDER[int(np.argmax(ps))] == actual,
            "brier_base": round(float(np.sum((pb - yb_)**2)), 4),
            "brier_shrB": round(float(np.sum((ps - yb_)**2)), 4),
        })

    df_all  = pd.DataFrame(all_detail)
    df_sp_d = pd.DataFrame(detail_sparse) if detail_sparse else pd.DataFrame()

    out_all  = OUT_G3 / "wc2026_all_compare.csv"
    out_sp   = OUT_G3 / "wc2026_sparse_compare.csv"
    df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
    if not df_sp_d.empty:
        df_sp_d.to_csv(out_sp, index=False, encoding="utf-8-sig")

    print(f"\n  输出文件:")
    print(f"    {out_all}")
    if not df_sp_d.empty:
        print(f"    {out_sp}")

    # ── 12. 结论 ─────────────────────────────────────────────────────────────
    print(f"\n{'#'*68}")
    print("  诊断结论")
    print(f"{'#'*68}")
    if not sparse_teams:
        print(f"  [发现] 2026 WC 已结算 {n_total} 场中，无严格稀疏队 (n<{SPARSE_THRESH})")
        print(f"         即使首次参加 WC 的队伍，在 1998 以来的国际比赛记录中均有足够数据")
        print(f"         --> 对这届 WC 来说，稀疏问题主要在于某些首次参赛队的"
              f"历史n={SPARSE_THRESH}~{MODERATE_THRESH}范围内")
    if sparse_rows_idx:
        if db_sp < -0.005:
            verdict = "[改善] 收缩降低了 Brier (概率质量提升)"
        elif db_sp > 0.005:
            verdict = "[变差] 收缩升高了 Brier (概率质量下降)"
        else:
            verdict = "[无显著差异] Brier 变化在 ±0.005 以内"
        print(f"\n  相对稀疏子集({len(sparse_rows_idx)}场, n<{label_thresh}): {verdict}")
        print(f"    ΔACC = {da_sp:+.4f}   ΔBrier = {db_sp:+.4f}")
    print(f"\n  整体({n_total}场): 收缩 ΔACC={da:+.4f}  ΔBrier={db:+.4f}")
    print(f"\n  样本量限制：{n_total} 场为 WC 前两轮，结论为定性参考")
    print("  正式论文应说明：此测试不含 WC 淘汰赛，样本不具统计显著性")
    print()


if __name__ == "__main__":
    main()
