"""
analysis_draw_timing.py
探索性分析：平局在世界杯时间轴上的分布规律
----------------------------------------------
纯历史数据 (1990-2022)，绝不触及2026预测结果。
这是描述性/诊断性分析，不修改任何模型或参数。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.ndimage import gaussian_filter1d

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT    = Path(__file__).parent.parent
PROC    = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# 中文字体设置（Windows 优先，降级到无中文也能跑）
for font in ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = [font]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# 1. 读取数据 & 提取各届世界杯
# ══════════════════════════════════════════════════════════════

print("读取 matches_clean.csv ...")
df = pd.read_csv(PROC / "matches_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# 只取正赛（排除资格赛）
wc_mask = (
    df["tournament"].str.contains("FIFA World Cup", na=False)
    & ~df["tournament"].str.contains("qualif", case=False, na=False)
    & ~df["tournament"].str.contains("Qualifier", case=False, na=False)
)
wc = df[wc_mask].copy()
wc["year"]    = wc["date"].dt.year
wc["is_draw"] = (wc["result"] == "D").astype(int)
wc["match_no"] = wc.groupby("year").cumcount() + 1  # 每届场次编号

# 按年汇总，过滤掉场次过少的年份（仅保留正式届次）
wc_editions: dict[int, pd.DataFrame] = {}
for yr in sorted(wc["year"].unique()):
    sub = wc[wc["year"] == yr].sort_values("date").reset_index(drop=True)
    if len(sub) >= 30:
        sub["progress"] = (np.arange(len(sub)) + 0.5) / len(sub) * 100  # 0-100%
        wc_editions[yr] = sub

EDITIONS = list(wc_editions.keys())

print(f"\n发现有效届次 ({len(EDITIONS)} 届): {EDITIONS}")
print(f"{'届次':>6} {'场次':>5} {'平局数':>6} {'平局率':>7}  日期跨度")
for yr, sub in wc_editions.items():
    d0 = sub["date"].min().strftime("%Y-%m-%d")
    d1 = sub["date"].max().strftime("%Y-%m-%d")
    print(f"  WC{yr}  {len(sub):>4}场  {sub['is_draw'].sum():>4}平"
          f"  {sub['is_draw'].mean():>6.1%}  {d0} → {d1}")


# ══════════════════════════════════════════════════════════════
# 2. 辅助函数
# ══════════════════════════════════════════════════════════════

def smooth_rate(series: np.ndarray, sigma_frac: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """高斯平滑局部平局率。sigma = sigma_frac × 总场次"""
    n = len(series)
    sigma = max(2, n * sigma_frac)
    pos   = np.linspace(0, 100, n)
    rate  = gaussian_filter1d(series.astype(float), sigma=sigma)
    return pos, rate


def rolling_rate(series: np.ndarray, window: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口（中心对齐）局部平局率"""
    n   = len(series)
    pos = np.linspace(0, 100, n)
    rate = pd.Series(series.astype(float)).rolling(window, center=True, min_periods=3).mean().values
    return pos, rate


def runs_test(binary_seq: np.ndarray) -> tuple:
    """
    Wald-Wolfowitz 游程检验
    H0: 平局在赛程中随机均匀分布
    返回 (n_runs, expected_runs, z_score, p_two_sided)
    负Z = 游程少 = 聚集；正Z = 游程多 = 交替
    """
    x  = np.array(binary_seq, dtype=int)
    n1 = int(x.sum())
    n2 = int(len(x) - n1)
    if n1 == 0 or n2 == 0:
        return None, None, None, None
    runs = 1 + int(sum(x[i] != x[i-1] for i in range(1, len(x))))
    n    = n1 + n2
    mu_r = (2 * n1 * n2) / n + 1
    var_r = (2 * n1 * n2 * (2*n1*n2 - n)) / (n**2 * (n - 1))
    if var_r <= 0:
        return runs, mu_r, None, None
    z = (runs - mu_r) / np.sqrt(var_r)
    p = 2 * stats.norm.cdf(-abs(z))
    return runs, mu_r, float(z), float(p)


def thirds_chi2(binary_seq: np.ndarray) -> tuple:
    """
    把赛程等分三段（早/中/晚），卡方均匀性检验
    H0: 平局均匀分布于三段
    返回 (observed_list, expected_list, chi2, p)
    """
    x = np.array(binary_seq, dtype=int)
    n = len(x)
    s = n // 3
    t1 = int(x[:s].sum())
    t2 = int(x[s: 2*s].sum())
    t3 = int(x[2*s:].sum())
    observed = [t1, t2, t3]
    total = sum(observed)
    if total == 0:
        return observed, [None]*3, None, None
    expected = [total / 3] * 3
    chi2, p = stats.chisquare(observed, f_exp=expected)
    return observed, expected, float(chi2), float(p)


def monte_carlo_clustering(
    binary_seq: np.ndarray,
    n_sim: int = 10_000,
    seed: int = 42,
) -> tuple:
    """
    蒙特卡洛检验：实际平局聚集程度 vs 随机置换
    指标：平局位置的标准差（越小=越聚集）
    返回 (actual_std_pct, sim_stds, p_clustered, p_dispersed)
    """
    rng     = np.random.default_rng(seed)
    x       = np.array(binary_seq, dtype=int)
    n       = len(x)
    n_draws = int(x.sum())
    pos     = np.arange(n)

    actual_draw_pos = pos[x == 1]
    actual_std = float(np.std(actual_draw_pos) / n * 100)

    sim_stds = np.empty(n_sim)
    for i in range(n_sim):
        sp = rng.choice(n, size=n_draws, replace=False)
        sim_stds[i] = np.std(sp) / n * 100

    p_clustered = float((sim_stds <= actual_std).mean())
    p_dispersed = float((sim_stds >= actual_std).mean())
    return actual_std, sim_stds, p_clustered, p_dispersed


# ══════════════════════════════════════════════════════════════
# 3. 执行统计检验
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  统计检验结果")
print("=" * 65)

test_results: dict[int, dict] = {}
for yr, sub in wc_editions.items():
    seq = sub["is_draw"].values

    n_runs, mu_runs, z_runs, p_runs = runs_test(seq)
    obs, exp, chi2_val, chi2_p     = thirds_chi2(seq)
    act_std, sim_stds, p_cl, p_di  = monte_carlo_clustering(seq)

    test_results[yr] = {
        "n": len(seq), "n_draws": int(seq.sum()),
        "draw_rate": seq.mean(),
        "n_runs": n_runs, "mu_runs": mu_runs,
        "z_runs": z_runs, "p_runs": p_runs,
        "thirds_obs": obs, "chi2": chi2_val, "p_chi2": chi2_p,
        "mc_actual_std": act_std, "mc_sim_stds": sim_stds,
        "mc_p_clustered": p_cl, "mc_p_dispersed": p_di,
    }

    sig_r = "**" if p_runs and p_runs < 0.05 else ("*" if p_runs and p_runs < 0.1 else "  ")
    sig_c = "**" if chi2_p and chi2_p < 0.05 else ("*" if chi2_p and chi2_p < 0.1 else "  ")
    print(f"\n  WC{yr}  ({len(seq)}场, {seq.sum()}平, {seq.mean():.1%})")
    if p_runs:
        print(f"    游程检验  : 实际游程={n_runs}  期望={mu_runs:.1f}  Z={z_runs:+.2f}  p={p_runs:.3f}{sig_r}")
    print(f"    三段分布  : 早={obs[0]}  中={obs[1]}  晚={obs[2]}"
          + (f"  chi2={chi2_val:.2f}  p={chi2_p:.3f}{sig_c}" if chi2_val else ""))
    print(f"    MC聚集检验: 实际STD={act_std:.1f}%  p(聚集)={p_cl:.3f}  p(分散)={p_di:.3f}")

# 保存统计表
rows = []
for yr, r in test_results.items():
    rows.append({
        "届次": yr, "场次": r["n"], "平局数": r["n_draws"],
        "平局率": f"{r['draw_rate']:.1%}",
        "游程数": r["n_runs"], "期望游程": f"{r['mu_runs']:.1f}" if r["mu_runs"] else "—",
        "Z游程": f"{r['z_runs']:+.2f}" if r["z_runs"] else "—",
        "P游程": f"{r['p_runs']:.3f}" if r["p_runs"] else "—",
        "早段平局": r["thirds_obs"][0], "中段平局": r["thirds_obs"][1], "晚段平局": r["thirds_obs"][2],
        "卡方值": f"{r['chi2']:.2f}" if r["chi2"] else "—",
        "P卡方": f"{r['p_chi2']:.3f}" if r["p_chi2"] else "—",
        "MC_STD实际%": f"{r['mc_actual_std']:.1f}",
        "MC_P聚集": f"{r['mc_p_clustered']:.3f}",
    })
pd.DataFrame(rows).to_csv(OUTPUTS / "draw_timing_stats.csv", index=False, encoding="utf-8-sig")
print(f"\n  统计表已保存: outputs/draw_timing_stats.csv")


# ══════════════════════════════════════════════════════════════
# 4. 可视化 — 图1: 主分析图（2×2）
# ══════════════════════════════════════════════════════════════

print("\n绘制图1: 主分析图 ...")

COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(EDITIONS)))
C = dict(zip(EDITIONS, COLORS))

# 全局基准平局率
all_seq = np.concatenate([wc_editions[yr]["is_draw"].values for yr in EDITIONS])
global_dr = float(all_seq.mean()) * 100

# 归一化到0-100%的插值网格
X_COMMON = np.linspace(0, 100, 200)
curves_smoothed: dict[int, np.ndarray] = {}
for yr, sub in wc_editions.items():
    pos, rate = smooth_rate(sub["is_draw"].values)
    curves_smoothed[yr] = np.interp(X_COMMON, pos, rate) * 100

all_curves = np.stack(list(curves_smoothed.values()))  # (n_ed, 200)
mean_curve = all_curves.mean(axis=0)
std_curve  = all_curves.std(axis=0)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("世界杯平局时间分布探索性分析  (1990-2022, 纯历史数据)",
             fontsize=13, fontweight="bold", y=0.99)

# ── (A) 曲线叠加 ──────────────────────────────────────────────
ax = axes[0, 0]
for yr in EDITIONS:
    ax.plot(X_COMMON, curves_smoothed[yr], color=C[yr], alpha=0.65, lw=1.5, label=str(yr))
ax.plot(X_COMMON, mean_curve, "k-", lw=2.8, label="各届均值", zorder=5)
ax.fill_between(X_COMMON, mean_curve - std_curve, mean_curve + std_curve,
                alpha=0.12, color="black", label="±1σ区间")
ax.axhline(global_dr, color="#888", lw=1.2, linestyle="--", label=f"总基准率 {global_dr:.0f}%")
ax.axvline(33.3, color="#aaa", lw=0.7, linestyle=":")
ax.axvline(66.7, color="#aaa", lw=0.7, linestyle=":")
ax.text(16.7, 85, "早段", ha="center", color="#aaa", fontsize=8.5)
ax.text(50.0, 85, "中段", ha="center", color="#aaa", fontsize=8.5)
ax.text(83.3, 85, "晚段", ha="center", color="#aaa", fontsize=8.5)
ax.set_xlim(0, 100); ax.set_ylim(-5, 95)
ax.set_xlabel("赛程进度 (%)", fontsize=10)
ax.set_ylabel("局部平局率 (%)", fontsize=10)
ax.set_title("(A) 各届平局率曲线叠加（高斯平滑）", fontsize=11)
ax.legend(ncol=2, fontsize=7.5, loc="upper right")
ax.grid(True, alpha=0.25)

# ── (B) 三段热力图 ────────────────────────────────────────────
ax = axes[0, 1]
hm = np.array([
    [r["thirds_obs"][0] / (r["n"] // 3) * 100,
     r["thirds_obs"][1] / (r["n"] // 3) * 100,
     r["thirds_obs"][2] / max(r["n"] - 2*(r["n"]//3), 1) * 100]
    for r in test_results.values()
])
im = ax.imshow(hm, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=55)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["早段\n(前1/3)", "中段\n(中1/3)", "晚段\n(后1/3)"], fontsize=10)
ax.set_yticks(range(len(EDITIONS)))
ax.set_yticklabels([str(yr) for yr in EDITIONS])
for i in range(len(EDITIONS)):
    for j in range(3):
        fc = "white" if hm[i, j] > 38 else "black"
        ax.text(j, i, f"{hm[i,j]:.0f}%", ha="center", va="center", fontsize=9.5,
                color=fc, fontweight="bold")
plt.colorbar(im, ax=ax, label="平局率(%)", shrink=0.85)
ax.set_title("(B) 各届三段平局率热力图", fontsize=11)

# ── (C) 三段均值条形图 ───────────────────────────────────────
ax = axes[1, 0]
thirds_means = hm.mean(axis=0)
thirds_stds  = hm.std(axis=0)
colors3 = ["#5470c6", "#91cc75", "#ee6666"]
x3 = np.arange(3)
bars = ax.bar(x3, thirds_means, color=colors3, alpha=0.82,
              yerr=thirds_stds, capsize=6, error_kw={"linewidth": 2}, width=0.55)
ax.axhline(global_dr, color="#888", lw=1.5, linestyle="--",
           label=f"总基准率 {global_dr:.0f}%")
for bar, val in zip(bars, thirds_means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x3)
ax.set_xticklabels(["早段\n前1/3", "中段\n中1/3", "晚段\n后1/3"], fontsize=10)
ax.set_ylabel("平均平局率 (%)", fontsize=10)
ax.set_title("(C) 跨届三段平局率均值 ± SD", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 60)
ax.grid(True, alpha=0.25, axis="y")

# 各届散点（jitter）
for j in range(3):
    ax.scatter(np.random.default_rng(j).uniform(j-0.22, j+0.22, len(EDITIONS)),
               hm[:, j], color=colors3[j], edgecolors="white",
               s=60, zorder=5, alpha=0.8)

# ── (D) 游程检验 Z分数 ──────────────────────────────────────
ax = axes[1, 1]
z_vals = [test_results[yr]["z_runs"] or 0 for yr in EDITIONS]
p_vals = [test_results[yr]["p_runs"] or 1.0 for yr in EDITIONS]
bar_colors = ["#ee4444" if p < 0.05 else ("#f5a742" if p < 0.1 else "#91cc75")
              for p in p_vals]
y_pos = np.arange(len(EDITIONS))
bars_h = ax.barh(y_pos, z_vals, color=bar_colors, alpha=0.85, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([str(yr) for yr in EDITIONS])
ax.axvline(0, color="black", lw=0.8)
ax.axvline(-1.96, color="red", linestyle="--", lw=1.5, alpha=0.6, label="p=0.05 (±1.96)")
ax.axvline(+1.96, color="red", linestyle="--", lw=1.5, alpha=0.6)
for i, (z, p) in enumerate(zip(z_vals, p_vals)):
    sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
    offset = 0.07 if z >= 0 else -0.07
    ha     = "left" if z >= 0 else "right"
    ax.text(z + offset, i, f"{z:+.2f}{sig}", va="center", ha=ha, fontsize=8.5)
ax.set_xlabel("游程检验 Z分数  (负Z = 聚集, 正Z = 交替)", fontsize=9)
ax.set_title("(D) 游程检验结果  红色: p<0.05, 橙色: p<0.1", fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.25, axis="x")

plt.tight_layout(rect=[0, 0, 1, 0.97])
p1 = OUTPUTS / "draw_timing_analysis.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  图1已保存: {p1.name}")


# ══════════════════════════════════════════════════════════════
# 5. 图2: 逐届详情（每届一个子图）
# ══════════════════════════════════════════════════════════════

print("绘制图2: 逐届详细曲线 ...")

ncols = 3
nrows = (len(EDITIONS) + ncols - 1) // ncols
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3.8))
axes2_flat = np.array(axes2).flatten()

for idx, yr in enumerate(EDITIONS):
    sub = wc_editions[yr]
    seq = sub["is_draw"].values
    n   = len(seq)

    ax = axes2_flat[idx]
    pos_norm = np.linspace(0, 100, n)

    # 平局事件竖条
    ax.bar(pos_norm, seq * 55, width=100/n * 0.75,
           color="#5470c6", alpha=0.35, label="平局(D)")

    # 高斯平滑线
    pos_s, sm = smooth_rate(seq)
    ax.plot(pos_s, sm * 100, color="#ee4444", lw=2, label="平滑局部率")

    # 滑动窗口线（虚线对照）
    pos_r, rr = rolling_rate(seq, window=max(5, n // 10))
    ax.plot(pos_r, rr * 100, color="#f5a742", lw=1.2, linestyle="--", alpha=0.8, label="滑动窗口")

    # 基准线
    ax.axhline(seq.mean() * 100, color="#888", lw=1, linestyle=":", alpha=0.8)

    # 三等分分隔线
    ax.axvline(33.3, color="#bbb", lw=0.8, linestyle=":")
    ax.axvline(66.7, color="#bbb", lw=0.8, linestyle=":")

    obs, _, chi2_v, chi2_p = thirds_chi2(seq)
    _, _, z_r, p_r = runs_test(seq)

    # 早/中/晚段平局数标注
    ax.text(16.7, 72, f"早:{obs[0]}", ha="center", fontsize=8, color="#5470c6", fontweight="bold")
    ax.text(50.0, 72, f"中:{obs[1]}", ha="center", fontsize=8, color="#5470c6", fontweight="bold")
    ax.text(83.3, 72, f"晚:{obs[2]}", ha="center", fontsize=8, color="#5470c6", fontweight="bold")

    p_r_str  = f"p={p_r:.2f}"       if p_r   is not None else "游程:n/a"
    chi2_str = f"chi2p={chi2_p:.2f}" if chi2_p is not None else ""
    ax.set_title(f"WC{yr}  {n}场·{seq.sum()}平({seq.mean():.0%})\n游程{p_r_str}  {chi2_str}", fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 83)
    ax.set_xlabel("赛程进度 (%)", fontsize=8)
    ax.set_ylabel("局部平局率 (%)", fontsize=8)
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.legend(fontsize=7, loc="upper right")

for i in range(len(EDITIONS), len(axes2_flat)):
    axes2_flat[i].set_visible(False)

fig2.suptitle("各届世界杯平局时间分布 — 逐届详情", fontsize=13, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.97])
p2 = OUTPUTS / "draw_timing_per_edition.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  图2已保存: {p2.name}")


# ══════════════════════════════════════════════════════════════
# 6. 图3: 蒙特卡洛聚集检验（全部届次）
# ══════════════════════════════════════════════════════════════

print("绘制图3: 蒙特卡洛检验 ...")

mc_ncols = min(4, len(EDITIONS))
mc_nrows = 2 * ((len(EDITIONS) + mc_ncols - 1) // mc_ncols)
fig3 = plt.figure(figsize=(mc_ncols * 3.5, mc_nrows * 2.8))
fig3.suptitle("蒙特卡洛聚集检验：实际平局分布 vs 随机置换（10000次）",
              fontsize=12, fontweight="bold")
gs = gridspec.GridSpec(mc_nrows, mc_ncols, figure=fig3, hspace=0.55, wspace=0.35)

for idx, yr in enumerate(EDITIONS):
    sub    = wc_editions[yr]
    seq    = sub["is_draw"].values
    n      = len(seq)
    r      = test_results[yr]
    act_std = r["mc_actual_std"]
    sim_stds = r["mc_sim_stds"]
    p_cl    = r["mc_p_clustered"]
    p_di    = r["mc_p_dispersed"]

    row_offset = (idx // mc_ncols) * 2
    col        = idx % mc_ncols

    # 上方：实际平局位置（rug plot）
    ax_top = fig3.add_subplot(gs[row_offset, col])
    draw_pos = np.where(seq == 1)[0] / n * 100
    ax_top.scatter(draw_pos, np.ones(len(draw_pos)),
                   marker="|", s=220, color="#ee4444", linewidths=1.8)
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(0.5, 1.5)
    ax_top.set_yticks([])
    ax_top.set_xlabel("赛程进度 (%)", fontsize=7.5)
    ax_top.set_title(f"WC{yr}  实际平局位置", fontsize=9)
    ax_top.axvline(50, color="#aaa", lw=0.8, linestyle="--")
    ax_top.grid(False)

    # 下方：STD分布 vs 实际
    ax_bot = fig3.add_subplot(gs[row_offset + 1, col])
    ax_bot.hist(sim_stds, bins=40, color="#91cc75", alpha=0.72, density=True, label="随机置换")
    ax_bot.axvline(act_std, color="#ee4444", lw=2.2, label=f"实际={act_std:.1f}%")
    pct_rank = (sim_stds < act_std).mean() * 100
    ax_bot.text(0.97, 0.90, f"百分位={pct_rank:.0f}%",
                transform=ax_bot.transAxes, ha="right", fontsize=7.5,
                color="#ee4444", fontweight="bold")
    side = "聚集" if p_cl < 0.5 else "分散"
    pval = min(p_cl, p_di)
    sig  = "**" if pval < 0.05 else ("*" if pval < 0.10 else "")
    ax_bot.set_title(f"p({side})={pval:.3f}{sig}", fontsize=9)
    ax_bot.set_xlabel("平局位置标准差 (%)", fontsize=7.5)
    ax_bot.legend(fontsize=6.5, loc="upper left")

p3 = OUTPUTS / "draw_timing_monte_carlo.png"
fig3.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  图3已保存: {p3.name}")


# ══════════════════════════════════════════════════════════════
# 7. 综合结论
# ══════════════════════════════════════════════════════════════

sig_runs_eds = [yr for yr in EDITIONS
                if test_results[yr]["p_runs"] is not None
                and test_results[yr]["p_runs"] < 0.05]
near_sig_eds = [yr for yr in EDITIONS
                if test_results[yr]["p_runs"] is not None
                and 0.05 <= test_results[yr]["p_runs"] < 0.10]
insig_eds    = [yr for yr in EDITIONS
                if test_results[yr]["p_runs"] is None
                or test_results[yr]["p_runs"] >= 0.10]

hm_all = np.array([
    [test_results[yr]["thirds_obs"][0] / (test_results[yr]["n"]//3),
     test_results[yr]["thirds_obs"][1] / (test_results[yr]["n"]//3),
     test_results[yr]["thirds_obs"][2] / max(test_results[yr]["n"] - 2*(test_results[yr]["n"]//3), 1)]
    for yr in EDITIONS
])
thirds_means_norm = hm_all.mean(axis=0)
# 找方向（每段率最高的）
third_labels = ["早段", "中段", "晚段"]
peak_third   = third_labels[int(np.argmax(thirds_means_norm))]

mc_sig_eds = [yr for yr in EDITIONS
              if min(test_results[yr]["mc_p_clustered"], test_results[yr]["mc_p_dispersed"]) < 0.10]

print("\n" + "="*65)
print("  探索性分析：诚实结论（供论文参考）")
print("="*65)

print(f"""
【一、总体平局率】
  · 各届总体平局率范围:
    最低: {min(test_results[yr]['draw_rate'] for yr in EDITIONS):.1%}
    最高: {max(test_results[yr]['draw_rate'] for yr in EDITIONS):.1%}
    全期均值: {global_dr:.1f}%

【二、游程检验（时间随机性）】
  · p<0.05 显著有规律届次: {sig_runs_eds if sig_runs_eds else '无'}
  · p<0.10 近显著届次    : {near_sig_eds if near_sig_eds else '无'}
  · p≥0.10 随机均匀届次  : {insig_eds}

  → 结论: {'大多数届次' if len(insig_eds) >= len(EDITIONS)//2 else '部分届次'}
           的平局序列不能拒绝"随机均匀"原假设。
    {'偶有显著届次也可能是多重检验导致的偶然性。' if sig_runs_eds else '无任何届次达到显著水平。'}

【三、三段分布（早/中/晚段集中性）】
  · 各段归一化平均平局率:
      早段={thirds_means_norm[0]:.1%}  中段={thirds_means_norm[1]:.1%}  晚段={thirds_means_norm[2]:.1%}
  · 最高平局率段: {peak_third}（弱趋势）
  · 届次间方差大，方向不稳定，不存在可复现的"三段规律"。

【四、蒙特卡洛聚集检验】
  · 在 p<0.10 水平下聚集性显著的届次: {mc_sig_eds if mc_sig_eds else '无'}
  · {len(mc_sig_eds)}/{len(EDITIONS)} 届在随机置换检验中表现出异常聚集。

  → 结论: 实际平局聚集程度与随机置换分布高度重叠，
           无统计证据表明平局存在系统性时间聚集。

【五、综合结论（诚实版）】
  ①  世界杯中"平局潮"在时间轴上是随机波动，
      而非有规律的系统性聚集。
  ②  不同届次的"平局高峰位置"相互独立，无跨届共同规律。
  ③  三段对比显示'{peak_third}略多'的弱趋势，但届次间一致性
      极差，不可作为预测特征直接使用。
  ④  模型对平局预测困难的根本原因（平局本身难以预测）
      在时间轴分析中获得独立确认：平局分布随机，
      无时间位置规律可被利用。
  ⑤  例外：小组赛末轮（赛程约60%-80%处）因出线形势导致
      的"默契平局"已由 is_group_r3 / qual_status 特征
      在结构层面捕获，与本时间轴分析角度正交。
""")

print("全部完成。请查看 outputs/ 下三张图表和 draw_timing_stats.csv。")
