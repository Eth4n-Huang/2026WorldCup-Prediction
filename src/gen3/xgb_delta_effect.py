"""
XGB-adj 效果量化诊断
只读三届回测 CSV，不重新训练模型，不改任何输出。
输出: outputs/gen3/xgb_delta_effect.txt
"""

import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "gen3"
OUT.mkdir(parents=True, exist_ok=True)

WC_YEARS = [2014, 2018, 2022]

# ── 读取三届回测 ──────────────────────────────────────────────
frames = {}
for y in WC_YEARS:
    p = ROOT / "outputs" / f"backtest_{y}.csv"
    df = pd.read_csv(p)
    frames[y] = df

all_rows = pd.concat(frames.values(), ignore_index=True)

lines = []
def pr(s=""):
    lines.append(s)
    print(s)

pr("=" * 65)
pr("XGB-adj 效果量化诊断（三届世界杯回测，192场）")
pr("=" * 65)

# ══════════════════════════════════════════════════════════════
# 1. 平局校准：XGB 高估还是低估平局？
# ══════════════════════════════════════════════════════════════
pr()
pr("【1】XGB 平局概率校准")
pr("-" * 40)

for y, df in frames.items():
    mean_pd     = df["xgb_prob_D"].mean()
    actual_dr   = (df["result"] == "D").mean()
    delta_ppt   = (mean_pd - actual_dr) * 100
    direction   = "高估" if delta_ppt > 0 else "低估"
    pr(f"  {y}: 均值P(D)={mean_pd:.4f}  实际平局率={actual_dr:.4f}"
       f"  差={delta_ppt:+.2f}ppt  [{direction}]")

mean_pd_all   = all_rows["xgb_prob_D"].mean()
actual_dr_all = (all_rows["result"] == "D").mean()
delta_all     = (mean_pd_all - actual_dr_all) * 100
direction_all = "高估" if delta_all > 0 else "低估"
pr(f"  三届合并: 均值P(D)={mean_pd_all:.4f}  实际平局率={actual_dr_all:.4f}"
   f"  差={delta_all:+.2f}ppt  [{direction_all}]")
pr(f"  (对照: DC 均值P(D) ≈ 0.2753，实际平局率 0.2135，差 +6.18ppt [高估])")

# ══════════════════════════════════════════════════════════════
# 2. argmax vs adj ACC 对比，逐届拆分
# ══════════════════════════════════════════════════════════════
pr()
pr("【2】XGB 纯argmax vs XGB+adj  准确率对比")
pr("-" * 40)

summary_rows = []
for y, df in frames.items():
    n          = len(df)
    acc_argmax = df["xgb_correct"].astype(float).mean()
    acc_adj    = df["xgb_adj_correct"].astype(float).mean()
    delta_acc  = (acc_adj - acc_argmax) * 100

    # 改判场次分析
    changed_mask = df["xgb_argmax"] != df["xgb_draw_adj"]
    changed      = df[changed_mask].copy()
    n_changed    = len(changed)

    # 改成D了几场
    to_d_mask  = (df["xgb_argmax"] != "D") & (df["xgb_draw_adj"] == "D")
    to_d       = df[to_d_mask].copy()
    n_to_d     = len(to_d)

    # 改对（真实是D）vs 改错
    correct_chg  = (to_d["result"] == "D").sum()
    wrong_chg    = (to_d["result"] != "D").sum()
    fp_rate      = wrong_chg / n_to_d * 100 if n_to_d > 0 else float("nan")

    pr(f"  {y}  (n={n})")
    pr(f"    argmax ACC={acc_argmax:.4f}  adj ACC={acc_adj:.4f}"
       f"  ΔACC={delta_acc:+.2f}ppt")
    pr(f"    改判场数={n_changed}  其中→D={n_to_d}"
       f"  改对={correct_chg}  改错={wrong_chg}"
       f"  假阳性率={fp_rate:.1f}%")

    summary_rows.append(dict(year=y, n=n,
                             acc_argmax=acc_argmax, acc_adj=acc_adj,
                             delta_acc=delta_acc,
                             n_changed=n_changed, n_to_d=n_to_d,
                             correct_chg=correct_chg, wrong_chg=wrong_chg,
                             fp_rate=fp_rate))

# 三届合并
n_all         = len(all_rows)
acc_arg_all   = all_rows["xgb_correct"].astype(float).mean()
acc_adj_all   = all_rows["xgb_adj_correct"].astype(float).mean()
delta_all_acc = (acc_adj_all - acc_arg_all) * 100

to_d_all      = all_rows[(all_rows["xgb_argmax"] != "D") &
                          (all_rows["xgb_draw_adj"] == "D")]
n_to_d_all    = len(to_d_all)
correct_all   = (to_d_all["result"] == "D").sum()
wrong_all     = (to_d_all["result"] != "D").sum()
fp_all        = wrong_all / n_to_d_all * 100 if n_to_d_all > 0 else float("nan")

pr()
pr(f"  三届合并  (n={n_all})")
pr(f"    argmax ACC={acc_arg_all:.4f}  adj ACC={acc_adj_all:.4f}"
   f"  ΔACC={delta_all_acc:+.2f}ppt")
pr(f"    改判→D={n_to_d_all}  改对={correct_all}  改错={wrong_all}"
   f"  假阳性率={fp_all:.1f}%")

# ══════════════════════════════════════════════════════════════
# 2b. 平局召回率对比
# ══════════════════════════════════════════════════════════════
pr()
pr("【2b】平局召回率对比")
pr("-" * 40)

for y, df in frames.items():
    actual_d   = df["result"] == "D"
    recall_arg = ((df["xgb_argmax"] == "D") & actual_d).sum()
    recall_adj = ((df["xgb_draw_adj"] == "D") & actual_d).sum()
    total_d    = actual_d.sum()
    pr(f"  {y}: 实际平局={total_d}场  "
       f"argmax召回={recall_arg}/{total_d}  adj召回={recall_adj}/{total_d}")

actual_d_all   = all_rows["result"] == "D"
recall_arg_all = ((all_rows["xgb_argmax"] == "D") & actual_d_all).sum()
recall_adj_all = ((all_rows["xgb_draw_adj"] == "D") & actual_d_all).sum()
total_d_all    = actual_d_all.sum()
pr(f"  三届合并: 实际平局={total_d_all}场  "
   f"argmax召回={recall_arg_all}/{total_d_all}  "
   f"adj召回={recall_adj_all}/{total_d_all}")

# ══════════════════════════════════════════════════════════════
# 2c. Brier Score 对比
# ══════════════════════════════════════════════════════════════
pr()
pr("【2c】Brier Score（概率质量，adj不改概率，结果应相同）")
pr("-" * 40)

LABEL_ORDER = ["A", "D", "H"]

def brier(df):
    scores = []
    for _, row in df.iterrows():
        y_true = [1.0 if row["result"] == lb else 0.0 for lb in LABEL_ORDER]
        y_pred = [row[f"xgb_prob_{lb}"] for lb in LABEL_ORDER]
        scores.append(sum((p - t) ** 2 for p, t in zip(y_pred, y_true)))
    return np.mean(scores)

for y, df in frames.items():
    b = brier(df)
    pr(f"  {y}: Brier={b:.4f}")

pr(f"  三届合并: Brier={brier(all_rows):.4f}")
pr("  (Brier 不随 adj 变化，因为 adj 只改点预测不改概率输出)")

# ══════════════════════════════════════════════════════════════
# 3. 结论
# ══════════════════════════════════════════════════════════════
pr()
pr("【3】结论")
pr("-" * 40)

# 判断方向
if delta_all_acc < -1.0:
    verdict = "有害"
    reason_q = "是否同因"
elif abs(delta_all_acc) <= 1.0:
    verdict = "中性（±1ppt以内）"
    reason_q = "同因性存疑"
else:
    verdict = "有益"
    reason_q = "N/A"

pr(f"  XGB-adj 总体效果: {verdict}")
pr(f"  三届平均 ΔACC = {delta_all_acc:+.2f}ppt")
pr(f"  假阳性率 = {fp_all:.1f}%  (改对{correct_all}场 / 共改{n_to_d_all}场)")
pr()
pr(f"  XGB 平局概率偏差 = {delta_all:+.2f}ppt  [{'高估' if delta_all>0 else '低估'}]")
pr()
if delta_all > 2.0 and delta_all_acc < -1.0:
    pr("  同因诊断: 是。XGB 也高估 P(D)，adj 在此基础上再强推平局，")
    pr("            导致假阳性高、ACC 受损——与 DC 的病因相同。")
elif delta_all > 2.0 and abs(delta_all_acc) <= 1.0:
    pr("  同因诊断: 部分相同。XGB 也有轻度高估 P(D)，但 adj 效果接近中性，")
    pr("            说明 XGB 的 δ 校准比 DC 的更保守，损失可控。")
elif delta_all < 0:
    pr("  同因诊断: 否。XGB 低估 P(D)，adj 属于'补残差'，")
    pr("            与 DC 的情形不同——adj 在 XGB 上更有正当性。")
else:
    pr(f"  同因诊断: 偏差={delta_all:+.2f}ppt，轻度高估但幅度远低于 DC (+6.2ppt)。")

# ── 保存 ──────────────────────────────────────────────────────
out_path = OUT / "xgb_delta_effect.txt"
out_path.write_text("\n".join(lines), encoding="utf-8")
pr()
pr(f"已保存: {out_path}")
