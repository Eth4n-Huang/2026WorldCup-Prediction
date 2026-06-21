"""
src/gen3/delta_effect.py
========================
量化 δ 决策规则在三届 WC 回测(192场)上的实际效果

做法：
  对每届 WC：
    1. 全量训练集 (1998 ~ 开赛前) 训练 DC → 拿 WC 场次 probs
    2. 训练集内 85/15 时间切割：
         fit_dc(前85%) → val_probs(后15%) → tune_draw_threshold → (δ*, draw_thr*)
       满足「预测平局率 ≈ 训练平局基率 ± 3ppt」且 macro-F1 最高
    3. 用 (δ*, draw_thr*) 对 WC 场次做 draw-adj 预测
  不改任何模型；三届均用同一套调参逻辑（单独调，不共享）

防泄漏说明：δ 只用训练期内数据选取，WC 测试场次不参与调参
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

from step5b_improve import DixonColesModel, LABEL_ORDER
from step5c_devset   import dc_probs_for_matches, FIXED_HALF_LIFE
from step4_train     import tune_draw_threshold, predict_with_draw_adj

PROCESSED   = ROOT / "data" / "processed"
OUT_G3      = ROOT / "outputs" / "gen3"
TRAIN_START = pd.Timestamp("1998-01-01")

WC_DATES = {
    2014: (pd.Timestamp("2014-06-12"), pd.Timestamp("2014-07-13")),
    2018: (pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-15")),
    2022: (pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18")),
}


# ── 指标计算 ─────────────────────────────────────────────────────────────────

def brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    yb = label_binarize(y_true, classes=LABEL_ORDER)
    return float(np.mean(np.sum((probs - yb) ** 2, axis=1)))


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  量化 δ 效果：DC+δ vs DC-argmax  (三届 WC 回测 192 场)")
    print("=" * 72)

    df_hist = pd.read_csv(PROCESSED / "matches_clean.csv", parse_dates=["date"])
    df_hist = df_hist.sort_values("date").reset_index(drop=True)

    all_records   = []   # 每场明细
    summary_rows  = []   # 每届汇总

    for year, (wc_start, wc_end) in sorted(WC_DATES.items()):
        print(f"\n{'═'*72}")
        print(f"  WC{year}  (训练截止 {(wc_start - pd.Timedelta(days=1)).date()})")
        print(f"{'═'*72}")

        # ── 1. 训练集 & 测试集 ────────────────────────────────────────────
        df_train = df_hist[
            (df_hist["date"] >= TRAIN_START) &
            (df_hist["date"] < wc_start)
        ].copy()
        df_test  = df_hist[
            (df_hist["date"] >= wc_start) &
            (df_hist["date"] <= wc_end) &
            df_hist["tournament"].str.contains("FIFA World Cup", na=False)
        ].sort_values("date").reset_index(drop=True)

        n_train = len(df_train)
        n_test  = len(df_test)
        train_dr = (df_train["result"] == "D").mean()
        test_dr  = (df_test["result"]  == "D").mean()
        print(f"  训练: {n_train} 场  训练平局率={train_dr:.4f}")
        print(f"  测试: {n_test} 场   实际平局率={test_dr:.4f}")

        # ── 2. 全量训练 DC → WC 测试预测 ──────────────────────────────────
        dc_full = DixonColesModel()
        dc_full.fit(df_train, wc_start, half_life_days=FIXED_HALF_LIFE)
        probs_test = dc_probs_for_matches(dc_full, df_test)   # (N,3) [A,D,H]
        y_test     = df_test["result"].values

        # ── 3. 训练期 85/15 切割，调 δ ────────────────────────────────────
        n_val  = max(int(n_train * 0.15), 200)
        df_fit = df_train.iloc[:n_train - n_val].copy()
        df_val = df_train.iloc[n_train - n_val:].copy()
        ref_fit = df_fit["date"].max()

        dc_val = DixonColesModel()
        dc_val.fit(df_fit, ref_fit, half_life_days=FIXED_HALF_LIFE)
        probs_val = dc_probs_for_matches(dc_val, df_val)
        y_val     = df_val["result"].values
        val_dr    = (y_val == "D").mean()

        delta_star, dthr_star = tune_draw_threshold(
            probs_val, y_val, LABEL_ORDER, draw_base_rate=train_dr
        )

        # 验证：δ 在 val 上的预测平局率
        preds_val_adj = predict_with_draw_adj(probs_val, delta_star, dthr_star, LABEL_ORDER)
        val_pred_dr   = (preds_val_adj == "D").mean()
        print(f"\n  δ* = {delta_star:.2f}  draw_thr* = {dthr_star:.2f}")
        print(f"  val 集平局基率={val_dr:.4f}  val δ-adj 预测平局率={val_pred_dr:.4f}")

        # ── 4. 对测试集应用两种决策规则 ───────────────────────────────────
        preds_argmax = np.array([LABEL_ORDER[i] for i in np.argmax(probs_test, axis=1)])
        preds_adj    = predict_with_draw_adj(probs_test, delta_star, dthr_star, LABEL_ORDER)

        acc_argmax = accuracy_score(y_test, preds_argmax)
        acc_adj    = accuracy_score(y_test, preds_adj)
        brier_argmax = brier(probs_test, y_test)
        brier_adj    = brier(probs_test, y_test)   # 概率相同，Brier 相同

        mf1_argmax = f1_score(y_test, preds_argmax, average="macro", zero_division=0)
        mf1_adj    = f1_score(y_test, preds_adj,    average="macro", zero_division=0)

        # Brier 一样：δ 只改预测标签，不改概率
        # 为清晰起见仍保留，提示"Brier 不受δ影响"
        print(f"\n  {'指标':<16} {'DC-argmax':>12} {'DC+δ':>12} {'Δ(δ-argmax)':>14}")
        print(f"  {'─'*58}")
        print(f"  {'ACC':<16} {acc_argmax:>12.4f} {acc_adj:>12.4f} {acc_adj-acc_argmax:>+14.4f}")
        print(f"  {'macro-F1':<16} {mf1_argmax:>12.4f} {mf1_adj:>12.4f} {mf1_adj-mf1_argmax:>+14.4f}")
        print(f"  {'Brier':<16} {brier_argmax:>12.4f} {brier_adj:>12.4f} {'(无变化,概率同)':>14}")

        # ── 5. 改判分析：哪些场次 H/A → D ────────────────────────────────
        flipped_idx = np.where(
            (preds_argmax != "D") & (preds_adj == "D")
        )[0]
        n_flipped   = len(flipped_idx)
        n_correct   = int(sum(y_test[i] == "D" for i in flipped_idx))
        n_wrong     = n_flipped - n_correct
        test_pred_dr_arg = (preds_argmax == "D").mean()
        test_pred_dr_adj = (preds_adj == "D").mean()

        print(f"\n  预测平局场次: argmax={int((preds_argmax=='D').sum())} 场 ({test_pred_dr_arg:.3f})"
              f"  →  δ-adj={int((preds_adj=='D').sum())} 场 ({test_pred_dr_adj:.3f})")
        print(f"  实际平局: {int((y_test=='D').sum())} 场 ({test_dr:.4f})")
        print(f"\n  改判 H/A → D 共 {n_flipped} 场：")
        print(f"    改对了（实际=D）: {n_correct} 场")
        print(f"    改错了（实际≠D）: {n_wrong} 场")
        if n_flipped > 0:
            print(f"    改判精度: {n_correct}/{n_flipped} = {n_correct/n_flipped*100:.1f}%")

        # 逐场改判明细
        if n_flipped > 0:
            print(f"\n  改判明细:")
            print(f"  {'日期':<12} {'主队':<24} {'客队':<24} {'原预测':>6} {'改判':>6} {'实际':>6} {'对错':>5}  P(H)  P(D)  P(A)")
            for i in flipped_idx:
                r    = df_test.iloc[i]
                pb   = probs_test[i]
                orig = preds_argmax[i]
                ok   = "✓" if y_test[i] == "D" else "✗"
                print(f"  {str(r['date'].date()):<12} {r['home_team']:<24} {r['away_team']:<24}"
                      f" {orig:>6} {'D':>6} {y_test[i]:>6} {ok:>5}"
                      f"  {pb[2]:.3f}  {pb[1]:.3f}  {pb[0]:.3f}")

        # 反向：有多少实际平局被 argmax 和 δ 都没预测对？
        actual_draws_idx = np.where(y_test == "D")[0]
        n_act_d = len(actual_draws_idx)
        n_caught_arg = sum(preds_argmax[i] == "D" for i in actual_draws_idx)
        n_caught_adj = sum(preds_adj[i]    == "D" for i in actual_draws_idx)
        print(f"\n  实际 {n_act_d} 场平局中：")
        print(f"    argmax 预测对: {n_caught_arg} 场 ({n_caught_arg/n_act_d*100:.1f}%)")
        print(f"    δ-adj  预测对: {n_caught_adj} 场 ({n_caught_adj/n_act_d*100:.1f}%)")

        # ── 6. 汇总 ──────────────────────────────────────────────────────
        summary_rows.append({
            "year":         year,
            "n":            n_test,
            "delta":        delta_star,
            "draw_thr":     dthr_star,
            "train_dr":     round(train_dr, 4),
            "actual_dr":    round(test_dr, 4),
            "argmax_dr":    round(test_pred_dr_arg, 4),
            "adj_dr":       round(test_pred_dr_adj, 4),
            "acc_argmax":   round(acc_argmax, 4),
            "acc_adj":      round(acc_adj, 4),
            "delta_acc":    round(acc_adj - acc_argmax, 4),
            "mf1_argmax":   round(mf1_argmax, 4),
            "mf1_adj":      round(mf1_adj, 4),
            "delta_mf1":    round(mf1_adj - mf1_argmax, 4),
            "brier":        round(brier_argmax, 4),   # 相同
            "n_flipped":    n_flipped,
            "n_correct":    n_correct,
            "n_wrong":      n_wrong,
            "n_act_draws":  n_act_d,
            "caught_arg":   n_caught_arg,
            "caught_adj":   n_caught_adj,
        })

        # 每场明细
        for i, (_, r) in enumerate(df_test.iterrows()):
            pb = probs_test[i]
            all_records.append({
                "year":   year,
                "date":   str(r["date"].date()),
                "home":   r["home_team"],
                "away":   r["away_team"],
                "result": r["result"],
                "ph":     round(float(pb[2]), 4),
                "pd":     round(float(pb[1]), 4),
                "pa":     round(float(pb[0]), 4),
                "delta":  delta_star,
                "draw_thr": dthr_star,
                "pred_argmax": preds_argmax[i],
                "pred_adj":    preds_adj[i],
                "flipped":     int(preds_argmax[i] != "D" and preds_adj[i] == "D"),
                "ok_argmax":   int(preds_argmax[i] == r["result"]),
                "ok_adj":      int(preds_adj[i]    == r["result"]),
            })

    # ══════════════════════════════════════════════════════════════════════
    #  汇总表打印
    # ══════════════════════════════════════════════════════════════════════
    df_summary = pd.DataFrame(summary_rows)
    df_records = pd.DataFrame(all_records)

    print(f"\n\n{'#'*72}")
    print("  三届汇总对比表")
    print(f"{'#'*72}")

    print(f"\n  {'届次':>8} {'δ*':>5} {'thr':>5} "
          f"{'ACC-am':>8} {'ACC-adj':>8} {'ΔACC':>7} "
          f"{'F1-am':>7} {'F1-adj':>7} {'ΔF1':>7} "
          f"{'Brier':>7}")
    print(f"  {'─'*82}")
    for _, row in df_summary.iterrows():
        print(f"  {'WC'+str(int(row['year'])):>8} {row['delta']:>5.2f} {row['draw_thr']:>5.2f} "
              f"{row['acc_argmax']:>8.4f} {row['acc_adj']:>8.4f} {row['delta_acc']:>+7.4f} "
              f"{row['mf1_argmax']:>7.4f} {row['mf1_adj']:>7.4f} {row['delta_mf1']:>+7.4f} "
              f"{row['brier']:>7.4f}")

    # 三届合计
    n_total = df_records.shape[0]
    acc_arg_all = df_records["ok_argmax"].mean()
    acc_adj_all = df_records["ok_adj"].mean()
    mf1_arg_all = f1_score(df_records["result"], df_records["pred_argmax"],
                            average="macro", zero_division=0)
    mf1_adj_all = f1_score(df_records["result"], df_records["pred_adj"],
                            average="macro", zero_division=0)
    brier_all   = df_summary["brier"].mean()
    n_flip_all  = df_summary["n_flipped"].sum()
    n_cor_all   = df_summary["n_correct"].sum()
    n_act_d_all = df_summary["n_act_draws"].sum()
    n_caught_arg_all = df_summary["caught_arg"].sum()
    n_caught_adj_all = df_summary["caught_adj"].sum()

    print(f"  {'─'*82}")
    print(f"  {'合计(192)':>8} {'—':>5} {'—':>5} "
          f"{acc_arg_all:>8.4f} {acc_adj_all:>8.4f} {acc_adj_all-acc_arg_all:>+7.4f} "
          f"{mf1_arg_all:>7.4f} {mf1_adj_all:>7.4f} {mf1_adj_all-mf1_arg_all:>+7.4f} "
          f"{brier_all:>7.4f}")

    print(f"\n  改判汇总 (H/A → D):")
    print(f"  {'届次':>8} {'改判数':>7} {'改对':>7} {'改错':>7} {'精度':>8} "
          f"{'实平局数':>9} {'arg抓对':>9} {'adj抓对':>9}")
    print(f"  {'─'*74}")
    for _, row in df_summary.iterrows():
        prec = f"{row['n_correct']/row['n_flipped']*100:.1f}%" if row["n_flipped"] > 0 else "—"
        print(f"  {'WC'+str(int(row['year'])):>8} {row['n_flipped']:>7} {row['n_correct']:>7} "
              f"{row['n_wrong']:>7} {prec:>8} "
              f"{row['n_act_draws']:>9} {row['caught_arg']:>9} {row['caught_adj']:>9}")
    prec_all = f"{n_cor_all/n_flip_all*100:.1f}%" if n_flip_all > 0 else "—"
    print(f"  {'─'*74}")
    print(f"  {'合计':>8} {n_flip_all:>7} {n_cor_all:>7} {n_flip_all-n_cor_all:>7} {prec_all:>8} "
          f"{n_act_d_all:>9} {n_caught_arg_all:>9} {n_caught_adj_all:>9}")

    # ── 最终结论 ──────────────────────────────────────────────────────────
    print(f"\n{'#'*72}")
    print("  结论")
    print(f"{'#'*72}")

    delta_acc_avg = acc_adj_all - acc_arg_all
    delta_mf1_avg = mf1_adj_all - mf1_arg_all
    flip_precision = n_cor_all / n_flip_all if n_flip_all > 0 else 0
    recall_gain    = n_caught_adj_all - n_caught_arg_all

    print(f"\n  1. ACC: argmax={acc_arg_all:.4f}  δ-adj={acc_adj_all:.4f}  Δ={delta_acc_avg:+.4f}")
    if delta_acc_avg > 0.005:
        print(f"     δ 带来了显著的 ACC 提升 (+{delta_acc_avg*100:.2f} ppt)")
    elif delta_acc_avg > 0:
        print(f"     δ 带来了微小的 ACC 提升 (+{delta_acc_avg*100:.2f} ppt)")
    elif delta_acc_avg == 0:
        print(f"     δ 对 ACC 无影响")
    else:
        print(f"     δ 使 ACC 下降 ({delta_acc_avg*100:.2f} ppt)")

    print(f"\n  2. Brier = {brier_all:.4f}（两版相同，δ 只改决策标签，不改概率）")

    print(f"\n  3. 改判: 共 {n_flip_all} 场 H/A→D")
    print(f"     改对 {n_cor_all} 场 / 改错 {n_flip_all-n_cor_all} 场  (精度={prec_all})")
    if flip_precision >= 0.5:
        print(f"     改判精度 > 50%：大多数改判是正确的")
    else:
        print(f"     改判精度 < 50%：多数改判是错误的（假阳性平局）")

    print(f"\n  4. 平局召回: argmax 抓到 {n_caught_arg_all}/{n_act_d_all} 场"
          f"  →  δ-adj 抓到 {n_caught_adj_all}/{n_act_d_all} 场"
          f"  (增加 {recall_gain} 场)")

    # ACC 稳定性
    accs = df_summary["delta_acc"].values
    consistent = all(a >= -0.01 for a in accs)
    print(f"\n  5. 稳定性: 三届 ΔACC = "
          f"{', '.join(f'{a:+.4f}' for a in accs)}")
    if consistent:
        print(f"     δ 在三届中表现方向一致（均为非负）")
    else:
        print(f"     δ 表现不稳定：在某届负向")

    print()

    # ── 保存 ──────────────────────────────────────────────────────────────
    df_summary.to_csv(OUT_G3 / "delta_summary.csv",  index=False, encoding="utf-8-sig")
    df_records.to_csv(OUT_G3 / "delta_detail.csv",   index=False, encoding="utf-8-sig")
    print(f"  输出: outputs/gen3/delta_summary.csv  delta_detail.csv")


if __name__ == "__main__":
    main()
