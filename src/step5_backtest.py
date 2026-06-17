"""
阶段4: 滚动回测
输入: data/processed/features.csv, data/raw/shootouts.csv, outputs/train_params.json
输出: outputs/backtest_YYYY.csv, 控制台汇总

每届独立训练（严禁使用开赛日后的数据）。
δ在各届训练数据内分别选取（不共享）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent))
from step4_train import (
    TRAIN_START, CLASSES,
    get_feature_cols, predict_with_draw_adj, fit_pipeline, compute_metrics,
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"

# 各届世界杯开赛日（严禁训练使用此日期及之后的数据）
WC_OPENING = {
    2014: pd.Timestamp("2014-06-12"),
    2018: pd.Timestamp("2018-06-14"),
    2022: pd.Timestamp("2022-11-20"),
}


# ══════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════

def get_stage(row) -> str:
    """从特征列推断比赛阶段"""
    if row.get("stage_group_r1", 0) == 1:  return "group_r1"
    if row.get("stage_group_r2", 0) == 1:  return "group_r2"
    if row.get("stage_group_r3", 0) == 1:  return "group_r3"
    if row.get("is_knockout", 0) == 1:     return "knockout"
    return "other"


def lookup_pk_winner(row, shootouts_df: pd.DataFrame) -> str | None:
    """
    查找点球大战胜者（用于淘汰赛平局→口径B真实结果）。
    返回 'home' / 'away' / None（未找到则跳过该场）
    """
    d = str(row["date"].date())
    ht, at = row["home_team"], row["away_team"]
    mask = (
        (shootouts_df["date"].astype(str) == d) &
        (
            ((shootouts_df["home_team"] == ht) & (shootouts_df["away_team"] == at)) |
            ((shootouts_df["home_team"] == at) & (shootouts_df["away_team"] == ht))
        )
    )
    hits = shootouts_df[mask]
    if len(hits) == 0:
        return None
    winner = hits.iloc[0]["winner"]
    return "home" if winner == ht else "away"


def true_knockout_winner(row, shootouts_df: pd.DataFrame) -> str | None:
    """
    淘汰赛真实晋级方（用于口径B评估）：
    H → 'home'  |  A → 'away'  |  D → 查点球表
    """
    if row["result"] == "H":   return "home"
    if row["result"] == "A":   return "away"
    return lookup_pk_winner(row, shootouts_df)


def prob_home_advance(probs_row: np.ndarray, label_order: list) -> float:
    """口径B: P(主队晋级) = P(H) + 0.5×P(D)"""
    ph = probs_row[label_order.index("H")]
    pd_ = probs_row[label_order.index("D")]
    return ph + 0.5 * pd_


def elo_wins_baseline(df_wc: pd.DataFrame, label_order: list):
    """基线a: Elo高者获胜（主场+100）, 不预测平局"""
    neutral = df_wc["neutral"].fillna(0).values.astype(float)
    elo_h   = df_wc["elo_home_pre"].values
    elo_a   = df_wc["elo_away_pre"].values
    preds   = np.where(elo_h + 100 * (1 - neutral) > elo_a, "H", "A")
    tiny    = 0.01
    probs   = np.full((len(preds), len(label_order)), tiny)
    for k, p in enumerate(preds):
        probs[k, label_order.index(p)] = 1.0 - (len(label_order) - 1) * tiny
    return preds, probs


def random_freq_baseline(y_train: np.ndarray, n: int, label_order: list, seed: int = 42):
    """基线b: 按训练期频率随机猜"""
    rng  = np.random.RandomState(seed)
    cnts = {c: (y_train == c).sum() for c in label_order}
    p    = np.array([cnts.get(c, 0) for c in label_order], dtype=float)
    p   /= p.sum()
    preds = rng.choice(label_order, size=n, p=p)
    probs = np.tile(p, (n, 1))
    return preds, probs


def subset_metrics(df_r: pd.DataFrame, pred_col: str, prob_cols: list,
                   label_order: list, tag: str = "") -> dict:
    """对 df_r 子集计算准确率/macro-F1/Brier/LL"""
    if len(df_r) == 0:
        return {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan"),
                "brier": float("nan"), "log_loss": float("nan")}
    y    = df_r["result"].values
    pred = df_r[pred_col].values
    prob = df_r[prob_cols].values
    m    = compute_metrics(y, pred, prob, label_order, tag)
    return {"n": len(df_r), **m}


def knockout_oral_b_acc(df_ko: pd.DataFrame, pred_advance_col: str,
                        true_advance_col: str) -> dict:
    """口径B评估：谁晋级"""
    valid = df_ko[df_ko[true_advance_col].notna()].copy()
    if len(valid) == 0:
        return {"n_b": 0, "acc_b": float("nan")}
    correct = (valid[pred_advance_col] == valid[true_advance_col]).mean()
    return {"n_b": len(valid), "acc_b": float(correct)}


def print_stage_table(year: str, stage_rows: list[dict]) -> None:
    """按阶段打印结果表"""
    hdr = (f"{'阶段':<14} {'n':>4} "
           f"{'BLa':>6} {'BLb':>6} "
           f"{'LR':>6} {'LR+D':>6} "
           f"{'XGB':>6} {'XGB+D':>6}")
    print(f"\n--- {year} 各阶段准确率（argmax / +draw_adj） ---")
    print(hdr)
    print("-" * 60)
    for r in stage_rows:
        print(f"{r['stage']:<14} {r['n']:>4} "
              f"{r.get('bla_acc', float('nan')):>6.3f} "
              f"{r.get('blb_acc', float('nan')):>6.3f} "
              f"{r.get('lr_acc',  float('nan')):>6.3f} "
              f"{r.get('lr_adj_acc', float('nan')):>6.3f} "
              f"{r.get('xgb_acc', float('nan')):>6.3f} "
              f"{r.get('xgb_adj_acc', float('nan')):>6.3f}")
    print("-" * 60)


def print_macro_f1_table(year: str, stage_rows: list[dict]) -> None:
    """同上，改成 macro-F1"""
    hdr = (f"{'阶段':<14} {'n':>4} "
           f"{'LR_mF1':>8} {'LR+D_mF1':>9} "
           f"{'XGB_mF1':>8} {'XGB+D_mF1':>10}")
    print(f"\n--- {year} 各阶段 macro-F1 ---")
    print(hdr)
    print("-" * 55)
    for r in stage_rows:
        print(f"{r['stage']:<14} {r['n']:>4} "
              f"{r.get('lr_mf1',  float('nan')):>8.4f} "
              f"{r.get('lr_adj_mf1', float('nan')):>9.4f} "
              f"{r.get('xgb_mf1', float('nan')):>8.4f} "
              f"{r.get('xgb_adj_mf1', float('nan')):>10.4f}")
    print("-" * 55)


def print_5_confident_errors(df_r: pd.DataFrame, model: str, label_order: list) -> None:
    """打印最自信的5个错误预测"""
    prob_cols = [f"{model}_prob_{c}" for c in label_order]
    pred_col  = f"{model}_argmax"

    wrong = df_r[df_r[pred_col] != df_r["result"]].copy()
    if len(wrong) == 0:
        print("  无预测错误（不应发生）")
        return
    wrong["max_prob"] = wrong[prob_cols].max(axis=1)
    top5 = wrong.nlargest(5, "max_prob")

    print(f"\n{'主队':<22} {'客队':<22} {'预测':>5} {'真实':>5} {'最高概率':>8}")
    print("-" * 68)
    for _, row in top5.iterrows():
        print(f"{row['home_team']:<22} {row['away_team']:<22} "
              f"{row[pred_col]:>5} {row['result']:>5} {row['max_prob']:>8.4f}")


# ══════════════════════════════════════════════
#  核心回测函数
# ══════════════════════════════════════════════

def run_wc_backtest(
    year: int,
    df: pd.DataFrame,
    feat_cols: list[str],
    lr_params: dict,
    xgb_params: dict,
    shootouts_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    对指定届世界杯运行滚动回测。
    返回 (df_results, metrics_by_stage)
    """
    opening = WC_OPENING[year]

    # ── 训练数据（严禁越界）──────────────────────
    df_train = df[(df["date"] >= TRAIN_START) & (df["date"] < opening)].copy()
    print(f"\n{'='*64}")
    print(f"  {year} WC 回测")
    print(f"  训练: {TRAIN_START} ~ {opening.date()-pd.Timedelta(days=1)}  ({len(df_train)} 场)")
    print(f"{'='*64}")

    # ── 训练模型（δ 在训练数据内自动选取）─────────
    pkg_lr = fit_pipeline(
        df_train, "lr", feat_cols=feat_cols,
        lr_C=lr_params["C"], lam=lr_params["lam"],
    )
    pkg_xgb = fit_pipeline(
        df_train, "xgb", feat_cols=feat_cols,
        xgb_max_depth=xgb_params["max_depth"],
        xgb_lr=xgb_params["lr"],
        xgb_n_est=xgb_params["n_est"],
        lam=xgb_params["lam"],
    )

    lo   = pkg_xgb["label_order"]   # ['A','D','H']
    lo_lr = pkg_lr["label_order"]

    print(f"  LR  δ={pkg_lr['delta']:.2f}  draw_thr={pkg_lr['draw_thr']:.2f}  "
          f"(λ={pkg_lr['lam']:.2f})")
    print(f"  XGB δ={pkg_xgb['delta']:.2f}  draw_thr={pkg_xgb['draw_thr']:.2f}  "
          f"(λ={pkg_xgb['lam']:.2f})")

    # ── WC 测试数据 ──────────────────────────────
    df_wc = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"].dt.year == year)
    ].sort_values("date").copy()
    print(f"  WC 场次: {len(df_wc)}  ({df_wc['date'].min().date()} ~ {df_wc['date'].max().date()})")

    # ── 特征矩阵 ────────────────────────────────
    X_wc     = df_wc[feat_cols].values.astype(np.float32)
    X_wc_s   = pkg_lr["scaler"].transform(X_wc)   # LR 需要标准化

    # ── 概率预测 ────────────────────────────────
    probs_lr  = pkg_lr["calibrated_model"].predict_proba(X_wc_s)   # (N,3) lo_lr顺序
    probs_xgb = pkg_xgb["calibrated_model"].predict_proba(X_wc)   # (N,3) lo顺序

    # 确保两个模型label_order一致（都是 ['A','D','H']）
    assert lo == lo_lr, f"label order mismatch: {lo} vs {lo_lr}"

    # ── 预测决策 ────────────────────────────────
    pred_lr_raw  = np.array([lo[i] for i in np.argmax(probs_lr,  axis=1)])
    pred_lr_adj  = predict_with_draw_adj(probs_lr,  pkg_lr["delta"],  pkg_lr["draw_thr"],  lo)
    pred_xgb_raw = np.array([lo[i] for i in np.argmax(probs_xgb, axis=1)])
    pred_xgb_adj = predict_with_draw_adj(probs_xgb, pkg_xgb["delta"], pkg_xgb["draw_thr"], lo)

    # ── 基线 ────────────────────────────────────
    preds_bla, probs_bla = elo_wins_baseline(df_wc, lo)
    preds_blb, probs_blb = random_freq_baseline(df_train["result"].values, len(df_wc), lo)

    # ── 口径B（淘汰赛晋级预测）────────────────────
    adv_xgb  = np.array(["home" if prob_home_advance(probs_xgb[i], lo) > 0.5 else "away"
                          for i in range(len(df_wc))])
    adv_lr   = np.array(["home" if prob_home_advance(probs_lr[i], lo) > 0.5 else "away"
                          for i in range(len(df_wc))])

    # 真实晋级方（D → 查点球表）
    true_adv = []
    for _, row in df_wc.iterrows():
        if row["is_knockout"] == 1:
            true_adv.append(true_knockout_winner(row, shootouts_df))
        else:
            # 小组赛不需要晋级方
            true_adv.append(None)

    # ── 组装明细 DataFrame ───────────────────────
    df_results = df_wc[["date", "home_team", "away_team", "result",
                         "elo_home_pre", "elo_away_pre", "neutral"]].copy()

    df_results["stage"]      = [get_stage(row) for _, row in df_wc.iterrows()]

    # LR 概率
    for j, c in enumerate(lo):
        df_results[f"lr_prob_{c}"]  = probs_lr[:, j]
    # XGB 概率
    for j, c in enumerate(lo):
        df_results[f"xgb_prob_{c}"] = probs_xgb[:, j]

    # 基线概率
    for j, c in enumerate(lo):
        df_results[f"bla_prob_{c}"] = probs_bla[:, j]
        df_results[f"blb_prob_{c}"] = probs_blb[:, j]

    df_results["lr_argmax"]    = pred_lr_raw
    df_results["lr_draw_adj"]  = pred_lr_adj
    df_results["xgb_argmax"]   = pred_xgb_raw
    df_results["xgb_draw_adj"] = pred_xgb_adj
    df_results["bla_pred"]     = preds_bla
    df_results["blb_pred"]     = preds_blb

    # 口径A 是否正确
    df_results["lr_correct"]      = (df_results["lr_argmax"]   == df_results["result"]).astype(int)
    df_results["lr_adj_correct"]  = (df_results["lr_draw_adj"] == df_results["result"]).astype(int)
    df_results["xgb_correct"]     = (df_results["xgb_argmax"]  == df_results["result"]).astype(int)
    df_results["xgb_adj_correct"] = (df_results["xgb_draw_adj"]== df_results["result"]).astype(int)

    # 口径B（淘汰赛晋级预测）
    df_results["xgb_adv_pred"] = adv_xgb
    df_results["lr_adv_pred"]  = adv_lr
    df_results["true_advance"] = true_adv   # 'home'/'away'/None

    # ── 保存明细 CSV ────────────────────────────
    csv_path = OUTPUTS_DIR / f"backtest_{year}.csv"
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  明细已保存: {csv_path.name}")

    # ══════════════════════════════════════════
    # 分阶段评估
    # ══════════════════════════════════════════
    lr_prob_cols  = [f"lr_prob_{c}"  for c in lo]
    xgb_prob_cols = [f"xgb_prob_{c}" for c in lo]
    bla_prob_cols = [f"bla_prob_{c}" for c in lo]
    blb_prob_cols = [f"blb_prob_{c}" for c in lo]

    stage_subsets = {
        "Overall":  df_results,
        "Group R1-R2": df_results[df_results["stage"].isin(["group_r1", "group_r2"])],
        "Group R3":    df_results[df_results["stage"] == "group_r3"],
        "Knockout(A)": df_results[df_results["stage"] == "knockout"],
    }

    stage_rows   = []
    metrics_dict = {}

    for stage_name, sub in stage_subsets.items():
        n = len(sub)
        if n == 0:
            stage_rows.append({"stage": stage_name, "n": 0})
            continue

        y_sub = sub["result"].values

        def _acc(pred_col):
            return accuracy_score(y_sub, sub[pred_col].values)

        def _mf1(pred_col):
            return f1_score(y_sub, sub[pred_col].values,
                            average="macro", zero_division=0)

        def _brier(prob_cols_):
            yb = label_binarize(y_sub, classes=lo)
            p  = sub[prob_cols_].values
            return float(np.mean([brier_score_loss(yb[:, i], p[:, i])
                                  for i in range(len(lo))]))

        def _ll(prob_cols_):
            return log_loss(y_sub, sub[prob_cols_].values, labels=lo)

        row = {
            "stage": stage_name, "n": n,
            "bla_acc":     _acc("bla_pred"),
            "blb_acc":     _acc("blb_pred"),
            "lr_acc":      _acc("lr_argmax"),
            "lr_adj_acc":  _acc("lr_draw_adj"),
            "xgb_acc":     _acc("xgb_argmax"),
            "xgb_adj_acc": _acc("xgb_draw_adj"),
            "lr_mf1":      _mf1("lr_argmax"),
            "lr_adj_mf1":  _mf1("lr_draw_adj"),
            "xgb_mf1":     _mf1("xgb_argmax"),
            "xgb_adj_mf1": _mf1("xgb_draw_adj"),
            "xgb_brier":   _brier(xgb_prob_cols),
            "xgb_ll":      _ll(xgb_prob_cols),
            "lr_brier":    _brier(lr_prob_cols),
            "lr_ll":       _ll(lr_prob_cols),
        }
        stage_rows.append(row)
        metrics_dict[stage_name] = row

    # 口径B（淘汰赛晋级）
    ko_sub = df_results[df_results["stage"] == "knockout"].copy()
    if len(ko_sub) > 0:
        ko_valid = ko_sub[ko_sub["true_advance"].notna()].copy()
        n_b   = len(ko_valid)
        if n_b > 0:
            acc_xgb_b = (ko_valid["xgb_adv_pred"] == ko_valid["true_advance"]).mean()
            acc_lr_b  = (ko_valid["lr_adv_pred"]  == ko_valid["true_advance"]).mean()
            ko_b_row = {
                "stage": "Knockout(B)", "n": n_b,
                "lr_acc": acc_lr_b, "xgb_acc": acc_xgb_b,
                # baselines for oral B
                "bla_acc": (ko_valid.apply(
                    lambda r: ("home" if r["bla_pred"] in ["H", "D"] else "away") == r["true_advance"], axis=1
                ).mean()),
                "blb_acc": (ko_valid.apply(
                    lambda r: ("home" if r["blb_pred"] in ["H", "D"] else "away") == r["true_advance"], axis=1
                ).mean()),
            }
            stage_rows.append(ko_b_row)
            metrics_dict["Knockout(B)"] = ko_b_row

    # ── 打印 ─────────────────────────────────────
    print_stage_table(str(year), stage_rows)
    print_macro_f1_table(str(year), stage_rows)

    print(f"\n  Brier / LogLoss (XGB argmax):")
    for r in stage_rows:
        if r["n"] > 0 and "xgb_brier" in r:
            print(f"    {r['stage']:<14}: Brier={r.get('xgb_brier', float('nan')):.4f}"
                  f"  LL={r.get('xgb_ll', float('nan')):.4f}")

    print(f"\n  最自信的5个错误 — XGB argmax")
    print_5_confident_errors(df_results, "xgb", lo)

    return df_results, metrics_dict


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("读取数据...")
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    shootouts_df = pd.read_csv(RAW_DIR / "shootouts.csv", parse_dates=["date"])

    feat_cols = get_feature_cols(df)
    print(f"特征数: {len(feat_cols)}")

    # 读取step4确定的最优超参数
    with open(OUTPUTS_DIR / "train_params.json", encoding="utf-8") as f:
        best_params = json.load(f)

    lr_params  = best_params["lr"]
    xgb_params = best_params["xgb"]
    print(f"LR: C={lr_params['C']}, λ={lr_params['lam']}")
    print(f"XGB: depth={xgb_params['max_depth']}, lr={xgb_params['lr']}, "
          f"n={xgb_params['n_est']}, λ={xgb_params['lam']}")

    # ── 三届回测 ─────────────────────────────────
    all_results  = {}
    all_metrics  = {}

    for year in [2014, 2018, 2022]:
        df_r, metrics = run_wc_backtest(
            year, df, feat_cols, lr_params, xgb_params, shootouts_df
        )
        all_results[year] = df_r
        all_metrics[year] = metrics

    # ══════════════════════════════════════════
    # 三届合并汇总表
    # ══════════════════════════════════════════
    print(f"\n\n{'#'*64}")
    print(f"  三届世界杯回测汇总（Overall）")
    print(f"{'#'*64}")

    summary_rows = []
    for year in [2014, 2018, 2022]:
        if "Overall" in all_metrics[year]:
            r = all_metrics[year]["Overall"]
            summary_rows.append({
                "届": str(year),
                "n": r["n"],
                "基线a_acc": r.get("bla_acc", float("nan")),
                "基线b_acc": r.get("blb_acc", float("nan")),
                "LR_acc":    r.get("lr_acc",  float("nan")),
                "LR+D_acc":  r.get("lr_adj_acc", float("nan")),
                "XGB_acc":   r.get("xgb_acc", float("nan")),
                "XGB+D_acc": r.get("xgb_adj_acc", float("nan")),
                "LR_mF1":   r.get("lr_mf1",   float("nan")),
                "XGB_mF1":  r.get("xgb_mf1",  float("nan")),
                "XGB_Brier": r.get("xgb_brier", float("nan")),
                "XGB_LL":    r.get("xgb_ll",  float("nan")),
            })

    # 计算三届均值
    def nanmean(vals):
        v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    avg_row = {"届": "均值", "n": sum(r["n"] for r in summary_rows)}
    for col in ["基线a_acc", "基线b_acc", "LR_acc", "LR+D_acc",
                "XGB_acc", "XGB+D_acc", "LR_mF1", "XGB_mF1", "XGB_Brier", "XGB_LL"]:
        avg_row[col] = nanmean([r[col] for r in summary_rows])
    summary_rows.append(avg_row)

    # 打印主表
    hdr = (f"{'届':<6} {'n':>4} {'BLa_acc':>8} {'BLb_acc':>8} "
           f"{'LR':>7} {'LR+D':>7} {'XGB':>7} {'XGB+D':>7}")
    print(hdr)
    print("-" * 62)
    for r in summary_rows:
        print(f"{r['届']:<6} {r['n']:>4} {r['基线a_acc']:>8.4f} {r['基线b_acc']:>8.4f} "
              f"{r['LR_acc']:>7.4f} {r['LR+D_acc']:>7.4f} "
              f"{r['XGB_acc']:>7.4f} {r['XGB+D_acc']:>7.4f}")
    print("-" * 62)

    # macro-F1 + Brier + LL
    print(f"\n{'届':<6} {'n':>4} {'LR_mF1':>8} {'XGB_mF1':>9} {'XGB_Brier':>10} {'XGB_LL':>8}")
    print("-" * 50)
    for r in summary_rows:
        print(f"{r['届']:<6} {r['n']:>4} {r['LR_mF1']:>8.4f} {r['XGB_mF1']:>9.4f} "
              f"{r['XGB_Brier']:>10.4f} {r['XGB_LL']:>8.4f}")
    print("-" * 50)

    # ── 分届淘汰赛口径B汇总 ────────────────────────
    print(f"\n{'='*48}")
    print("  淘汰赛口径B（谁晋级）")
    print(f"{'='*48}")
    print(f"{'届':<6} {'n_B':>5} {'BLa_B':>7} {'BLb_B':>7} {'LR_B':>7} {'XGB_B':>7}")
    print("-" * 40)
    acc_b_list = []
    for year in [2014, 2018, 2022]:
        kb = all_metrics[year].get("Knockout(B)", {})
        n_b     = kb.get("n", 0)
        bla_b   = kb.get("bla_acc", float("nan"))
        blb_b   = kb.get("blb_acc", float("nan"))
        lr_b    = kb.get("lr_acc",  float("nan"))
        xgb_b   = kb.get("xgb_acc", float("nan"))
        print(f"{year:<6} {n_b:>5} {bla_b:>7.4f} {blb_b:>7.4f} {lr_b:>7.4f} {xgb_b:>7.4f}")
        if not np.isnan(xgb_b):
            acc_b_list.append(xgb_b)

    avg_b = nanmean(acc_b_list) if acc_b_list else float("nan")
    print("-" * 40)
    print(f"{'均值':<6} {'':>5} {'':>7} {'':>7} {'':>7} {avg_b:>7.4f}")

    # ── 验收结论 ────────────────────────────────
    print(f"\n{'#'*64}")
    print("  验收检查")
    print(f"{'#'*64}")
    avg_xgb_acc = avg_row["XGB_acc"]
    avg_bla_acc = avg_row["基线a_acc"]
    print(f"  XGB argmax 三届平均准确率: {avg_xgb_acc:.4f}")
    print(f"  基线a（Elo高者胜）平均:    {avg_bla_acc:.4f}")
    check1 = "PASS" if avg_xgb_acc >= 0.48 else "FAIL (目标>=48%)"
    check2 = "PASS" if avg_xgb_acc > avg_bla_acc else "FAIL (应>基线a)"
    check3 = "PASS" if avg_b >= 0.60 else f"注: 口径B均值={avg_b:.3f} (预期60-75%)"
    print(f"  >= 48%?  {check1}")
    print(f"  > 基线a? {check2}")
    print(f"  口径B:   {check3}")
