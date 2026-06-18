"""
step10_init_2026.py — 2026 WC 一次性初始化
=============================================
冻结规格：DC + XGB+int，全量历史数据（截至2026-06-10）定版训练。
本脚本运行后模型本体不再重训。

输出：
  outputs/dc_model_2026.pkl         ← 冻结 DC 模型
  outputs/xgb_model_2026.pkl        ← 冻结 XGB 模型
  data/processed/wc2026_results.csv ← 本届已完赛（初始含2026-06-11两场）
  outputs/live_predictions_2026.csv ← 预测日志（今日预追加）
"""
from __future__ import annotations
import json, pickle, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step4_train import (
    TRAIN_START, get_feature_cols, fit_pipeline, predict_with_draw_adj,
)
from step5b_improve import DixonColesModel
from step5c_devset import add_interaction_features, fit_dc_fast, dc_probs_for_matches
from metrics import LABEL_ORDER, multiclass_brier, bla_probs_canonical
from live_features import (
    FEAT_COLS_INTER, build_feature_matrix,
    build_elo_series, get_current_elo,
)
from wc_data import WC_GROUPS, WC_HOSTS

PROC = Path(__file__).parent.parent / "data" / "processed"
RAW  = Path(__file__).parent.parent / "data" / "raw"
OUT  = Path(__file__).parent.parent / "outputs"

WC_OPENING = pd.Timestamp("2026-06-11")
WC_YEAR    = 2026
WC2026_GROUPS  = WC_GROUPS[2026]
WC2026_HOSTS   = WC_HOSTS[2026]
TEAM_TO_GROUP  = {t: g for g, ts in WC2026_GROUPS.items() for t in ts}

# ── 冻结规格 Elo 参数 ─────────────────────────────────────────
with open(OUT / "elo_best_params.json") as f:
    ELO_PARAMS = json.load(f)
print(f"[ELO] H_adv={ELO_PARAMS['H_adv']} K_wc={ELO_PARAMS['K_wc']} K_major={ELO_PARAMS['K_major']}")

# ── 已确认比赛结果（2026-06-11，人工核实）────────────────────
CONFIRMED_RESULTS = [
    # date, home, away, hs, as_, neutral, group, round
    ("2026-06-11", "Mexico",      "South Africa",  2, 0, False, "A", "R1"),
    ("2026-06-11", "South Korea", "Czech Republic",2, 1, True,  "A", "R1"),
    # 注：2026-06-12 比赛比分待用户提供（由 daily_update.py 填入）
]


def _result_label(hs, as_):
    return "H" if hs > as_ else ("A" if hs < as_ else "D")


# ══════════════════════════════════════════════
#  1. 加载历史训练数据
# ══════════════════════════════════════════════

def load_train_data():
    df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
    df = add_interaction_features(df.sort_values("date").reset_index(drop=True))
    df_train = df[(df["date"] >= pd.Timestamp(TRAIN_START)) &
                  (df["date"] < WC_OPENING)].copy()
    print(f"[数据] 训练集: {len(df_train)} 场  (截至 {df_train['date'].max().date()})")
    return df, df_train


# ══════════════════════════════════════════════
#  2. 训练并保存 DC 和 XGB 模型
# ══════════════════════════════════════════════

def train_and_save(df_train: pd.DataFrame):
    print("\n[训练] DC 双泊松 (half_life=730)...", end="", flush=True)
    dc = fit_dc_fast(df_train, WC_OPENING)
    with open(OUT / "dc_model_2026.pkl", "wb") as f:
        pickle.dump(dc, f)
    print(f" 完成  rho={dc.rho:.4f}  log_gamma={dc.log_gamma:.4f}")

    with open(OUT / "train_params.json") as f:
        tp = json.load(f)
    xgb_p = tp["xgb"]
    print(f"[训练] XGB+int  max_depth={xgb_p['max_depth']} lr={xgb_p['lr']}"
          f" n_est={xgb_p['n_est']}...", end="", flush=True)
    pkg = fit_pipeline(
        df_train, "xgb",
        feat_cols=FEAT_COLS_INTER,
        xgb_max_depth=xgb_p["max_depth"],
        xgb_lr=xgb_p["lr"],
        xgb_n_est=xgb_p["n_est"],
        lam=xgb_p["lam"],
    )
    with open(OUT / "xgb_model_2026.pkl", "wb") as f:
        pickle.dump(pkg, f)
    print(f" 完成  δ={pkg['delta']:.2f} thr={pkg['draw_thr']:.2f}")

    return dc, pkg


# ══════════════════════════════════════════════
#  3. 写入本届已完赛结果
# ══════════════════════════════════════════════

def init_live_results():
    hist_df = pd.read_csv(PROC / "matches_clean.csv", parse_dates=["date"])
    # 从 matches_clean 拿 k_factor
    kmap = (hist_df[["home_team","away_team","date","k_factor"]]
            .drop_duplicates().set_index(["date","home_team","away_team"])["k_factor"]
            .to_dict())

    rows = []
    for dt, ht, at, hs, as_, neu, grp, rnd in CONFIRMED_RESULTS:
        res = _result_label(hs, as_)
        kf  = kmap.get((pd.Timestamp(dt), ht, at), 60)
        rows.append({
            "date": dt, "home_team": ht, "away_team": at,
            "home_score": hs, "away_score": as_, "result": res,
            "neutral": neu, "tournament": "FIFA World Cup",
            "k_factor": kf, "group": grp, "round": rnd,
        })
    df_live = pd.DataFrame(rows)
    df_live["date"] = pd.to_datetime(df_live["date"])
    path = PROC / "wc2026_results.csv"
    df_live.to_csv(path, index=False)
    print(f"\n[结果] 已写入 wc2026_results.csv ({len(df_live)} 场)")
    return df_live


# ══════════════════════════════════════════════
#  4. 生成今日预测
# ══════════════════════════════════════════════

def generate_predictions(dc, xgb_pkg, df_live):
    # 加载完整历史（含已完赛）
    hist_clean = pd.read_csv(PROC / "matches_clean.csv", parse_dates=["date"])
    # 合并：历史有效比赛 + 本届 WC 已完赛
    all_matches = pd.concat([
        hist_clean[hist_clean["home_score"].notna()],
        df_live[df_live["home_score"].notna()],
    ]).sort_values("date").reset_index(drop=True)

    # 当前 Elo
    elo_series = build_elo_series(hist_clean, df_live, ELO_PARAMS)
    elo_dict   = get_current_elo(elo_series)

    # 取今日（2026-06-12）和明天（2026-06-13）的待预测场次
    sched = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    wc26  = sched[(sched["tournament"] == "FIFA World Cup") &
                  (sched["date"].dt.year == 2026) &
                  (sched["home_score"].isna())].sort_values("date")
    target_dates = [pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-13")]
    upcoming = wc26[wc26["date"].isin(target_dates)].copy()
    print(f"\n[预测] 目标场次: {len(upcoming)} 场 (6月12-13日)")

    # 计算特征
    X_df = build_feature_matrix(
        upcoming, WC2026_GROUPS, all_matches, elo_dict,
        df_live, TEAM_TO_GROUP
    )
    X = X_df[FEAT_COLS_INTER].values.astype("float32")

    # DC 预测
    dc_probs = dc_probs_for_matches(dc, upcoming)

    # XGB 预测
    xgb_probs = xgb_pkg["calibrated_model"].predict_proba(X)
    xgb_delta = xgb_pkg["delta"]
    xgb_thr   = xgb_pkg["draw_thr"]

    # 拼 BLa (参照) — 需要 elo_home_pre/away_pre，从 X_df 取
    hist_train = hist_clean[hist_clean["date"] < WC_OPENING]
    draw_rate = float((hist_train["result"] == "D").mean())
    H_ADV = 125.0
    bla_probs_list = []
    for i in range(len(X_df)):
        elo_h = float(X_df.iloc[i]["elo_home_pre"])
        elo_a = float(X_df.iloc[i]["elo_away_pre"])
        neu   = bool(upcoming.iloc[i]["neutral"])
        h_eff = elo_h + H_ADV * (0 if neu else 1)
        we    = 1.0 / (1.0 + 10.0 ** (-(h_eff - elo_a) / 400.0))
        bla_probs_list.append([
            (1 - we) * (1 - draw_rate),  # A
            draw_rate,                    # D
            we * (1 - draw_rate),         # H
        ])
    bla_probs_ = np.array(bla_probs_list)

    # 结果
    lo = LABEL_ORDER
    pred_rows = []
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for i, (_, row) in enumerate(upcoming.iterrows()):
        dc_p  = dc_probs[i]
        xp    = xgb_probs[i]
        bla_p = bla_probs_[i]
        dc_pred  = lo[np.argmax(dc_p)]
        xgb_pred = lo[np.argmax(xp)]
        xgb_adj  = predict_with_draw_adj(xp.reshape(1, -1), xgb_delta, xgb_thr, lo)[0]
        dc_conf  = float(np.max(dc_p))
        high_conf = dc_conf > 0.6

        pred_rows.append({
            "timestamp":   now_ts,
            "pred_date":   str(datetime.now().date()),
            "match_date":  str(row["date"].date()),
            "home_team":   row["home_team"],
            "away_team":   row["away_team"],
            "dc_pa":       round(float(dc_p[0]), 4),
            "dc_pd":       round(float(dc_p[1]), 4),
            "dc_ph":       round(float(dc_p[2]), 4),
            "dc_pred":     dc_pred,
            "xgb_pa":      round(float(xp[0]), 4),
            "xgb_pd":      round(float(xp[1]), 4),
            "xgb_ph":      round(float(xp[2]), 4),
            "xgb_pred":    xgb_pred,
            "xgb_adj":     xgb_adj,
            "bla_ph":      round(float(bla_p[2]), 4),
            "high_conf":   int(high_conf),
            "actual_result": "",
            "dc_correct":  "",
            "xgb_correct": "",
            "adj_correct": "",
            "manual_adj_applied": "",
        })

    df_pred = pd.DataFrame(pred_rows)

    # 追加写入（若文件不存在则建立）
    pred_path = OUT / "live_predictions_2026.csv"
    if pred_path.exists():
        df_pred.to_csv(pred_path, mode="a", header=False, index=False)
    else:
        df_pred.to_csv(pred_path, index=False)
    print(f"[预测] 已追加 {len(df_pred)} 行至 live_predictions_2026.csv")

    return df_pred, upcoming, X_df


# ══════════════════════════════════════════════
#  5. 打印预测表（严格前瞻，供用户对账）
# ══════════════════════════════════════════════

def print_predictions(df_pred, upcoming, X_df):
    elo_h = X_df["elo_home_pre"].values
    elo_a = X_df["elo_away_pre"].values

    print("\n" + "=" * 72)
    print("  2026-06-12 赛前预测（严格前瞻，DC主/XGB参）")
    print("=" * 72)
    j12 = df_pred[df_pred["match_date"] == "2026-06-12"]
    for i, (_, r) in enumerate(j12.iterrows()):
        row_sched = upcoming[upcoming["home_team"] == r["home_team"]].iloc[0]
        ei = upcoming.reset_index(drop=True).index[upcoming["home_team"] == r["home_team"]].tolist()
        ei = ei[0] if ei else i
        print(f"\n  {r['home_team']} vs {r['away_team']}")
        print(f"    Elo: {elo_h[ei]:.0f} vs {elo_a[ei]:.0f}  "
              f"neutral={row_sched['neutral']}")
        print(f"    DC :  P(H)={r['dc_ph']:.3f}  P(D)={r['dc_pd']:.3f}  "
              f"P(A)={r['dc_pa']:.3f}  → 点预测:{r['dc_pred']}"
              f"{'  ★高置信' if r['high_conf'] else ''}")
        print(f"    XGB:  P(H)={r['xgb_ph']:.3f}  P(D)={r['xgb_pd']:.3f}  "
              f"P(A)={r['xgb_pa']:.3f}  → argmax:{r['xgb_pred']} adj:{r['xgb_adj']}")
        print(f"    BLa:  P(H)={r['bla_ph']:.3f}  (参照)")

    print("\n" + "=" * 72)
    print("  2026-06-13 预测（6月13日全部场次）")
    print("=" * 72)
    j13 = df_pred[df_pred["match_date"] == "2026-06-13"]
    print(f"\n  {'主队':<22} {'客队':<22} {'DC:H/D/A':>12} {'预测':>5} {'XGB:H/D/A':>12} {'XGB':>5} {'BLa(H)':>7} {'★':>2}")
    print("  " + "-" * 85)
    for _, r in j13.iterrows():
        star = "★" if r["high_conf"] else ""
        print(f"  {r['home_team']:<22} {r['away_team']:<22} "
              f"{r['dc_ph']:.2f}/{r['dc_pd']:.2f}/{r['dc_pa']:.2f} {r['dc_pred']:>5}"
              f"  {r['xgb_ph']:.2f}/{r['xgb_pd']:.2f}/{r['xgb_pa']:.2f} {r['xgb_pred']:>5}"
              f"  {r['bla_ph']:.2f}  {star:>2}")

    print(f"\n  注: ★ = DC最高概率>0.6（高置信）")


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  step10_init_2026.py — 2026 WC 初始化")
    print("=" * 65)

    df_full, df_train = load_train_data()

    # 检查模型是否已存在（跳过重训）
    dc_path  = OUT / "dc_model_2026.pkl"
    xgb_path = OUT / "xgb_model_2026.pkl"
    if dc_path.exists() and xgb_path.exists():
        print("\n[跳过] 模型文件已存在，直接加载...")
        with open(dc_path, "rb")  as f: dc      = pickle.load(f)
        with open(xgb_path, "rb") as f: xgb_pkg = pickle.load(f)
        print(f"  DC: rho={dc.rho:.4f}  log_gamma={dc.log_gamma:.4f}")
        print(f"  XGB: δ={xgb_pkg['delta']:.2f} thr={xgb_pkg['draw_thr']:.2f}")
    else:
        dc, xgb_pkg = train_and_save(df_train)

    df_live = init_live_results()
    df_pred, upcoming, X_df = generate_predictions(dc, xgb_pkg, df_live)
    print_predictions(df_pred, upcoming, X_df)

    print("\n" + "=" * 65)
    print("  初始化完成。请告知 2026-06-12 两场比赛比分，")
    print("  再运行 daily_update.py 录入并看 6月13日完整预测。")
    print("=" * 65)


if __name__ == "__main__":
    main()
