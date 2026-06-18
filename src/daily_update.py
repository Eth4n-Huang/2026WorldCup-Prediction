"""
daily_update.py — 2026 WC 每日滚动预测
==========================================
每天运行一次。流程:
  1. 录入昨日比分 (WebFetch 维基百科 → 回退手动)
  2. 更新动态特征 (Elo/积分/休息天数)，模型本体不重训
  3. 生成今日预测 (DC主 + XGB对照)
  4. 结算昨日预测对错，打印滚动 ACC/Brier vs 回测基准
  5. 可选: 人工伤病 Elo 修正

用法: python src/daily_update.py [--date YYYY-MM-DD] [--skip-fetch] [--predict-only]
  --date         指定运行日期 (默认今天)
  --skip-fetch   跳过 WebFetch，直接手动录入
  --predict-only 只生成预测，不录入比分 (调试用)
"""
from __future__ import annotations
import argparse, json, pickle, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from step4_train      import predict_with_draw_adj
from step5c_devset    import dc_probs_for_matches, dc_top_scores_for_matches
from metrics          import LABEL_ORDER, multiclass_brier
from live_features    import (
    FEAT_COLS_INTER, build_feature_matrix,
    build_elo_series, get_current_elo, get_round,
)
from wc_data          import WC_GROUPS, WC_HOSTS

# ── 路径 ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "outputs"

# ── 常量 ─────────────────────────────────────────────────────
WC_OPENING    = pd.Timestamp("2026-06-11")
WC2026_GROUPS = WC_GROUPS[2026]
WC2026_HOSTS  = WC_HOSTS[2026]
TEAM_TO_GROUP = {t: g for g, ts in WC2026_GROUPS.items() for t in ts}
H_ADV         = 125.0
WC_KFACTOR    = 60

# 回测基准 (三届 2014/2018/2022 冻结规格均值)
BACKTEST_ACC_DC   = 0.5521
BACKTEST_BRIER_DC = 0.5794

# live_predictions_2026.csv 列顺序（与 step10 保持一致）
PRED_COLS = [
    "timestamp", "pred_date", "match_date", "home_team", "away_team",
    "dc_pa", "dc_pd", "dc_ph", "dc_pred",
    "xgb_pa", "xgb_pd", "xgb_ph", "xgb_pred", "xgb_adj",
    "bla_ph", "high_conf",
    "actual_result", "dc_correct", "xgb_correct", "adj_correct",
    "manual_adj_applied", "is_current",
    "dc_top_scores",
]


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def _result_label(hs: int, as_: int) -> str:
    return "H" if hs > as_ else ("A" if hs < as_ else "D")


def _safe(s: object, n: int = 22) -> str:
    return str(s)[:n]


def load_frozen_models():
    with open(OUT / "dc_model_2026.pkl",  "rb") as f: dc  = pickle.load(f)
    with open(OUT / "xgb_model_2026.pkl", "rb") as f: xgb = pickle.load(f)
    return dc, xgb


def load_live_results() -> pd.DataFrame:
    path = PROC / "wc2026_results.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame(columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "result", "neutral", "tournament", "k_factor", "group", "round",
    ])


def save_live_results(df: pd.DataFrame):
    df.to_csv(PROC / "wc2026_results.csv", index=False)


def load_live_predictions() -> pd.DataFrame:
    path = OUT / "live_predictions_2026.csv"
    if path.exists():
        df = pd.read_csv(path)
        for c in PRED_COLS:
            if c not in df.columns:
                df[c] = ""
        return df[PRED_COLS]
    return pd.DataFrame(columns=PRED_COLS)


def save_live_predictions(df: pd.DataFrame):
    """整体保存预测日志（允许更新结算列，但不删行）"""
    df[PRED_COLS].to_csv(OUT / "live_predictions_2026.csv", index=False)


def load_group_schedule() -> pd.DataFrame:
    """从 results.csv 读取 2026 WC 小组赛场次（含未赛行）"""
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    return (df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
            .sort_values("date").reset_index(drop=True))


# ══════════════════════════════════════════════════════════════
#  1. 录入比分
# ══════════════════════════════════════════════════════════════

def get_pending_matches(schedule: pd.DataFrame,
                        live_results: pd.DataFrame,
                        before_date: pd.Timestamp) -> pd.DataFrame:
    """返回已过日期但尚未录入结果的场次"""
    played_keys = set(zip(
        live_results["home_team"].astype(str),
        live_results["away_team"].astype(str),
    ))
    mask = (schedule["date"] < before_date) & schedule.apply(
        lambda r: (r["home_team"], r["away_team"]) not in played_keys, axis=1
    )
    return schedule[mask].sort_values("date")


def _try_webfetch_wiki() -> dict:
    """
    尝试从维基百科 2026 WC 页面获取比分。
    轻量正则解析，失败时返回 {}，由调用方回退手动录入。
    """
    import re
    WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
    try:
        import urllib.request
        req = urllib.request.Request(
            WIKI_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WCBot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        print(f"  [WebFetch] 获取到 {len(html)//1024} KB")
    except Exception as e:
        print(f"  [WebFetch] 连接失败: {e}")
        return {}

    # 轻量解析: 维基百科 WC 页面的分组表格中含 "X – Y" 比分模式
    # 由于 HTML 结构复杂，这里只做最简版提取，可靠性不保证
    # 维基百科 WC 页面结构复杂，需专用解析器；轻量正则不可靠
    # 返回空，由手动录入接管
    _ = re  # 标记已导入
    return {}


def ask_results_manually(pending: pd.DataFrame) -> list[dict]:
    """命令行交互录入比分"""
    new_rows = []
    print("\n" + "=" * 62)
    print("  手动录入比分  (格式: 主队进球-客队进球，跳过直接回车)")
    print("=" * 62)
    for _, r in pending.iterrows():
        ht = r["home_team"]; at = r["away_team"]
        dt = str(r["date"].date())
        prompt = f"  {dt}  {_safe(ht, 22):<22} vs {_safe(at, 22):<22} : "
        try:
            score_str = input(prompt).strip().lstrip("﻿")
        except EOFError:
            print("  [跳过 EOF]")
            continue
        if not score_str:
            continue
        try:
            hs, as_ = map(int, score_str.split("-"))
        except ValueError:
            print(f"  [跳过] 无法解析: {score_str!r}")
            continue
        grp = TEAM_TO_GROUP.get(ht) or TEAM_TO_GROUP.get(at) or "?"
        rnd = get_round(pd.Timestamp(dt))
        new_rows.append({
            "date":       dt,
            "home_team":  ht,
            "away_team":  at,
            "home_score": hs,
            "away_score": as_,
            "result":     _result_label(hs, as_),
            "neutral":    bool(r.get("neutral", True)),
            "tournament": "FIFA World Cup",
            "k_factor":   WC_KFACTOR,
            "group":      grp,
            "round":      rnd,
        })
    return new_rows


# ══════════════════════════════════════════════════════════════
#  2. 生成预测
# ══════════════════════════════════════════════════════════════

def generate_day_predictions(
    target_date:  pd.Timestamp,
    dc,
    xgb_pkg:     dict,
    live_results: pd.DataFrame,
    schedule:    pd.DataFrame,
    manual_adj:  dict | None = None,
) -> pd.DataFrame:
    """为 target_date 的场次生成预测，返回预测 DataFrame"""
    upcoming = schedule[schedule["date"] == target_date].copy()
    if len(upcoming) == 0:
        print(f"  [预测] {target_date.date()} 无待预测场次")
        return pd.DataFrame(columns=PRED_COLS)

    # 历史 + 本届已完赛
    hist_clean  = pd.read_csv(PROC / "matches_clean.csv", parse_dates=["date"])
    all_matches = pd.concat([
        hist_clean[hist_clean["home_score"].notna()],
        live_results[live_results["home_score"].notna()],
    ]).sort_values("date").reset_index(drop=True)

    # Elo (含本届结果)
    with open(OUT / "elo_best_params.json") as f:
        elo_params = json.load(f)
    elo_series = build_elo_series(hist_clean, live_results, elo_params)
    elo_dict   = get_current_elo(elo_series)

    # 叠加人工修正 (不持久化)
    if manual_adj:
        for team, delta in manual_adj.items():
            if team in elo_dict:
                elo_dict[team] = elo_dict[team] + delta

    # 特征矩阵
    X_df = build_feature_matrix(
        upcoming, WC2026_GROUPS, all_matches, elo_dict,
        live_results, TEAM_TO_GROUP, manual_adj or {},
    )
    X = X_df[FEAT_COLS_INTER].values.astype("float32")

    # 预测
    dc_probs      = dc_probs_for_matches(dc, upcoming)
    dc_top_scores = dc_top_scores_for_matches(dc, upcoming)
    xgb_probs = xgb_pkg["calibrated_model"].predict_proba(X)
    xgb_delta = xgb_pkg["delta"]
    xgb_thr   = xgb_pkg["draw_thr"]

    # BLa (Elo 基线，仅参考)
    hist_train = hist_clean[hist_clean["date"] < WC_OPENING]
    draw_rate  = float((hist_train["result"] == "D").mean())
    bla_phs = []
    for i in range(len(X_df)):
        elo_h = float(X_df.iloc[i]["elo_home_pre"])
        elo_a = float(X_df.iloc[i]["elo_away_pre"])
        neu   = bool(upcoming.iloc[i]["neutral"])
        h_eff = elo_h + H_ADV * (0 if neu else 1)
        we    = 1.0 / (1.0 + 10.0 ** (-(h_eff - elo_a) / 400.0))
        bla_phs.append(we * (1 - draw_rate))

    lo     = LABEL_ORDER
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    adj_note_global = ",".join(f"{k}:{v:+d}" for k, v in (manual_adj or {}).items())

    rows = []
    for i, (_, row) in enumerate(upcoming.iterrows()):
        dc_p  = dc_probs[i]
        xp    = xgb_probs[i]
        dc_pred  = lo[np.argmax(dc_p)]
        xgb_pred = lo[np.argmax(xp)]
        xgb_adj  = predict_with_draw_adj(
            xp.reshape(1, -1), xgb_delta, xgb_thr, lo
        )[0]
        affected = manual_adj and (
            row["home_team"] in manual_adj or row["away_team"] in manual_adj
        )
        rows.append({
            "timestamp":          now_ts,
            "pred_date":          str(datetime.now().date()),
            "match_date":         str(row["date"].date()),
            "home_team":          row["home_team"],
            "away_team":          row["away_team"],
            "dc_pa":              round(float(dc_p[0]), 4),
            "dc_pd":              round(float(dc_p[1]), 4),
            "dc_ph":              round(float(dc_p[2]), 4),
            "dc_pred":            dc_pred,
            "xgb_pa":             round(float(xp[0]), 4),
            "xgb_pd":             round(float(xp[1]), 4),
            "xgb_ph":             round(float(xp[2]), 4),
            "xgb_pred":           xgb_pred,
            "xgb_adj":            xgb_adj,
            "bla_ph":             round(bla_phs[i], 4),
            "high_conf":          int(float(np.max(dc_p)) > 0.6),
            "actual_result":      "",
            "dc_correct":         "",
            "xgb_correct":        "",
            "adj_correct":        "",
            "manual_adj_applied": adj_note_global if affected else "",
            "is_current":         1,
            "dc_top_scores":      dc_top_scores[i],
        })
    return pd.DataFrame(rows, columns=PRED_COLS)


# ══════════════════════════════════════════════════════════════
#  3. 预测去重写入（upsert）
# ══════════════════════════════════════════════════════════════

_PROB_COLS = ["dc_ph", "dc_pd", "dc_pa"]

def _is_current_mask(df: pd.DataFrame) -> pd.Series:
    return df["is_current"].astype(str).isin(["1", "1.0"])


def _upsert_predictions(df_preds: pd.DataFrame, df_new: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    对 df_new 中每场比赛按 (match_date, home_team, away_team) 做 upsert：
      - 不存在 is_current=1 的记录 → 直接追加（is_current=1）
      - 已存在且概率未变（Elo 未更新）→ 跳过
      - 已存在但概率已变（Elo 因新结果更新）→ 旧行置 is_current=0，追加新行 is_current=1

    Returns: (updated_df, added_count, updated_count)
    """
    df = df_preds.copy()
    new_rows = []
    added = updated = 0

    for _, nr in df_new.iterrows():
        k0 = str(nr["match_date"])
        k1 = str(nr["home_team"])
        k2 = str(nr["away_team"])

        cur_mask = (
            (df["match_date"].astype(str) == k0) &
            (df["home_team"].astype(str)  == k1) &
            (df["away_team"].astype(str)  == k2) &
            _is_current_mask(df)
        )
        existing = df[cur_mask]

        if len(existing) == 0:
            row_d = nr.to_dict()
            row_d["is_current"] = 1
            new_rows.append(row_d)
            added += 1
        else:
            ex = existing.iloc[0]
            try:
                same = all(
                    abs(float(ex[c]) - float(nr[c])) < 0.0005
                    for c in _PROB_COLS
                )
            except (ValueError, TypeError):
                same = False

            if not same:
                df.loc[cur_mask, "is_current"] = 0
                row_d = nr.to_dict()
                row_d["is_current"] = 1
                new_rows.append(row_d)
                updated += 1
            # else: 概率相同 → 跳过，不重复写入

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df, added, updated


# ══════════════════════════════════════════════════════════════
#  4. 结算预测
# ══════════════════════════════════════════════════════════════

def check_results_consistency(df_preds: pd.DataFrame, live_results: pd.DataFrame):
    """一致性自检：results.csv 有比分但 live_predictions actual_result 为空的场次"""
    if live_results.empty or df_preds.empty:
        return
    cur = df_preds[_is_current_mask(df_preds)]
    unsettled_keys = set(zip(
        cur[~cur["actual_result"].astype(str).isin(["H","D","A"])]["match_date"].astype(str),
        cur[~cur["actual_result"].astype(str).isin(["H","D","A"])]["home_team"].astype(str),
        cur[~cur["actual_result"].astype(str).isin(["H","D","A"])]["away_team"].astype(str),
    ))
    gaps = []
    for _, r in live_results.iterrows():
        if str(r.get("result","")) not in ("H","D","A"):
            continue
        date_s = str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"])[:10]
        key = (date_s, str(r["home_team"]), str(r["away_team"]))
        if key in unsettled_keys:
            gaps.append(f"    {date_s}  {r['home_team']} vs {r['away_team']}  result={r['result']}")
    if gaps:
        print(f"\n  ⚠ [一致性警告] 以下 {len(gaps)} 场在 results.csv 有比分，但 live_predictions 未结算：")
        for g in gaps:
            print(g)
        print("  → 重新运行 daily_update.py 会自动修复（启动时全量同步）")
    else:
        print("\n  [一致性检查] ✓ results.csv 与 live_predictions 完全同步，无漏结算场次")


def settle_predictions(df_preds: pd.DataFrame,
                       new_results: pd.DataFrame) -> pd.DataFrame:
    """把新比赛结果填回预测表的结算列（不增行，只填已有空单元格）"""
    if df_preds.empty:
        return df_preds
    df = df_preds.copy()
    for _, r in new_results.iterrows():
        true_res = str(r["result"])
        if true_res not in ("H", "D", "A"):
            continue
        mask = (
            (df["home_team"] == r["home_team"]) &
            (df["away_team"] == r["away_team"]) &
            (df["match_date"].astype(str) == str(r["date"].date())) &
            (df["actual_result"].astype(str).isin(["", "nan"]))
        )
        if not mask.any():
            continue
        idx = df[mask].index
        df.loc[idx, "actual_result"] = true_res
        df.loc[idx, "dc_correct"]    = df.loc[idx, "dc_pred"].apply(
            lambda p: int(str(p) == true_res))
        df.loc[idx, "xgb_correct"]   = df.loc[idx, "xgb_pred"].apply(
            lambda p: int(str(p) == true_res))
        df.loc[idx, "adj_correct"]    = df.loc[idx, "xgb_adj"].apply(
            lambda p: int(str(p) == true_res))
    return df


# ══════════════════════════════════════════════════════════════
#  4. 滚动统计
# ══════════════════════════════════════════════════════════════

MIN_N_FOR_SIGNIFICANCE = 40   # 样本量低于此值时，不展示有误导性的Δ判定


def print_rolling_stats(df_preds: pd.DataFrame):
    """
    全部统计量均从当前 df_preds 实时计算（is_current=1 且已结算的去重场次），
    不缓存、不沿用历史打印值——每次调用都是基于当下数据的重新统计。
    """
    cur_mask = (
        _is_current_mask(df_preds)
        if "is_current" in df_preds.columns
        else pd.Series([True] * len(df_preds), index=df_preds.index)
    )
    settled = df_preds[
        cur_mask &
        df_preds["actual_result"].astype(str).isin(["H", "D", "A"])
    ].copy()
    n = len(settled)
    if n == 0:
        print("\n  [滚动统计] 尚无已结算预测")
        return

    y  = settled["actual_result"].values
    dc_probs_arr = settled[["dc_pa", "dc_pd", "dc_ph"]].values.astype(float)
    dc_acc   = float((settled["dc_correct"].astype(int) == 1).mean())
    dc_brier = multiclass_brier(y, dc_probs_arr)
    xgb_acc  = float((settled["xgb_correct"].astype(int) == 1).mean())
    adj_acc  = float((settled["adj_correct"].astype(int) == 1).mean())

    da = dc_acc   - BACKTEST_ACC_DC
    db = dc_brier - BACKTEST_BRIER_DC

    print(f"\n  ┌─── 滚动统计 (n={n} 场已结算, is_current=1 去重实时计算) ──────┐")

    if n < MIN_N_FOR_SIGNIFICANCE:
        # 二项分布正态近似下的95%CI半宽度（百分点）
        ci_pp = 1.96 * ((dc_acc * (1 - dc_acc) / n) ** 0.5) * 100
        print(f"  │  DC  : ACC={dc_acc:.4f}  Brier={dc_brier:.4f}")
        print(f"  │  ⚠ n={n} < {MIN_N_FOR_SIGNIFICANCE}：Δ_ACC={da:+.4f} 相对回测基准的95%CI"
              f"半宽度≈±{ci_pp:.0f}pp，差值无统计显著性，仅供参考，不作偏高/偏低判定")
    else:
        status = "正常✓" if abs(da) < 0.05 else ("偏高↑" if da > 0 else "偏低↓")
        print(f"  │  DC  : ACC={dc_acc:.4f}  Brier={dc_brier:.4f}  "
              f"Δ_ACC={da:+.4f}  Δ_Brier={db:+.4f}  {status}")
    print(f"  │  XGB : ACC={xgb_acc:.4f}  adj_ACC={adj_acc:.4f}")
    print(f"  │  回测基准: ACC≈{BACKTEST_ACC_DC:.4f}  Brier≈{BACKTEST_BRIER_DC:.4f}")
    print(f"  └──────────────────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════
#  5. 打印预测表
# ══════════════════════════════════════════════════════════════

def print_prediction_table(df_pred: pd.DataFrame, title: str):
    if df_pred.empty:
        return
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    hdr = (f"  {'主队':<20} {'客队':<20} "
           f"{'DC H/D/A':>11} {'DC':>3} "
           f"{'XGB H/D/A':>11} {'X':>3} {'adj':>3} "
           f"{'BLa(H)':>7} {'':>2}")
    print(hdr)
    print("  " + "─" * 78)
    for _, r in df_pred.iterrows():
        star = "★" if r["high_conf"] else "  "
        res  = ("→" + str(r["actual_result"])
                if str(r.get("actual_result", "")) in ("H", "D", "A") else "")
        print(
            f"  {_safe(r['home_team']):<20} {_safe(r['away_team']):<20}"
            f" {float(r['dc_ph']):.2f}/{float(r['dc_pd']):.2f}/{float(r['dc_pa']):.2f}"
            f" {str(r['dc_pred']):>3}"
            f"  {float(r['xgb_ph']):.2f}/{float(r['xgb_pd']):.2f}/{float(r['xgb_pa']):.2f}"
            f" {str(r['xgb_pred']):>3} {str(r['xgb_adj']):>3}"
            f"  {float(r['bla_ph']):.2f}"
            f"  {star}{res}"
        )
    print(f"\n  注: H/D/A=主胜概率/平局/客胜  DC主模型  XGB对照  adj=draw调整  ★置信>0.6")


# ══════════════════════════════════════════════════════════════
#  6. 积分榜
# ══════════════════════════════════════════════════════════════

def print_standings(live_results: pd.DataFrame):
    if live_results.empty:
        return
    from live_features import group_standings_48
    print(f"\n  ── 当前小组积分榜 ──────────────────────────────────────────")
    has_any = False
    for grp_lbl in sorted(WC2026_GROUPS.keys()):
        teams   = WC2026_GROUPS[grp_lbl]
        matches = live_results[live_results["group"] == grp_lbl].copy()
        if len(matches) == 0:
            continue
        has_any = True
        st = group_standings_48(teams, matches)
        # st = {team_name: {pts, gd, gf, rank, mp}}
        rows = sorted(
            st.items(),
            key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"], -kv[1]["gf"])
        )
        line_parts = []
        for rk, (team_name, data) in enumerate(rows, 1):
            mp = data.get("mp", 0)
            line_parts.append(
                f"{rk}.{team_name[:12]}({data['pts']}分/{mp}场)"
            )
        print(f"  组{grp_lbl}: " + "  ".join(line_parts))
    if not has_any:
        print("  (尚无已完赛场次)")


# ══════════════════════════════════════════════════════════════
#  7. 人工修正询问
# ══════════════════════════════════════════════════════════════

def ask_manual_adjustments() -> dict:
    print("\n  [人工修正] 是否有伤病/名单变动需要调整 Elo？")
    print("  格式: 队名:±数值,队名:±数值   (回车跳过)")
    try:
        raw = input("  > ").strip()
    except EOFError:
        return {}
    if not raw:
        return {}
    adj: dict = {}
    for part in raw.split(","):
        try:
            team, val = part.strip().rsplit(":", 1)
            adj[team.strip()] = int(val.strip())
        except (ValueError, IndexError):
            pass
    if adj:
        print(f"  → 本次修正: {adj}")
    return adj


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="2026 WC 每日滚动预测")
    parser.add_argument("--date", default=None,
                        help="运行日期 YYYY-MM-DD (默认今天)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="跳过 WebFetch 直接手动录入")
    parser.add_argument("--predict-only", action="store_true",
                        help="只生成预测，不录入比分")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    today = (pd.Timestamp(args.date)
             if args.date else pd.Timestamp.today().normalize())

    print(f"\n{'='*65}")
    print(f"  daily_update.py   运行日期: {today.date()}")
    print(f"{'='*65}")

    dc, xgb_pkg  = load_frozen_models()
    live_results = load_live_results()
    schedule     = load_group_schedule()
    df_preds     = load_live_predictions()

    print(f"\n  已录入结果: {len(live_results)} 场  |  预测日志: {len(df_preds)} 条")

    # ── 启动全量同步：将 results.csv 中已有比分回写到 predictions ──
    # 捕获"直接改 results 但未经 daily_update 结算"的场次，防止漏出统计
    if not live_results.empty and not df_preds.empty:
        _pre = int(df_preds["actual_result"].astype(str).isin(["H","D","A"]).sum())
        df_preds = settle_predictions(df_preds, live_results)
        _post = int(df_preds["actual_result"].astype(str).isin(["H","D","A"]).sum())
        if _post > _pre:
            print(f"  [自动同步] 补回结算 {_post - _pre} 场"
                  f"（results 有比分但 predictions 未回写）")

    # ── 步骤 1: 录入昨日及更早未录入的比分 ───────────────────────
    if not args.predict_only:
        pending = get_pending_matches(schedule, live_results, today)

        if len(pending) > 0:
            print(f"\n  待录入场次 ({len(pending)} 场):")
            for _, r in pending.iterrows():
                print(f"    {str(r['date'].date())}  "
                      f"{_safe(r['home_team'], 22)} vs {_safe(r['away_team'], 22)}")

            # 尝试 WebFetch
            new_rows = []
            if not args.skip_fetch:
                print("\n  [WebFetch] 尝试从维基百科抓取比分...")
                fetched = _try_webfetch_wiki()
                if fetched:
                    print(f"  [WebFetch] 自动获取 {len(fetched)} 场")
                    # TODO: 转换为 new_rows 格式
                else:
                    print("  [WebFetch] 无法自动解析，切换手动录入")

            if not new_rows:
                new_rows = ask_results_manually(pending)

            if new_rows:
                df_new = pd.DataFrame(new_rows)
                df_new["date"] = pd.to_datetime(df_new["date"])
                live_results = pd.concat(
                    [live_results, df_new], ignore_index=True
                ).sort_values("date").reset_index(drop=True)
                save_live_results(live_results)
                print(f"\n  [保存] 录入 {len(new_rows)} 场 → "
                      f"wc2026_results.csv 共 {len(live_results)} 行")

                # 结算已有预测的结算列
                if not df_preds.empty:
                    df_preds = settle_predictions(df_preds, df_new)

        else:
            print(f"\n  [OK] 截至 {today.date()} 之前的所有场次均已录入")

    # ── 步骤 2: 打印滚动统计 ─────────────────────────────────────
    print_rolling_stats(df_preds)

    # ── 步骤 3: 询问人工修正 ─────────────────────────────────────
    manual_adj: dict = {}
    try:
        manual_adj = ask_manual_adjustments()
    except EOFError:
        pass

    # ── 步骤 4: 生成今日预测并打印 ──────────────────────────────
    print(f"\n  [预测] 生成 {today.date()} 场次预测...")
    df_today = generate_day_predictions(
        today, dc, xgb_pkg, live_results, schedule, manual_adj
    )
    if not df_today.empty:
        print_prediction_table(df_today, f"预测 — {today.date()}")
        df_preds, n_add, n_upd = _upsert_predictions(df_preds, df_today)
        if n_add == 0 and n_upd == 0:
            print(f"  [跳过] {today.date()} 预测已存在且特征未变，不重复写入")
        elif n_upd > 0:
            print(f"  [更新] {n_upd} 场预测因Elo变化已更新（旧行置is_current=0）")

    # ── 步骤 5: 明日预览 ─────────────────────────────────────────
    tomorrow = today + pd.Timedelta(days=1)
    print(f"\n  [预测] 生成 {tomorrow.date()} 预览...")
    df_tmr = generate_day_predictions(
        tomorrow, dc, xgb_pkg, live_results, schedule, manual_adj
    )
    if not df_tmr.empty:
        print_prediction_table(
            df_tmr, f"预览 — {tomorrow.date()} (明天运行时正式追加)")

    # ── 步骤 6: 保存整个预测日志 ─────────────────────────────────
    save_live_predictions(df_preds)
    n_current = int(_is_current_mask(df_preds).sum()) if "is_current" in df_preds.columns else len(df_preds)
    print(f"\n  [保存] 日志共 {len(df_preds)} 条（is_current=1: {n_current} 条）")

    # ── 步骤 7: 打印最近结果摘要 ─────────────────────────────────
    if not live_results.empty:
        recent = live_results.sort_values("date").tail(8)
        print(f"\n  最近录入结果:")
        for _, r in recent.iterrows():
            icon = {"H": "主胜", "D": "平", "A": "客胜"}.get(str(r["result"]), "?")
            print(f"    {str(r['date'].date())}  "
                  f"{_safe(r['home_team'], 20)}{int(r['home_score'])}-"
                  f"{int(r['away_score'])} {_safe(r['away_team'], 20)}  ({icon})")

    # ── 步骤 8: 积分榜 ───────────────────────────────────────────
    try:
        print_standings(live_results)
    except Exception as e:
        print(f"\n  [积分榜] 计算失败: {e}")

    # ── 步骤 8b: 一致性自检 ──────────────────────────────────────
    check_results_consistency(df_preds, live_results)

    # ── 步骤 9: 刷新静态看板 ─────────────────────────────────────
    try:
        import importlib.util, pathlib
        _spec = importlib.util.spec_from_file_location(
            "build_dashboard",
            pathlib.Path(__file__).parent / "build_dashboard.py"
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
    except Exception as e:
        print(f"\n  [看板] 生成失败: {e}")

    print(f"\n  完毕。明天运行: python src/daily_update.py")
    print(f"  使用约定: 先录今日比分，再看明日预测，顺序不可反。")


if __name__ == "__main__":
    main()
