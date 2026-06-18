"""
live_features.py — 为 2026 WC 实时比赛计算特征
模型本体已冻结；每次调用只更新动态特征（Elo、积分、休息天数）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from step6_elo_opt import compute_elo_series
from wc_data import CONFEDERATION, WC_HOSTS
from team_names import norm_team   # 唯一权威映射，与 step1 共用
from best_thirds import (          # 最佳第三名完整逻辑
    best_thirds_ranking,
    qual_status_with_thirds,
    collect_current_thirds,
)

LABEL_ORDER = ["A", "D", "H"]
WC2026_HOSTS = {"United States", "Canada", "Mexico"}
WC2026_YEAR  = 2026


# ── Round 日期范围（方便判断 R1/R2/R3）────────────────────────
# 按 results.csv 中的日期模式确定
WC2026_ROUND_DATES = {
    "R1": (pd.Timestamp("2026-06-11"), pd.Timestamp("2026-06-17")),
    "R2": (pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-23")),
    "R3": (pd.Timestamp("2026-06-24"), pd.Timestamp("2026-06-27")),
}


def get_round(date: pd.Timestamp) -> str:
    d = pd.Timestamp(date)
    for rnd, (lo, hi) in WC2026_ROUND_DATES.items():
        if lo <= d <= hi:
            return rnd
    return "KO"


def get_stage_flags(rnd: str) -> dict:
    """返回 stage one-hot 特征"""
    return {
        "is_knockout":    int(rnd == "KO"),
        "is_group_r3":    int(rnd == "R3"),
        "stage_group_r1": int(rnd == "R1"),
        "stage_group_r2": int(rnd == "R2"),
        "stage_group_r3": int(rnd == "R3"),
        "stage_knockout": int(rnd == "KO"),
        "stage_other":    0,
    }


# ══════════════════════════════════════════════
#  1. Elo 演化（全量重算）
# ══════════════════════════════════════════════

def build_elo_series(hist_df: pd.DataFrame,
                     live_df:  pd.DataFrame,
                     elo_params: dict) -> pd.DataFrame:
    """
    将历史数据 + 本届已完赛结果合并，重算 Elo 序列。
    返回含 elo_home_pre/away_pre/home_post/away_post 的 DataFrame（时间升序）。
    hist_df: matches_clean.csv 过滤后的历史数据
    live_df: wc2026_results.csv（本届已完赛）
    """
    needed = ["date", "home_team", "away_team", "home_score", "away_score",
              "result", "neutral", "k_factor"]
    hist = hist_df[needed].dropna(subset=["home_score", "away_score"]).copy()
    if len(live_df) > 0:
        live = live_df[needed].dropna(subset=["home_score", "away_score"]).copy()
        combined = pd.concat([hist, live]).sort_values("date").reset_index(drop=True)
    else:
        combined = hist.sort_values("date").reset_index(drop=True)
    return compute_elo_series(combined, elo_params)


def get_current_elo(elo_series: pd.DataFrame) -> dict:
    """返回每队最新 Elo 值 {team: elo}"""
    last_home = (elo_series[["home_team", "date", "elo_home_post"]]
                 .rename(columns={"home_team": "team", "elo_home_post": "elo"}))
    last_away = (elo_series[["away_team", "date", "elo_away_post"]]
                 .rename(columns={"away_team": "team", "elo_away_post": "elo"}))
    all_elo = pd.concat([last_home, last_away]).sort_values("date")
    return all_elo.groupby("team")["elo"].last().to_dict()


# ══════════════════════════════════════════════
#  2. 滚动统计
# ══════════════════════════════════════════════

def team_rolling(team: str, before_date: pd.Timestamp,
                 all_matches: pd.DataFrame, n: int) -> dict:
    """计算球队在 before_date 前的最近 n 场统计"""
    t = norm_team(team)  # 名称标准化，匹配训练集
    tm = all_matches[
        ((all_matches["home_team"] == t) | (all_matches["away_team"] == t)) &
        (all_matches["date"] < before_date) &
        all_matches["home_score"].notna()
    ].sort_values("date").tail(n)

    wins, gf_l, ga_l = [], [], []
    for _, r in tm.iterrows():
        if r["home_team"] == t:
            wins.append(1.0 if r["result"] == "H" else (0.5 if r["result"] == "D" else 0.0))
            gf_l.append(float(r["home_score"]))
            ga_l.append(float(r["away_score"]))
        else:  # team == away_team
            wins.append(1.0 if r["result"] == "A" else (0.5 if r["result"] == "D" else 0.0))
            gf_l.append(float(r["away_score"]))
            ga_l.append(float(r["home_score"]))

    return {
        "winrate": float(np.mean(wins)) if wins else 0.5,
        "gf":      float(np.mean(gf_l)) if gf_l else 1.0,
        "ga":      float(np.mean(ga_l)) if ga_l else 1.0,
        "last_date": tm["date"].max() if len(tm) > 0 else None,
    }


# ══════════════════════════════════════════════
#  3. H2H
# ══════════════════════════════════════════════

def h2h_stats(home: str, away: str, before_date: pd.Timestamp,
              all_matches: pd.DataFrame) -> tuple[float, int]:
    """
    主队视角的历史交锋胜率（胜1平0.5负0均值）。
    样本 < 3 时返回 (0.5, 0)。
    """
    h = norm_team(home); a = norm_team(away)
    mask = (
        (((all_matches["home_team"] == h) & (all_matches["away_team"] == a)) |
         ((all_matches["home_team"] == a) & (all_matches["away_team"] == h))) &
        (all_matches["date"] < before_date) &
        all_matches["home_score"].notna()
    )
    df = all_matches[mask]
    if len(df) < 3:
        return 0.5, len(df)
    scores = []
    for _, r in df.iterrows():
        if r["home_team"] == h:
            scores.append(1.0 if r["result"] == "H" else (0.5 if r["result"] == "D" else 0.0))
        else:
            scores.append(1.0 if r["result"] == "A" else (0.5 if r["result"] == "D" else 0.0))
    return float(np.mean(scores)), len(df)


# ══════════════════════════════════════════════
#  4. WC 经验
# ══════════════════════════════════════════════

def wc_experience(team: str, before_date: pd.Timestamp,
                  all_matches: pd.DataFrame) -> int:
    """历史 FIFA World Cup 正赛出场场次"""
    t = norm_team(team)
    mask = (
        ((all_matches["home_team"] == t) | (all_matches["away_team"] == t)) &
        (all_matches["tournament"] == "FIFA World Cup") &
        (all_matches["date"] < before_date)
    )
    return int(mask.sum())


# ══════════════════════════════════════════════
#  5. 小组积分榜
# ══════════════════════════════════════════════

def group_standings_48(group_teams: list[str],
                       live_results: pd.DataFrame,
                       group_label: str = "?") -> dict:
    """
    按 48 队赛制计算一个小组的积分榜。
    返回: {team: {pts, gd, gf, rank}}
    """
    pts = {t: 0 for t in group_teams}
    gd  = {t: 0 for t in group_teams}
    gf  = {t: 0 for t in group_teams}
    mp  = {t: 0 for t in group_teams}

    for _, r in live_results.iterrows():
        ht = r["home_team"]; at = r["away_team"]
        if ht not in pts or at not in pts:
            continue
        hs = int(r["home_score"]); as_ = int(r["away_score"])
        mp[ht] += 1; mp[at] += 1
        gf[ht] += hs; gf[at] += as_
        gd[ht] += hs - as_; gd[at] += as_ - hs
        if r["result"] == "H":
            pts[ht] += 3
        elif r["result"] == "D":
            pts[ht] += 1; pts[at] += 1
        else:
            pts[at] += 3

    sorted_teams = sorted(group_teams,
                          key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    rank = {t: i + 1 for i, t in enumerate(sorted_teams)}
    return {t: {"pts": pts[t], "gd": gd[t], "gf": gf[t],
                "rank": rank[t], "mp": mp[t]} for t in group_teams}


def compute_all_group_standings(wc_groups: dict,
                                live_wc: pd.DataFrame) -> dict[str, dict]:
    """
    一次性计算全部12组积分榜，供跨组第三名比较使用。
    返回 {grp: {team: {pts,gd,gf,rank,mp}}}
    """
    return {
        grp: group_standings_48(teams, live_wc, grp)
        for grp, teams in wc_groups.items()
    }


# ══════════════════════════════════════════════
#  6. 核心：为一场比赛生成完整特征
# ══════════════════════════════════════════════

def compute_match_features(
    match_date:  pd.Timestamp,
    home:        str,
    away:               str,
    neutral:            bool,
    group_label:        str,
    wc_groups:          dict,           # {"A": [...], "B": [...], ...}
    all_matches:        pd.DataFrame,   # 全部历史 + 本届已完赛（有比分）
    elo_dict:           dict,           # 当前 {team: elo}
    live_wc:            pd.DataFrame,   # 本届已完赛（用于组内积分）
    all_group_standings: dict | None = None,  # 全12组积分榜，用于最佳第三名跨组比较
    manual_adj:         dict | None = None,
) -> dict:
    """
    返回与 features.csv 同名的特征字典（65 列）+ 附加键 best_third_prob_home/away。
    附加键不在 FEAT_COLS_INTER 中，不传入冻结模型，供蒙特卡洛和监控使用。
    """
    adj    = manual_adj or {}
    # norm_team 将赛程表名（如"Bosnia and Herzegovina"）映射到训练集名（"Bosnia-Herzegovina"）
    # elo_dict 以训练集名为键，必须先 normalize 再查找，否则回退 1500
    home_n = norm_team(home)
    away_n = norm_team(away)
    ELO_H  = float(elo_dict.get(home_n, 1500)) + adj.get(home, 0)
    ELO_A  = float(elo_dict.get(away_n, 1500)) + adj.get(away, 0)
    H_ADV  = 125.0  # 冻结
    h_eff  = ELO_H + H_ADV * (0 if neutral else 1)
    elo_diff = h_eff - ELO_A

    rnd = get_round(match_date)
    stage_flags = get_stage_flags(rnd)

    # ── 近期状态 ────────────────────────────────
    h5  = team_rolling(home, match_date, all_matches, 5)
    h10 = team_rolling(home, match_date, all_matches, 10)
    a5  = team_rolling(away, match_date, all_matches, 5)
    a10 = team_rolling(away, match_date, all_matches, 10)

    # ── 休息天数 ────────────────────────────────
    def rest_days(team, before):
        t = norm_team(team)  # 同样需要 normalize
        tm = all_matches[
            ((all_matches["home_team"] == t) | (all_matches["away_team"] == t)) &
            (all_matches["date"] < before)
        ]
        if len(tm) == 0:
            return 30
        last = tm["date"].max()
        return int((before - last).days)

    h_rest = rest_days(home, match_date)
    a_rest = rest_days(away, match_date)

    # ── H2H ─────────────────────────────────────
    h2h_wr, h2h_n = h2h_stats(home, away, match_date, all_matches)

    # ── WC 经验 ─────────────────────────────────
    h_wc_exp = wc_experience(home, match_date, all_matches)
    a_wc_exp = wc_experience(away, match_date, all_matches)

    # ── 情境 ────────────────────────────────────
    is_host_home = int(home in WC2026_HOSTS)
    is_host_away = int(away in WC2026_HOSTS)
    confed_h = CONFEDERATION.get(home, "OTHER")
    confed_a = CONFEDERATION.get(away, "OTHER")

    # ── 小组积分 ─────────────────────────────────
    group_teams = wc_groups.get(group_label, [])
    group_pts_h = group_pts_a = 0
    group_gd_h  = group_gd_a  = 0
    group_rank_h = group_rank_a = 2
    qual_h = qual_a = 1
    must_h = must_a = 0
    rot_h  = rot_a  = 0
    btp_h  = btp_a  = 0.0   # best_third_prob

    if group_teams and rnd in ("R1", "R2", "R3"):
        stnd = group_standings_48(group_teams, live_wc, group_label)
        remaining = max(0, 3 - max(v["mp"] for v in stnd.values()))

        # 如未传入全组积分榜则即时计算（保证向后兼容）
        ags = all_group_standings or compute_all_group_standings(wc_groups, live_wc)

        for tm, col in [(home, "h"), (away, "a")]:
            if tm in stnd:
                s = stnd[tm]
                qs, btp = qual_status_with_thirds(
                    tm, stnd, remaining, ags, group_label
                )
                if col == "h":
                    group_pts_h  = s["pts"]
                    group_gd_h   = s["gd"]
                    group_rank_h = s["rank"]
                    qual_h = qs;  btp_h = btp
                    must_h = int(qs == 0 or (rnd == "R3" and qs == 1))
                    rot_h  = int(qs == 2 and rnd == "R3")
                else:
                    group_pts_a  = s["pts"]
                    group_gd_a   = s["gd"]
                    group_rank_a = s["rank"]
                    qual_a = qs;  btp_a = btp
                    must_a = int(qs == 0 or (rnd == "R3" and qs == 1))
                    rot_a  = int(qs == 2 and rnd == "R3")
    elif rnd == "KO":
        must_h = must_a = 1

    # ── confederation one-hot ────────────────────
    confeds = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "OTHER", "UEFA"]
    confed_h_ohe = {f"confed_home_{c}": int(confed_h == c) for c in confeds}
    confed_a_ohe = {f"confed_away_{c}": int(confed_a == c) for c in confeds}

    # ── 组装 ────────────────────────────────────
    feat = {
        "home_last5_winrate":      h5["winrate"],
        "home_last10_winrate":     h10["winrate"],
        "home_last5_goals_for":    h5["gf"],
        "home_last5_goals_against":h5["ga"],
        "away_last5_winrate":      a5["winrate"],
        "away_last10_winrate":     a10["winrate"],
        "away_last5_goals_for":    a5["gf"],
        "away_last5_goals_against":a5["ga"],
        "home_days_since_last":    h_rest,
        "away_days_since_last":    a_rest,
        "rest_diff":               h_rest - a_rest,
        "home_prev_et":            0,
        "away_prev_et":            0,
        "home_wc_exp":             h_wc_exp,
        "away_wc_exp":             a_wc_exp,
        "is_home_advantage":       int(not neutral),
        "is_host_home":            is_host_home,
        "is_host_away":            is_host_away,
        "same_confed":             int(confed_h == confed_a),
        "is_world_cup":            1,
        "is_qualifier":            0,
        "is_friendly":             0,
        "h2h_winrate":             h2h_wr,
        "h2h_n":                   h2h_n,
        "group_pts_home":          group_pts_h,
        "group_pts_away":          group_pts_a,
        "group_gd_home":           group_gd_h,
        "group_gd_away":           group_gd_a,
        "group_rank_home":         group_rank_h,
        "group_rank_away":         group_rank_a,
        "qual_status_home":        qual_h,
        "qual_status_away":        qual_a,
        "must_win_home":           must_h,
        "must_win_away":           must_a,
        "motivation_diff":         must_h - must_a,
        "rotation_home":           rot_h,
        "rotation_away":           rot_a,
        **stage_flags,
        **confed_h_ohe,
        **confed_a_ohe,
        "elo_home_pre":            ELO_H,
        "elo_away_pre":            ELO_A,
        "elo_diff":                elo_diff,
        # 交互特征
        "elo_diff_ko":             elo_diff * stage_flags["is_knockout"],
        "rest_diff_ko":            (h_rest - a_rest) * stage_flags["is_knockout"],
        # ── 附加键：不在 FEAT_COLS_INTER，不传入冻结模型 ────────
        # 供蒙特卡洛模拟、监控显示和未来重训时使用
        "best_third_prob_home":    btp_h,
        "best_third_prob_away":    btp_a,
    }
    return feat


# ══════════════════════════════════════════════
#  7. 批量生成特征 DataFrame
# ══════════════════════════════════════════════

FEAT_COLS_BASE = [
    "home_last5_winrate", "home_last10_winrate", "home_last5_goals_for",
    "home_last5_goals_against", "away_last5_winrate", "away_last10_winrate",
    "away_last5_goals_for", "away_last5_goals_against",
    "home_days_since_last", "away_days_since_last", "rest_diff",
    "home_prev_et", "away_prev_et", "home_wc_exp", "away_wc_exp",
    "is_home_advantage", "is_host_home", "is_host_away", "same_confed",
    "is_world_cup", "is_qualifier", "is_friendly",
    "h2h_winrate", "h2h_n",
    "group_pts_home", "group_pts_away", "group_gd_home", "group_gd_away",
    "group_rank_home", "group_rank_away",
    "qual_status_home", "qual_status_away",
    "must_win_home", "must_win_away", "motivation_diff",
    "rotation_home", "rotation_away",
    "is_knockout", "is_group_r3",
    "stage_group_r1", "stage_group_r2", "stage_group_r3",
    "stage_knockout", "stage_other",
    "confed_home_AFC", "confed_home_CAF", "confed_home_CONCACAF",
    "confed_home_CONMEBOL", "confed_home_OFC", "confed_home_OTHER",
    "confed_home_UEFA",
    "confed_away_AFC", "confed_away_CAF", "confed_away_CONCACAF",
    "confed_away_CONMEBOL", "confed_away_OFC", "confed_away_OTHER",
    "confed_away_UEFA",
    "elo_home_pre", "elo_away_pre", "elo_diff",
]
FEAT_COLS_INTER = FEAT_COLS_BASE + ["elo_diff_ko", "rest_diff_ko"]


def build_feature_matrix(matches_to_predict: pd.DataFrame,
                         wc_groups: dict,
                         all_matches: pd.DataFrame,
                         elo_dict: dict,
                         live_wc: pd.DataFrame,
                         team_to_group: dict,
                         manual_adj: dict | None = None) -> pd.DataFrame:
    """
    批量计算特征，返回 DataFrame（行对应 matches_to_predict）。
    全12组积分榜仅计算一次，传给每场比赛避免重复计算。
    """
    # 预计算全12组积分榜（跨组最佳第三名比较用）
    ags = compute_all_group_standings(wc_groups, live_wc)

    rows = []
    for _, row in matches_to_predict.iterrows():
        grp = team_to_group.get(row["home_team"],
              team_to_group.get(row["away_team"], "?"))
        feat = compute_match_features(
            match_date          = pd.Timestamp(row["date"]),
            home                = row["home_team"],
            away                = row["away_team"],
            neutral             = bool(row.get("neutral", True)),
            group_label         = grp,
            wc_groups           = wc_groups,
            all_matches         = all_matches,
            elo_dict            = elo_dict,
            live_wc             = live_wc,
            all_group_standings = ags,
            manual_adj          = manual_adj,
        )
        rows.append(feat)
    return pd.DataFrame(rows, columns=FEAT_COLS_INTER)
