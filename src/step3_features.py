"""
阶段2b-2e: 特征工程
输入: data/processed/matches_with_elo.csv + data/raw/shootouts.csv
输出: data/processed/features.csv

按时间顺序逐场计算赛前特征，严格无泄漏：
处理第i场时只使用前i-1场的信息。
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from wc_data import WC_GROUPS, WC_TEAM_GROUP, WC_HOSTS, WC_GROUP_STAGE_END, CONFEDERATION

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"


# ══════════════════════════════════════════════
#  小组赛出线模拟（阶段2d核心）
# ══════════════════════════════════════════════

def compute_standings(played: list[dict]) -> dict[str, dict]:
    """
    根据已赛结果精确计算积分表。
    played: list of {"home","away","home_score","away_score"}
    返回: {team: {"pts":int,"gd":int,"gf":int,"ga":int}}
    """
    standings: dict[str, dict] = defaultdict(lambda: {"pts": 0, "gd": 0, "gf": 0, "ga": 0})
    for g in played:
        h, a, hs, as_ = g["home"], g["away"], g["home_score"], g["away_score"]
        standings[h]["gf"] += hs; standings[h]["ga"] += as_
        standings[a]["gf"] += as_; standings[a]["ga"] += hs
        standings[h]["gd"] += hs - as_; standings[a]["gd"] += as_ - hs
        if hs > as_:
            standings[h]["pts"] += 3
        elif hs == as_:
            standings[h]["pts"] += 1; standings[a]["pts"] += 1
        else:
            standings[a]["pts"] += 3
    return {t: dict(v) for t, v in standings.items()}


def simulate_group(
    standings: dict[str, dict],
    remaining: list[tuple[str, str]],
    n_qualify: int = 2,
) -> dict[str, dict]:
    """
    枚举剩余比赛所有胜/平/负组合，确定每队的出线状态。
    remaining: [(home, away), ...] 尚未踢的对阵
    返回: {team: {"qual_status":0/1/2, "must_win":bool}}

    qual_status: 2=已出线, 1=争夺中, 0=已淘汰
    must_win: 本场不胜即出局（当前赛前视角）

    简化：模拟用 ±1 净胜球；平局消除细微GD差异不影响结论
    """
    teams = list(standings.keys())

    # 每个可能结果下的积分/gd增量
    OUTCOMES = {
        "H": (3, 0, 1, -1),   # home_pts, away_pts, home_gd, away_gd
        "D": (1, 1, 0, 0),
        "A": (0, 3, -1, 1),
    }

    can_qualify    = {t: False for t in teams}
    always_qualify = {t: True  for t in teams}

    for combo in product("HDA", repeat=len(remaining)):
        sim = {t: {"pts": v["pts"], "gd": v["gd"], "gf": v["gf"]} for t, v in standings.items()}
        for (home, away), res in zip(remaining, combo):
            hp, ap, hgd, agd = OUTCOMES[res]
            sim[home]["pts"] += hp; sim[away]["pts"] += ap
            sim[home]["gd"]  += hgd; sim[away]["gd"] += agd

        ranked = sorted(teams,
                        key=lambda t: (-sim[t]["pts"], -sim[t]["gd"], -sim[t]["gf"]))
        qualifiers = set(ranked[:n_qualify])
        for t in teams:
            if t in qualifiers:
                can_qualify[t] = True
            else:
                always_qualify[t] = False

    result = {}
    for t in teams:
        if always_qualify[t]:
            qs = 2
        elif can_qualify[t]:
            qs = 1
        else:
            qs = 0
        result[t] = {"qual_status": qs}

    # must_win：假设本队本场平局 → 还能否出线
    # 这里 remaining[0] 是"当前这场"，其余是后续场次
    if remaining:
        cur_home, cur_away = remaining[0]
        rest = remaining[1:]
        for team in [cur_home, cur_away]:
            # 固定本场为平局
            sim_draw = {t: {"pts": v["pts"], "gd": v["gd"], "gf": v["gf"]}
                        for t, v in standings.items()}
            sim_draw[cur_home]["pts"] += 1; sim_draw[cur_away]["pts"] += 1
            # 其余场次最优结果（对该队）
            can_after_draw = False
            for combo in product("HDA", repeat=len(rest)):
                sim2 = {t: dict(v) for t, v in sim_draw.items()}
                for (h2, a2), r2 in zip(rest, combo):
                    hp2, ap2, hgd2, agd2 = OUTCOMES[r2]
                    sim2[h2]["pts"] += hp2; sim2[a2]["pts"] += ap2
                    sim2[h2]["gd"]  += hgd2; sim2[a2]["gd"] += agd2
                ranked2 = sorted(teams,
                                 key=lambda t: (-sim2[t]["pts"], -sim2[t]["gd"], -sim2[t]["gf"]))
                if team in set(ranked2[:n_qualify]):
                    can_after_draw = True
                    break
            result[team]["must_win"] = not can_after_draw

    # 补全无法计算must_win的队（remaining为空或不在当前比赛中）
    for t in teams:
        result[t].setdefault("must_win", False)

    return result


# ══════════════════════════════════════════════
#  主特征构建
# ══════════════════════════════════════════════

def build_features(df: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    """
    对输入DataFrame（已含Elo列）逐行按时间顺序计算所有赛前特征。
    df 必须已按 date 升序排列。
    """
    # 点球大战记录 → 含ET的比赛集合 {(date_str, home, away)}
    et_matches: set[tuple] = set()
    for _, row in shootouts.iterrows():
        et_matches.add((str(row["date"])[:10], row["home_team"], row["away_team"]))

    # ── 状态容器 ────────────────────────────────
    # 每队近期赛果列表（滚动窗口，超过10场的旧数据不必删，取tail即可）
    team_hist: dict[str, list] = defaultdict(list)
    # H2H历史
    h2h_hist: dict[tuple, list] = defaultdict(list)
    # WC组内已赛场次计数 {(wc_year, team): int}
    wc_game_cnt: dict[tuple, int] = defaultdict(int)
    # WC组内已赛结果（用于计算积分表）{(wc_year, group): [played_dict, ...]}
    wc_played: dict[tuple, list] = defaultdict(list)
    # 累计WC正赛出场数 {team: int}
    wc_exp: dict[str, int] = defaultdict(int)

    rows_out = []

    for idx, row in df.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        date  = row["date"]
        year  = date.year
        tourn = str(row["tournament"]).lower()
        neutral = bool(row["neutral"])

        is_wc       = "fifa world cup" in tourn and "qualification" not in tourn
        is_qualifier= "qualification" in tourn or "qualifier" in tourn
        is_friendly = "friendly" in tourn

        # ── WC年份与小组赛判断 ────────────────────
        wc_year = year if is_wc else None
        # 确定具体WC届次（2026年的比赛也属于2026WC）
        if is_wc:
            # 从WC_GROUPS识别最近的WC届次（允许±2年容差，应对如2002赛事跨年等情况）
            candidates = [y for y in WC_GROUPS if abs(y - year) <= 2]
            wc_year = min(candidates, key=lambda y: abs(y - year)) if candidates else year

        in_wc_group = (
            is_wc
            and wc_year in WC_TEAM_GROUP
            and home in WC_TEAM_GROUP.get(wc_year, {})
            and away in WC_TEAM_GROUP.get(wc_year, {})
            and str(date.date()) <= WC_GROUP_STAGE_END.get(wc_year, "9999-12-31")
        )

        # 本场两队在当前WC的出场数（赛前）
        home_wc_cnt = wc_game_cnt[(wc_year, home)] if is_wc else 0
        away_wc_cnt = wc_game_cnt[(wc_year, away)] if is_wc else 0

        # 判断阶段
        if in_wc_group:
            round_num = min(home_wc_cnt, away_wc_cnt)  # 0→R1, 1→R2, 2→R3
            stage = f"group_r{round_num + 1}"
        elif is_wc:
            stage = "knockout"
        else:
            stage = "other"

        # ── 2b: 近期状态特征 ─────────────────────
        def team_rolling(team):
            h = team_hist[team]
            if not h:
                return {
                    "last5_winrate": 0.5, "last10_winrate": 0.5,
                    "last5_goals_for": 1.2, "last5_goals_against": 1.2,
                    "days_since_last": 30,
                    "prev_et": 0,
                }
            last5  = h[-5:]
            last10 = h[-10:]
            last   = h[-1]
            return {
                "last5_winrate":       np.mean([g["win"] for g in last5]),
                "last10_winrate":      np.mean([g["win"] for g in last10]),
                "last5_goals_for":     np.mean([g["gf"]  for g in last5]),
                "last5_goals_against": np.mean([g["ga"]  for g in last5]),
                "days_since_last":     (date - last["date"]).days,
                "prev_et":             int(last["was_et"]),
            }

        hs = team_rolling(home)
        as_ = team_rolling(away)

        rest_diff = hs["days_since_last"] - as_["days_since_last"]

        # ── 2b: WC经验 ───────────────────────────
        home_wc_exp = wc_exp[home]
        away_wc_exp = wc_exp[away]

        # ── 2c: 静态/情境特征 ────────────────────
        is_host_home = (
            is_wc
            and wc_year in WC_HOSTS
            and home in WC_HOSTS[wc_year]
        )
        is_host_away = (
            is_wc
            and wc_year in WC_HOSTS
            and away in WC_HOSTS[wc_year]
        )
        confed_home = CONFEDERATION.get(home, "OTHER")
        confed_away = CONFEDERATION.get(away, "OTHER")

        # H2H：主队在历史交锋中的得分率（胜1平0.5负0）
        h2h_key = (home, away)
        h2h_key_rev = (away, home)
        h2h_games = (h2h_hist[h2h_key] +
                     [{"result": "A" if g["result"] == "H" else ("H" if g["result"] == "A" else "D")}
                      for g in h2h_hist[h2h_key_rev]])
        h2h_n = len(h2h_games)
        if h2h_n >= 3:
            score_map = {"H": 1.0, "D": 0.5, "A": 0.0}
            h2h_winrate = np.mean([score_map[g["result"]] for g in h2h_games])
        else:
            h2h_winrate = 0.5  # 样本不足，用中性值

        # ── 2d: 出线形势（仅WC小组赛）───────────────
        group_key = (wc_year, WC_TEAM_GROUP.get(wc_year, {}).get(home)) if in_wc_group else None

        if in_wc_group and group_key:
            # 当前积分表
            played_so_far = wc_played[group_key]
            group_teams = WC_GROUPS[wc_year][group_key[1]]

            # 用实际比赛结果计算精确积分
            cur_standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in group_teams}
            for pg in played_so_far:
                s = compute_standings([pg])
                for t, v in s.items():
                    for k in ("pts", "gd", "gf"):
                        cur_standings[t][k] += v[k]

            # 确定剩余比赛（含本场）
            all_pairs = [(group_teams[i], group_teams[j])
                         for i in range(len(group_teams))
                         for j in range(i+1, len(group_teams))]
            played_pairs = {(pg["home"], pg["away"]) for pg in played_so_far}
            played_pairs |= {(pg["away"], pg["home"]) for pg in played_so_far}
            # 本场排在第一位
            remaining_games = [(home, away)] + [
                (a, b) for (a, b) in all_pairs
                if (a, b) not in played_pairs and not (a == home and b == away)
                   and (a, b) not in {(away, home)}
            ]

            try:
                sim_result = simulate_group(cur_standings, remaining_games)
                qual_home  = sim_result.get(home, {}).get("qual_status", 1)
                qual_away  = sim_result.get(away, {}).get("qual_status", 1)
                must_home  = int(sim_result.get(home, {}).get("must_win", False))
                must_away  = int(sim_result.get(away, {}).get("must_win", False))
            except Exception:
                qual_home = qual_away = 1
                must_home = must_away = 0

            # 排名（当前积分表）
            ranked = sorted(group_teams,
                            key=lambda t: (-cur_standings[t]["pts"],
                                           -cur_standings[t]["gd"],
                                           -cur_standings[t]["gf"]))
            home_rank = ranked.index(home) + 1 if home in ranked else 4
            away_rank = ranked.index(away) + 1 if away in ranked else 4

            gp_home  = cur_standings[home]["pts"]
            gp_away  = cur_standings[away]["pts"]
            gd_home  = cur_standings[home]["gd"]
            gd_away  = cur_standings[away]["gd"]

            # rotation_risk: 已出线且是小组末轮
            rotation_home = int(stage == "group_r3" and qual_home == 2)
            rotation_away = int(stage == "group_r3" and qual_away == 2)
        else:
            # 非WC小组赛/淘汰赛：按CLAUDE.md 2e节设置中性/规定值
            qual_home = qual_away = 1          # 中性值
            # 淘汰赛双方must_win=1（CLAUDE.md 2e明确要求）
            if stage == "knockout":
                must_home = must_away = 1
            else:
                must_home = must_away = 0
            home_rank = away_rank = 2          # 中性值
            gp_home = gp_away = gd_home = gd_away = 0
            rotation_home = rotation_away = 0  # 淘汰赛无轮换风险

        # ── 特征汇总 ─────────────────────────────
        feat = {
            # Elo（来自step2）
            "elo_home_pre":  row["elo_home_pre"],
            "elo_away_pre":  row["elo_away_pre"],
            "elo_diff":      row["elo_diff"],
            # 2b: 近期状态
            "home_last5_winrate":       hs["last5_winrate"],
            "home_last10_winrate":      hs["last10_winrate"],
            "home_last5_goals_for":     hs["last5_goals_for"],
            "home_last5_goals_against": hs["last5_goals_against"],
            "away_last5_winrate":       as_["last5_winrate"],
            "away_last10_winrate":      as_["last10_winrate"],
            "away_last5_goals_for":     as_["last5_goals_for"],
            "away_last5_goals_against": as_["last5_goals_against"],
            "home_days_since_last":     hs["days_since_last"],
            "away_days_since_last":     as_["days_since_last"],
            "rest_diff":                rest_diff,
            "home_prev_et":             hs["prev_et"],
            "away_prev_et":             as_["prev_et"],
            "home_wc_exp":              home_wc_exp,
            "away_wc_exp":              away_wc_exp,
            # 2c: 情境
            "is_home_advantage": int(not neutral),
            "is_host_home":      int(is_host_home),
            "is_host_away":      int(is_host_away),
            "confed_home":       confed_home,
            "confed_away":       confed_away,
            "same_confed":       int(confed_home == confed_away),
            "is_world_cup":      int(is_wc),
            "is_qualifier":      int(is_qualifier),
            "is_friendly":       int(is_friendly),
            "h2h_winrate":       h2h_winrate,
            "h2h_n":             h2h_n,
            # 2d: 出线形势
            "group_pts_home":    gp_home,
            "group_pts_away":    gp_away,
            "group_gd_home":     gd_home,
            "group_gd_away":     gd_away,
            "group_rank_home":   home_rank,
            "group_rank_away":   away_rank,
            "qual_status_home":  qual_home,
            "qual_status_away":  qual_away,
            "must_win_home":     must_home,
            "must_win_away":     must_away,
            "motivation_diff":   must_home - must_away,
            "rotation_home":     rotation_home,
            "rotation_away":     rotation_away,
            # 2e: 阶段
            "stage":             stage,
            "is_knockout":       int(stage == "knockout"),
            "is_group_r3":       int(stage == "group_r3"),
        }
        rows_out.append(feat)

        # ── 更新状态（赛后）────────────────────────
        hs_score = row["home_score"]
        as_score = row["away_score"]
        res      = row["result"]

        was_et = (str(date.date()), home, away) in et_matches or \
                 (stage == "knockout" and res == "D")

        for team, gf, ga, is_home in [
            (home, hs_score, as_score, True),
            (away, as_score, hs_score, False),
        ]:
            win_val = (1.0 if (is_home and res == "H") or (not is_home and res == "A")
                       else 0.5 if res == "D" else 0.0)
            team_hist[team].append({
                "date": date, "win": win_val,
                "gf": gf, "ga": ga, "was_et": was_et,
            })

        h2h_hist[(home, away)].append({"date": date, "result": res})

        if is_wc and wc_year:
            wc_game_cnt[(wc_year, home)] += 1
            wc_game_cnt[(wc_year, away)] += 1
            if is_wc:
                wc_exp[home] += 1
                wc_exp[away] += 1

        if in_wc_group and group_key:
            wc_played[group_key].append({
                "home": home, "away": away,
                "home_score": int(hs_score), "away_score": int(as_score),
            })

    feat_df = pd.DataFrame(rows_out, index=df.index)
    # one-hot编码 stage
    stage_dummies = pd.get_dummies(feat_df["stage"], prefix="stage").astype(int)
    feat_df = pd.concat([feat_df.drop(columns=["stage"]), stage_dummies], axis=1)
    # one-hot编码 confederation
    for col in ["confed_home", "confed_away"]:
        dummies = pd.get_dummies(feat_df[col], prefix=col).astype(int)
        feat_df = pd.concat([feat_df.drop(columns=[col]), dummies], axis=1)

    return feat_df


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("读取数据...")
    df = pd.read_csv(PROCESSED_DIR / "matches_with_elo.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    shootouts = pd.read_csv(RAW_DIR / "shootouts.csv", parse_dates=["date"])

    print(f"计算特征（共 {len(df)} 行）...")
    feat_df = build_features(df, shootouts)

    # 合并原始列 + 特征列
    keep_cols = ["date", "home_team", "away_team", "home_score", "away_score",
                 "result", "tournament", "k_factor", "neutral"]
    out = pd.concat([df[keep_cols].reset_index(drop=True),
                     feat_df.reset_index(drop=True)], axis=1)

    out.to_csv(PROCESSED_DIR / "features.csv", index=False)
    print(f"已保存: data/processed/features.csv  ({len(out)} 行 × {len(out.columns)} 列)")

    # ── 验收检查 ────────────────────────────────
    print("\n=== 验收检查 ===")
    wc22 = out[(out["tournament"] == "FIFA World Cup") &
               (out["date"] >= "2022-11-20") &
               (out["date"] <= "2022-12-02")]
    print(f"2022WC小组赛行数: {len(wc22)}  (期望48)")
    print(f"特征总数: {len(out.columns)} 列")
    print(f"\n2022WC小组赛出线特征样例:")
    sample_cols = ["home_team","away_team","stage_group_r1","stage_group_r2",
                   "stage_group_r3","is_knockout","qual_status_home","must_win_home",
                   "group_pts_home","group_rank_home"]
    available = [c for c in sample_cols if c in out.columns]
    print(wc22[available].head(6).to_string(index=False))

    ko22 = out[(out["tournament"] == "FIFA World Cup") &
               (out["date"] > "2022-12-02")]
    print(f"\n2022WC淘汰赛行数: {len(ko22)}  (期望16)")
    print(f"is_knockout均值: {ko22['is_knockout'].mean():.1f}  (期望1.0)")

    # 近期状态特征非空验证
    print(f"\n近期特征缺失检查 (should be 0):")
    for c in ["elo_diff","home_last5_winrate","rest_diff","home_wc_exp"]:
        print(f"  {c} NaN数: {out[c].isna().sum()}")
