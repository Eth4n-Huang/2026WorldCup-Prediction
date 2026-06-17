"""
FIFA 2026 最佳第三名（Best Third-Place Teams）逻辑 v2

v1 的关键漏洞:
  _sim_remaining() 把"当前第3名"当作固定输入，只扰动该队的得分增量。
  如果某队当前第4，它永远不会出现在 other_thirds 列表里，于是概率直接为0。

v2 修正原则:
  每次模拟先跑完本组和其他11组的全部剩余比赛，
  再从最终积分表中找各组第3名，做跨组排序。
  "谁是第3名"是模拟产物，不是输入前提。

对外接口:
  best_thirds_ranking(thirds)
      精确排名（所有组赛结束后 / 蒙特卡洛内部调用）
  estimate_best_third_prob(own_team, own_group_stnd, own_remaining,
                           other_group_standings, ...)
      蒙特卡洛：P(以第3名晋级前8 | 最终确实排第3)
  qual_status_with_thirds(team, stnd, remaining, all_group_standings, own_group, ...)
      替换旧 qual_status_team()，返回 (qual_status, best_third_prob)
  collect_current_thirds(all_group_standings)
      快照辅助函数，供监控/展示用
"""
from __future__ import annotations
import random

N_GROUPS       = 12
N_SLOTS        = 8        # 最佳第三名晋级名额
GAMES_PER_TEAM = 3        # 每队小组赛总场次

# 4队单循环的3种完整配对方案（每种方案恰好覆盖全部4队各1场）
_PAIRINGS_4 = [
    [(0, 1), (2, 3)],
    [(0, 2), (1, 3)],
    [(0, 3), (1, 2)],
]

# qual_status 阈值（组赛已结束时才触发确定性判断）
_HIGH_PROB = 0.85
_LOW_PROB  = 0.10


# ══════════════════════════════════════════════════════════════
#  1. 精确排名（蒙特卡洛内部 / 所有组赛结束后使用）
# ══════════════════════════════════════════════════════════════

def best_thirds_ranking(thirds: list[dict]) -> list[dict]:
    """
    按 FIFA 2026 规则对全部第3名精确排名。
    Input : [{"team": str, "pts": int, "gd": int, "gf": int}, ...]
    Output: 同结构列表，追加 "rank" (1-12) 和 "qualifies" (前8为True)。
    排序  : pts↓ → gd↓ → gf↓ → 稳定（无公平竞赛数据时停在 gf 级）
    """
    if not thirds:
        return []
    ranked = sorted(thirds, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    return [{**t, "rank": i + 1, "qualifies": i < N_SLOTS} for i, t in enumerate(ranked)]


# ══════════════════════════════════════════════════════════════
#  2. 比赛模拟原语
# ══════════════════════════════════════════════════════════════

def _sim_match(rng: random.Random) -> tuple[int, int, int, int]:
    """
    模拟一场比赛，返回 (h_goals, a_goals, h_pts, a_pts)。
    概率基于世界杯组赛历史：主胜≈38%, 平≈24%, 客胜≈38%。
    """
    r = rng.random()
    if r < 0.38:                    # 主场胜
        hg = rng.randint(1, 4)
        ag = max(0, hg - rng.randint(1, 3))
        return hg, ag, 3, 0
    elif r < 0.62:                  # 平局
        g = rng.randint(0, 2)
        return g, g, 1, 1
    else:                           # 客场胜
        ag = rng.randint(1, 4)
        hg = max(0, ag - rng.randint(1, 3))
        return hg, ag, 0, 3


def _sim_group_to_final(
    current_stnd:     dict,          # {team: {pts, gd, gf, mp, ...}}
    remaining_rounds: int,
    rng:              random.Random,
) -> dict:                           # {team: {pts, gd, gf, rank: 1-4}}
    """
    在当前积分基础上随机模拟剩余 remaining_rounds 轮，
    返回含最终排名(rank=1-4)的积分快照。

    4队小组: 每轮恰好2场，全队都出场（无轮空）。
    随机从3种配对方案中不重复取 remaining_rounds 个。
    非4队小组降级处理（只排名当前积分，不模拟）。
    """
    teams = list(current_stnd.keys())

    sim = {t: {"pts": current_stnd[t]["pts"],
               "gd":  current_stnd[t].get("gd", 0),
               "gf":  current_stnd[t].get("gf", 0)}
           for t in teams}

    if len(teams) == 4 and remaining_rounds > 0:
        # 随机打乱3种配对顺序，取前 remaining_rounds 个
        perm = [0, 1, 2]
        rng.shuffle(perm)
        for pairing_idx in perm[:remaining_rounds]:
            for i, j in _PAIRINGS_4[pairing_idx]:
                h, a = teams[i], teams[j]
                hg, ag, hp, ap = _sim_match(rng)
                sim[h]["gf"] += hg;  sim[a]["gf"] += ag
                sim[h]["gd"] += hg - ag;  sim[a]["gd"] += ag - hg
                sim[h]["pts"] += hp;  sim[a]["pts"] += ap

    ranked = sorted(teams,
                   key=lambda t: (sim[t]["pts"], sim[t]["gd"], sim[t]["gf"]),
                   reverse=True)
    for i, t in enumerate(ranked):
        sim[t]["rank"] = i + 1
    return sim


# ══════════════════════════════════════════════════════════════
#  3. 辅助：当前第3名快照（仅供监控/展示）
# ══════════════════════════════════════════════════════════════

def collect_current_thirds(all_group_standings: dict[str, dict]) -> dict[str, dict]:
    """
    从每组当前积分榜提取第3名快照（附 remaining 字段）。
    注意: 此函数只做展示/日志，不用于 estimate_best_third_prob。
    """
    thirds: dict[str, dict] = {}
    for grp, stnd in all_group_standings.items():
        if not stnd:
            continue
        sorted_teams = sorted(
            stnd.items(),
            key=lambda x: (x[1]["pts"], x[1].get("gd", 0), x[1].get("gf", 0)),
            reverse=True,
        )
        if len(sorted_teams) < 3:
            continue
        t3, t3s = sorted_teams[2]
        max_mp = max(v.get("mp", 0) for v in stnd.values())
        thirds[grp] = {
            "team":      t3,
            "pts":       t3s["pts"],
            "gd":        t3s.get("gd", 0),
            "gf":        t3s.get("gf", 0),
            "remaining": max(0, GAMES_PER_TEAM - max_mp),
        }
    return thirds


# ══════════════════════════════════════════════════════════════
#  4. 蒙特卡洛概率估计（v2 修正版）
# ══════════════════════════════════════════════════════════════

def estimate_best_third_prob(
    own_team:              str,
    own_group_stnd:        dict,   # 本组4队完整积分榜 {team:{pts,gd,gf,mp}}
    own_remaining_rounds:  int,    # 本组剩余轮数
    other_group_standings: dict,   # {grp: {team:{pts,gd,gf,mp}}} 其他11组
    n_slots:               int = N_SLOTS,
    n_sim:                 int = 2000,
    seed:                  int = 42,
) -> float:
    """
    P( own_team 通过最佳第三名路径晋级 | own_team 在本组最终排名第3 )

    每次模拟:
      ① 跑完本组剩余比赛 → 确定 own_team 最终名次
      ② 若 own_team ≠ 第3名 → 跳过（不记入分子/分母）
      ③ 跑完其他11组剩余比赛 → 找各组第3名
      ④ 跨组排序 → own_team 是否入前 n_slots

    返回值: qualify_count / third_count
      - third_count == 0 说明该队在任何模拟中都无法排第3 → 返回 0.0
    """
    rng = random.Random(seed)

    # 预计算其他组的剩余轮数
    other_items: list[tuple[dict, int]] = []
    for stnd in other_group_standings.values():
        max_mp = max((v.get("mp", 0) for v in stnd.values()), default=0)
        remaining = max(0, GAMES_PER_TEAM - max_mp)
        other_items.append((stnd, remaining))

    qualify_count = 0
    third_count   = 0

    for _ in range(n_sim):
        # ① 模拟本组 → own_team 最终名次
        own_final = _sim_group_to_final(own_group_stnd, own_remaining_rounds, rng)
        if own_final[own_team]["rank"] != 3:
            continue        # 非第3名情形不计入（第1/2=直接出线，第4=淘汰，均与本函数无关）
        third_count += 1

        own_key = (own_final[own_team]["pts"],
                   own_final[own_team]["gd"],
                   own_final[own_team]["gf"])

        # ② 模拟其他11组 → 找各组第3名 → 统计比本队强的个数
        n_better = 0
        for stnd, remaining in other_items:
            final = _sim_group_to_final(stnd, remaining, rng)
            # 找这组的第3名
            t3 = next((t for t, v in final.items() if v["rank"] == 3), None)
            if t3 is None:
                continue
            other_key = (final[t3]["pts"], final[t3]["gd"], final[t3]["gf"])
            if other_key > own_key:
                n_better += 1
            elif other_key == own_key:
                n_better += rng.randint(0, 1)   # 完全平局：随机决胜（代理公平竞赛/抽签）

        if n_better < n_slots:
            qualify_count += 1

    return qualify_count / third_count if third_count > 0 else 0.0


# ══════════════════════════════════════════════════════════════
#  5. 主接口：替换旧 qual_status_team()
# ══════════════════════════════════════════════════════════════

def qual_status_with_thirds(
    team:                str,
    stnd:                dict,          # 本组 {team:{pts,gd,gf,rank,mp}}
    remaining_in_group:  int,           # 本组剩余轮数
    all_group_standings: dict | None = None,
    own_group:           str = "?",
    n_slots:             int = N_SLOTS,
    n_sim:               int = 1000,
    seed:                int = 42,
) -> tuple[int, float]:
    """
    Returns (qual_status, best_third_prob).
      qual_status     : 0=已淘汰, 1=争夺中, 2=已出线
      best_third_prob : P(晋级 | 最终第3) ∈ [0,1]
                        对非第3竞争者（已锁前2或已淘汰）返回 0.0

    逻辑:
      1. 已锁前2 → (2, 0.0)
      2. 连第3都到不了（max_pts < pts_3rd）→ (0, 0.0)
      3. 可能第3:
         a. 无跨组数据 → (1, 0.5)
         b. 有跨组数据 → 调用 estimate_best_third_prob
            - 组赛已结束 (remaining==0) 且当前恰好第3:
              btp > _HIGH_PROB → (2, btp)
              btp < _LOW_PROB  → (0, btp)
              else             → (1, btp)
            - 组赛未结束或尚未确定名次:
              → 保守给 (1, btp)（名次未定，不提前宣判）
    """
    s   = stnd[team]
    pts = s["pts"]
    rnk = s["rank"]
    n   = len(stnd)
    max_pts = pts + remaining_in_group * 3

    sorted_pts = sorted([v["pts"] for v in stnd.values()], reverse=True)
    pts_2nd = sorted_pts[1] if n >= 2 else 0
    pts_3rd = sorted_pts[2] if n >= 3 else 0

    # ── 已确定直接出线 ───────────────────────────────────────
    if pts >= 6 and rnk <= 2:
        return (2, 0.0)
    # 积分领先到不可能被超越（超出其他任何队最高可得分）
    if n >= 2 and pts > pts_2nd + remaining_in_group * 3:
        return (2, 0.0)

    # ── 确定连第3都到不了 → 已淘汰 ──────────────────────────
    if max_pts < pts_3rd:
        return (0, 0.0)

    # ── 可能排第3（含当前第4但仍可逆转的情形）───────────────
    if all_group_standings is None or own_group not in all_group_standings:
        return (1, 0.5)

    own_group_stnd      = all_group_standings[own_group]
    other_group_stnd    = {g: s for g, s in all_group_standings.items()
                           if g != own_group}

    if not other_group_stnd:
        # 无其他组数据（应不会发生）
        return (1, 0.5)

    btp = estimate_best_third_prob(
        team,
        own_group_stnd,
        remaining_in_group,
        other_group_stnd,
        n_slots = n_slots,
        n_sim   = n_sim,
        seed    = seed,
    )

    # ── 映射至 qual_status ───────────────────────────────────
    # 只有当组赛已结束且当前排名确实是第3时，才可能宣判确定结论
    if remaining_in_group == 0 and rnk == 3:
        if btp >= _HIGH_PROB:
            return (2, btp)
        elif btp <= _LOW_PROB:
            return (0, btp)

    # 其他情形（还有比赛 / 还不确定名次）：保守给争夺中
    return (1, btp)
