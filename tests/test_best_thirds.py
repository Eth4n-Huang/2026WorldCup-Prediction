"""
best_thirds.py 单元测试 v2

新增场景:
  test_fourth_can_rise_to_third — 当前第4，末轮赢球可升第3并进前8
  → 断言 qual_status=1 (争夺中)，而非 0 (已淘汰)
  这是 v2 修正的核心 regression test。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from best_thirds import (
    best_thirds_ranking,
    collect_current_thirds,
    estimate_best_third_prob,
    qual_status_with_thirds,
    _sim_group_to_final,
    N_SLOTS,
)
import random


# ══════════════════════════════════════════════════════
#  辅助
# ══════════════════════════════════════════════════════

def _make_thirds(specs: list[tuple]) -> list[dict]:
    return [{"team": t, "pts": p, "gd": g, "gf": f} for t, p, g, f in specs]


def _done_stnd(team_data: dict) -> dict:
    """构造"已打完"的积分榜（mp=3）并附 rank。"""
    sorted_t = sorted(team_data, key=lambda t: (
        -team_data[t]["pts"],
        -team_data[t].get("gd", 0),
        -team_data[t].get("gf", 0),
    ))
    return {
        t: {**team_data[t], "mp": 3, "rank": i + 1}
        for i, t in enumerate(sorted_t)
    }


# ══════════════════════════════════════════════════════
#  1. best_thirds_ranking — 排序正确性
# ══════════════════════════════════════════════════════

def test_ranking_by_points():
    thirds = _make_thirds([
        ("A", 7, 3, 6), ("B", 5, 1, 4), ("C", 4, 0, 3),
        ("D", 4,-1, 2), ("E", 3, 0, 2), ("F", 3,-2, 1),
        ("G", 2,-1, 2), ("H", 2,-3, 1), ("I", 1,-2, 1),
        ("J", 1,-4, 0), ("K", 0,-3, 1), ("L", 0,-5, 0),
    ])
    ranked = best_thirds_ranking(thirds)
    assert ranked[0]["team"] == "A"
    assert all(r["qualifies"] for r in ranked[:8])
    assert all(not r["qualifies"] for r in ranked[8:])
    print("PASS: test_ranking_by_points")


def test_tiebreak_gd():
    ranked = best_thirds_ranking(_make_thirds([("X", 4, 2, 5), ("Y", 4,-1, 4)]))
    assert ranked[0]["team"] == "X"
    print("PASS: test_tiebreak_gd")


def test_tiebreak_gf():
    ranked = best_thirds_ranking(_make_thirds([("P", 5, 2, 7), ("Q", 5, 2, 4)]))
    assert ranked[0]["team"] == "P"
    print("PASS: test_tiebreak_gf")


def test_boundary_8th_vs_9th():
    thirds = _make_thirds([
        ("T1",7,5,9),("T2",6,3,7),("T3",6,2,6),("T4",5,1,5),
        ("T5",5,1,4),("T6",5,0,3),("T7",4,2,5),("T8",4,0,3),
        ("T9",3,1,4),("T10",3,0,3),("T11",2,-1,2),("T12",1,-3,1),
    ])
    ranked = best_thirds_ranking(thirds)
    assert ranked[7]["qualifies"] is True,  f"第8名应晋级: {ranked[7]}"
    assert ranked[8]["qualifies"] is False, f"第9名不应晋级: {ranked[8]}"
    print("PASS: test_boundary_8th_vs_9th")


# ══════════════════════════════════════════════════════
#  2. _sim_group_to_final — 模拟原语
# ══════════════════════════════════════════════════════

def test_sim_group_no_remaining():
    """remaining=0 时不模拟，直接根据当前积分排名。"""
    stnd = {
        "A": {"pts": 9, "gd": 5, "gf": 9, "mp": 3},
        "B": {"pts": 4, "gd": 0, "gf": 4, "mp": 3},
        "C": {"pts": 3, "gd":-2, "gf": 3, "mp": 3},
        "D": {"pts": 0, "gd":-3, "gf": 1, "mp": 3},
    }
    rng = random.Random(0)
    final = _sim_group_to_final(stnd, 0, rng)
    assert final["A"]["rank"] == 1
    assert final["D"]["rank"] == 4
    print("PASS: test_sim_group_no_remaining")


def test_sim_group_rank_stability():
    """多次模拟：第3名身份可能不同（验证非固定）。"""
    stnd = {t: {"pts": 3, "gd": 0, "gf": 3, "mp": 1} for t in "ABCD"}
    rng = random.Random(0)
    thirds_seen = set()
    for _ in range(200):
        final = _sim_group_to_final(stnd, 2, rng)
        thirds_seen.add(next(t for t, v in final.items() if v["rank"] == 3))
    assert len(thirds_seen) > 1, f"第3名应因模拟而变化，实际只见到: {thirds_seen}"
    print(f"PASS: test_sim_group_rank_stability (第3名出现过: {sorted(thirds_seen)})")


# ══════════════════════════════════════════════════════
#  3. collect_current_thirds
# ══════════════════════════════════════════════════════

def test_collect_current_thirds_basic():
    stnd = _done_stnd({
        "Brazil":  {"pts": 9, "gd": 6, "gf": 9},
        "France":  {"pts": 6, "gd": 2, "gf": 6},
        "Germany": {"pts": 3, "gd":-1, "gf": 3},
        "Poland":  {"pts": 0, "gd":-7, "gf": 1},
    })
    thirds = collect_current_thirds({"A": stnd})
    assert thirds["A"]["team"] == "Germany"
    assert thirds["A"]["remaining"] == 0
    print("PASS: test_collect_current_thirds_basic")


def test_collect_current_thirds_remaining():
    stnd = {t: {"pts": 3, "gd": 0, "gf": 2, "mp": 1}
            for i, t in enumerate(["Spain","Italy","Croatia","Albania"])}
    # 手动设 rank
    for i, t in enumerate(["Spain","Italy","Croatia","Albania"]):
        stnd[t]["rank"] = i + 1
    thirds = collect_current_thirds({"B": stnd})
    assert thirds["B"]["remaining"] == 2
    print("PASS: test_collect_current_thirds_remaining")


# ══════════════════════════════════════════════════════
#  4. estimate_best_third_prob 极端情形
# ══════════════════════════════════════════════════════

def _build_all_group_standings(own_group: str, own_stnd: dict,
                               other_pts: int = 1) -> dict:
    """构造12组积分榜：own_group 用 own_stnd，其余11组第3名均为 other_pts。"""
    ags = {own_group: own_stnd}
    for g in "ABCDEFGHIJKL":
        if g == own_group:
            continue
        ags[g] = _done_stnd({
            f"{g}1": {"pts": 9, "gd": 6, "gf": 9},
            f"{g}2": {"pts": 6, "gd": 2, "gf": 6},
            f"{g}3": {"pts": other_pts, "gd": other_pts - 4, "gf": other_pts},
            f"{g}4": {"pts": 0, "gd":-7, "gf": 1},
        })
    return ags


def test_prob_nearly_certain():
    """本组第3已打完、7pts +5gd，其他11组第3名均1pt -3gd → 概率≈1。"""
    own = _done_stnd({
        "T1": {"pts": 9, "gd": 5, "gf": 8},
        "T2": {"pts": 6, "gd": 2, "gf": 5},
        "T3": {"pts": 7, "gd": 5, "gf": 8},   # 当前实际第3（7 < 9 和 6 的问题？）
        "T4": {"pts": 0, "gd":-8, "gf": 1},
    })
    # 手动修正：用明确的强第3
    own = _done_stnd({
        "T1": {"pts": 9, "gd": 5, "gf": 8},
        "T2": {"pts": 8, "gd": 3, "gf": 7},
        "T3": {"pts": 7, "gd": 4, "gf": 6},   # 第3名, 7pts +4gd
        "T4": {"pts": 0, "gd":-8, "gf": 1},
    })
    other = {g: _done_stnd({
        f"{g}1": {"pts": 9, "gd": 5, "gf": 8},
        f"{g}2": {"pts": 6, "gd": 2, "gf": 5},
        f"{g}3": {"pts": 1, "gd":-4, "gf": 1},
        f"{g}4": {"pts": 0, "gd":-7, "gf": 0},
    }) for g in "BCDEFGHIJKL"}

    p = estimate_best_third_prob("T3", own, 0, other, n_sim=400, seed=0)
    assert p > 0.90, f"强势第三名概率应接近1，实际={p:.3f}"
    print(f"PASS: test_prob_nearly_certain  (p={p:.3f})")


def test_prob_nearly_zero():
    """本组第3 0pts -6gd，其他11组第3名均7pts +4gd → 概率≈0。"""
    own = _done_stnd({
        "T1": {"pts": 9, "gd": 5, "gf": 8},
        "T2": {"pts": 7, "gd": 3, "gf": 6},
        "T3": {"pts": 0, "gd":-6, "gf": 1},
        "T4": {"pts": 0, "gd":-7, "gf": 0},
    })
    other = {g: _done_stnd({
        f"{g}1": {"pts": 9, "gd": 5, "gf": 8},
        f"{g}2": {"pts": 6, "gd": 2, "gf": 5},
        f"{g}3": {"pts": 7, "gd": 4, "gf": 7},
        f"{g}4": {"pts": 0, "gd":-7, "gf": 0},
    }) for g in "BCDEFGHIJKL"}

    p = estimate_best_third_prob("T3", own, 0, other, n_sim=400, seed=0)
    assert p < 0.05, f"弱势第三名概率应接近0，实际={p:.3f}"
    print(f"PASS: test_prob_nearly_zero  (p={p:.3f})")


# ══════════════════════════════════════════════════════
#  5. 关键 regression test:
#     当前第4、末轮赢球可升第3并进前8 → qual_status=1 (争夺中)
# ══════════════════════════════════════════════════════

def test_fourth_can_rise_to_third_contesting():
    """
    v1 漏判场景:
      本组局面:
        T1: 9pts +5gd  (已出线)
        T2: 6pts +2gd  (已出线)
        T3: 4pts  0gd  (当前第3)
        T4: 3pts -2gd  (当前第4, 即目标队)  ← 评估对象
      还有1轮: T4 赢球 → 6pts > T3 的 4pts → T4 升第3
      其他11组第3名均1pt -3gd → T4 以6pts轻松入前8

    期望: qual_status=1 (争夺中), best_third_prob > 0
    v1 错误返回: (0, 0.0)  ← 因为 _sim_remaining 固定了 T3 作为第3名
    """
    own_stnd = {
        "T1": {"pts": 9, "gd": 5, "gf": 8, "mp": 2, "rank": 1},
        "T2": {"pts": 6, "gd": 2, "gf": 5, "mp": 2, "rank": 2},
        "T3": {"pts": 4, "gd": 0, "gf": 4, "mp": 2, "rank": 3},
        "T4": {"pts": 3, "gd":-2, "gf": 3, "mp": 2, "rank": 4},  # 目标队
    }
    ags = _build_all_group_standings("A", own_stnd, other_pts=1)

    qs, btp = qual_status_with_thirds(
        "T4", own_stnd, remaining_in_group=1,
        all_group_standings=ags, own_group="A",
        n_sim=800, seed=42,
    )

    assert qs == 1, (
        f"当前第4末轮可升第3时 qual_status 应为1(争夺中)，实际={qs}。"
        f"\n  v1 漏洞: 固定第3名身份导致错误返回0。"
    )
    assert btp > 0.0, f"best_third_prob 应>0，实际={btp:.3f}"
    print(f"PASS: test_fourth_can_rise_to_third_contesting"
          f"  qs={qs}, btp={btp:.3f}")


# ══════════════════════════════════════════════════════
#  6. qual_status_with_thirds 其他场景
# ══════════════════════════════════════════════════════

def test_confirmed_top2():
    stnd = _done_stnd({
        "T1": {"pts": 9, "gd": 5, "gf": 8},
        "T2": {"pts": 6, "gd": 2, "gf": 5},
        "T3": {"pts": 3, "gd":-1, "gf": 3},
        "T4": {"pts": 0, "gd":-6, "gf": 1},
    })
    qs, btp = qual_status_with_thirds("T1", stnd, 0, None, "X")
    assert qs == 2 and btp == 0.0
    qs2, _ = qual_status_with_thirds("T2", stnd, 0, None, "X")
    assert qs2 == 2
    print("PASS: test_confirmed_top2")


def test_rank4_cant_reach_third():
    """第4名且最高积分 < 当前第3 → 已淘汰。"""
    stnd = _done_stnd({
        "T1": {"pts": 9, "gd": 5, "gf": 8},
        "T2": {"pts": 6, "gd": 2, "gf": 5},
        "T3": {"pts": 6, "gd": 1, "gf": 5},
        "T4": {"pts": 0, "gd":-6, "gf": 1},
    })
    # T4: max_pts = 0 + 0*3 = 0 < pts_3rd = 6 → 淘汰
    qs, btp = qual_status_with_thirds("T4", stnd, 0, None, "X")
    assert qs == 0 and btp == 0.0, f"应淘汰: qs={qs}, btp={btp}"
    print("PASS: test_rank4_cant_reach_third")


def test_strong_third_confirmed_qualify():
    """
    组赛已结束，本组第3 7pts +4gd，其他11组第3名全是 1pt -3gd
    → qual_status=2 (已出线)
    """
    own = _done_stnd({
        "T1": {"pts": 9, "gd": 5, "gf": 8},
        "T2": {"pts": 8, "gd": 3, "gf": 7},
        "T3": {"pts": 7, "gd": 4, "gf": 6},
        "T4": {"pts": 0, "gd":-8, "gf": 1},
    })
    ags = _build_all_group_standings("A", own, other_pts=1)

    qs, btp = qual_status_with_thirds(
        "T3", own, 0, ags, "A", n_sim=400, seed=7
    )
    assert qs == 2 or btp > 0.85, f"强第三名应已出线: qs={qs}, btp={btp:.3f}"
    print(f"PASS: test_strong_third_confirmed_qualify  qs={qs}, btp={btp:.3f}")


def test_weak_third_eliminated():
    """
    组赛已结束，本组第3 0pts -6gd，其他11组第3名全是 7pts +4gd
    → qual_status=0 (已淘汰)
    """
    own = _done_stnd({
        "T1": {"pts": 9, "gd": 5, "gf": 8},
        "T2": {"pts": 7, "gd": 3, "gf": 6},
        "T3": {"pts": 0, "gd":-6, "gf": 1},
        "T4": {"pts": 0, "gd":-7, "gf": 0},
    })
    other = {g: _done_stnd({
        f"{g}1": {"pts": 9, "gd": 5, "gf": 8},
        f"{g}2": {"pts": 6, "gd": 2, "gf": 5},
        f"{g}3": {"pts": 7, "gd": 4, "gf": 7},
        f"{g}4": {"pts": 0, "gd":-7, "gf": 0},
    }) for g in "BCDEFGHIJKL"}

    ags = {"A": own, **other}
    qs, btp = qual_status_with_thirds(
        "T3", own, 0, ags, "A", n_sim=400, seed=7
    )
    assert qs == 0 or btp < 0.15, f"弱第三名应已淘汰: qs={qs}, btp={btp:.3f}"
    print(f"PASS: test_weak_third_eliminated  qs={qs}, btp={btp:.3f}")


# ══════════════════════════════════════════════════════
#  主运行
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    test_ranking_by_points()
    test_tiebreak_gd()
    test_tiebreak_gf()
    test_boundary_8th_vs_9th()
    test_sim_group_no_remaining()
    test_sim_group_rank_stability()
    test_collect_current_thirds_basic()
    test_collect_current_thirds_remaining()
    test_prob_nearly_certain()
    test_prob_nearly_zero()
    # ── 关键 regression test (v1 漏洞) ──
    test_fourth_can_rise_to_third_contesting()
    # ── 其他 qual_status 场景 ──
    test_confirmed_top2()
    test_rank4_cant_reach_third()
    test_strong_third_confirmed_qualify()
    test_weak_third_eliminated()
    print("\nAll best_thirds tests passed. (15/15)")
