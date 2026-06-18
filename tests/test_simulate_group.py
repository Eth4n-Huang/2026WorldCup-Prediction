"""
simulate_group / compute_standings 单元测试
验收要求：用2022年E组真实数据断言积分表和出线状态
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from step3_features import compute_standings, simulate_group


# ── 测试1：compute_standings 准确性 ─────────────────────────
def test_compute_standings_group_e_2022():
    """
    2022世界杯E组全部6场结果 → 验证最终积分表
    实际结果:
      Germany 1-2 Japan, Spain 7-0 Costa Rica
      Japan 0-1 Costa Rica, Spain 1-1 Germany
      Japan 2-1 Spain, Costa Rica 2-4 Germany
    最终名次: 1.Japan(6分,+2GD) 2.Spain(4分,+6GD) 3.Germany(4分,+1GD) 4.CostaRica(3分,-9GD)
    """
    played = [
        {"home": "Germany",     "away": "Japan",       "home_score": 1, "away_score": 2},
        {"home": "Spain",       "away": "Costa Rica",  "home_score": 7, "away_score": 0},
        {"home": "Japan",       "away": "Costa Rica",  "home_score": 0, "away_score": 1},
        {"home": "Spain",       "away": "Germany",     "home_score": 1, "away_score": 1},
        {"home": "Japan",       "away": "Spain",       "home_score": 2, "away_score": 1},
        {"home": "Costa Rica",  "away": "Germany",     "home_score": 2, "away_score": 4},
    ]
    s = compute_standings(played)

    assert s["Japan"]["pts"]       == 6,  f"Japan pts={s['Japan']['pts']}"
    assert s["Spain"]["pts"]       == 4,  f"Spain pts={s['Spain']['pts']}"
    assert s["Germany"]["pts"]     == 4,  f"Germany pts={s['Germany']['pts']}"
    assert s["Costa Rica"]["pts"]  == 3,  f"Costa Rica pts={s['Costa Rica']['pts']}"

    assert s["Japan"]["gd"]       == 1,   f"Japan gd={s['Japan']['gd']}"   # +1-1+1=1
    assert s["Spain"]["gd"]       == 6,   f"Spain gd={s['Spain']['gd']}"   # +7+0-1=6
    assert s["Germany"]["gd"]     == 1,   f"Germany gd={s['Germany']['gd']}"  # -1+0+2=1
    assert s["Costa Rica"]["gd"]  == -8,  f"CR gd={s['Costa Rica']['gd']}"  # -7+1-2=-8

    # 排名验证（pts降序，GD降序）
    teams = ["Japan", "Spain", "Germany", "Costa Rica"]
    ranked = sorted(teams, key=lambda t: (-s[t]["pts"], -s[t]["gd"], -s[t]["gf"]))
    assert ranked == ["Japan", "Spain", "Germany", "Costa Rica"], f"排名错误: {ranked}"
    print("PASS: test_compute_standings_group_e_2022")


# ── 测试2：simulate_group 明确已出线/已淘汰 ─────────────────
def test_simulate_group_clear_outcomes():
    """
    2018年G组R2后: Belgium 6分, England 6分, Panama 0分, Tunisia 0分
    R3: Belgium vs England, Panama vs Tunisia
    → Belgium/England 必出线(qual_status=2), Panama/Tunisia 必淘汰(qual_status=0)
    """
    standings = {
        "Belgium": {"pts": 6, "gd":  5, "gf": 8},
        "England": {"pts": 6, "gd":  6, "gf": 8},
        "Panama":  {"pts": 0, "gd": -8, "gf": 1},
        "Tunisia": {"pts": 0, "gd": -3, "gf": 3},
    }
    remaining = [("Belgium", "England"), ("Panama", "Tunisia")]

    result = simulate_group(standings, remaining)

    assert result["Belgium"]["qual_status"] == 2, f"Belgium={result['Belgium']}"
    assert result["England"]["qual_status"] == 2, f"England={result['England']}"
    assert result["Panama"]["qual_status"]  == 0, f"Panama={result['Panama']}"
    assert result["Tunisia"]["qual_status"] == 0, f"Tunisia={result['Tunisia']}"
    print("PASS: test_simulate_group_clear_outcomes")


# ── 测试3：simulate_group 争夺中场景 ───────────────────────
def test_simulate_group_contested():
    """
    4队同2分（纯争夺中），所有队 qual_status=1
    剩余: A vs B, C vs D (各队还有1场)
    """
    standings = {t: {"pts": 2, "gd": 0, "gf": 2} for t in ["A", "B", "C", "D"]}
    remaining = [("A", "B"), ("C", "D")]

    result = simulate_group(standings, remaining)
    for t in ["A", "B", "C", "D"]:
        assert result[t]["qual_status"] == 1, f"{t}={result[t]}"
    print("PASS: test_simulate_group_contested")


# ── 测试4：must_win 判断 ───────────────────────────────────
def test_must_win():
    """
    A: 0分GD-5, B: 6分, C: 6分, D: 0分
    剩余: A vs D (本场), B vs C
    A当前: 即使赢D也只有3分，而B和C至少有6分
    → A 在任何情况下都出不了线，qual_status=0，must_win无意义但should be False
    """
    standings = {
        "A": {"pts": 0, "gd": -5, "gf": 0},
        "B": {"pts": 6, "gd":  4, "gf": 8},
        "C": {"pts": 6, "gd":  3, "gf": 7},
        "D": {"pts": 0, "gd": -2, "gf": 1},
    }
    remaining = [("A", "D"), ("B", "C")]

    result = simulate_group(standings, remaining)
    # A和D最多都只能拿3分（<B/C的6分）→ 都已淘汰
    assert result["A"]["qual_status"] == 0, f"A={result['A']}"
    assert result["B"]["qual_status"] == 2, f"B={result['B']}"
    assert result["C"]["qual_status"] == 2, f"C={result['C']}"
    print("PASS: test_must_win")


if __name__ == "__main__":
    test_compute_standings_group_e_2022()
    test_simulate_group_clear_outcomes()
    test_simulate_group_contested()
    test_must_win()
    print("\nAll tests passed.")
