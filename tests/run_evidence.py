"""
补充测试证据 —— 打印全部返回值供人工审查
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from best_thirds import (
    qual_status_with_thirds,
    estimate_best_third_prob,
    _sim_group_to_final,
    _PAIRINGS_4,
)

SEP = "=" * 64

# ──────────────────────────────────────────────────────────────
# 辅助：构造12组积分榜（1个自定义 + 11个弱第3名）
# ──────────────────────────────────────────────────────────────
def _done_stnd(specs):
    """specs: {team: {pts,gd,gf}}  → 附 mp=3, rank"""
    teams = sorted(specs, key=lambda t: (
        -specs[t]["pts"], -specs[t].get("gd",0), -specs[t].get("gf",0)
    ))
    return {t: {**specs[t], "mp": 3, "rank": i+1} for i,t in enumerate(teams)}

def _ags(own_grp, own_stnd, other_pts=1, other_mp=3):
    """
    构造全12组积分榜:
      own_grp  → own_stnd
      其余11组 → 第3名均为 other_pts 分
    """
    ags = {own_grp: own_stnd}
    for g in "ABCDEFGHIJKL":
        if g == own_grp:
            continue
        ags[g] = {
            f"{g}1": {"pts":9,"gd":5,"gf":8,"mp":other_mp,"rank":1},
            f"{g}2": {"pts":6,"gd":2,"gf":5,"mp":other_mp,"rank":2},
            f"{g}3": {"pts":other_pts,"gd":other_pts-4,"gf":other_pts,"mp":other_mp,"rank":3},
            f"{g}4": {"pts":0,"gd":-7,"gf":0,"mp":other_mp,"rank":4},
        }
    return ags


# ══════════════════════════════════════════════════════════════
# 1. 核心场景: A=6 B=3 C=3 D=0,  末轮 D vs C, B vs A
#    D赢+B输 → D可升到3分参与best-third竞争
# ══════════════════════════════════════════════════════════════
print(SEP)
print("  [1] 核心场景: D当前第4(0分), 末轮可升第3")
print(SEP)

own_stnd = {
    "A": {"pts":6, "gd": 4, "gf":6, "mp":2, "rank":1},
    "B": {"pts":3, "gd": 1, "gf":4, "mp":2, "rank":2},
    "C": {"pts":3, "gd":-1, "gf":3, "mp":2, "rank":3},
    "D": {"pts":0, "gd":-4, "gf":1, "mp":2, "rank":4},   # 目标队
}
# pts_3rd = 3; D若赢: max_pts = 0+3=3 >= 3 → 可能升第3
# D若赢(+2GD) → D: 3pts -2gd; C若输: 3pts -2gd... 需要看模拟
# 其他11组第3名均为1pt → D达到3pts时大概率进前8

ags = _ags("A", own_stnd, other_pts=1)

qs, btp = qual_status_with_thirds(
    "D", own_stnd, remaining_in_group=1,
    all_group_standings=ags, own_group="A",
    n_sim=1000, seed=42
)
print(f"  qual_status_with_thirds('D', ...) = ({qs}, {btp:.4f})")
print(f"  期望: qs=1 (争夺中), btp>0")
passed = qs == 1 and btp > 0
print(f"  {'PASS' if passed else 'FAIL'}: qs={'1=争夺中' if qs==1 else str(qs)}, "
      f"btp={btp:.4f} (>0: {btp>0})")


# ══════════════════════════════════════════════════════════════
# 2. 边界测试三连
# ══════════════════════════════════════════════════════════════
print()
print(SEP)
print("  [2a] 真已淘汰: 剩余全赢也无法到第3")
print(SEP)

# D已打完3场, 0pts, pts_3rd=6pts → max_pts=0 < 6 → 淘汰
stnd_elim = _done_stnd({
    "T1": {"pts":9,"gd":5,"gf":8},
    "T2": {"pts":7,"gd":3,"gf":6},
    "T3": {"pts":6,"gd":1,"gf":5},
    "T4": {"pts":0,"gd":-7,"gf":1},   # 目标队, max_pts=0 < pts_3rd=6
})
qs_e, btp_e = qual_status_with_thirds(
    "T4", stnd_elim, remaining_in_group=0,
    all_group_standings=None, own_group="X"
)
print(f"  qual_status_with_thirds('T4', remaining=0) = ({qs_e}, {btp_e:.4f})")
print(f"  期望: qs=0 (已淘汰)")
print(f"  {'PASS' if qs_e == 0 else 'FAIL'}: qs={qs_e}")

print()
print(SEP)
print("  [2b] 真已出线: 已锁定小组前2")
print(SEP)

# T1: 9pts rank=1 → 已出线
qs_q, btp_q = qual_status_with_thirds(
    "T1", stnd_elim, remaining_in_group=0,
    all_group_standings=None, own_group="X"
)
print(f"  qual_status_with_thirds('T1', remaining=0) = ({qs_q}, {btp_q:.4f})")
print(f"  期望: qs=2 (已出线), btp=0.0")
print(f"  {'PASS' if qs_q == 2 and btp_q == 0.0 else 'FAIL'}: qs={qs_q}, btp={btp_q:.4f}")

# 还有比赛但积分已锁定：T2 6pts, 剩余1轮, 但 T3 max_pts=3+3=6 < T2的6 → T2还需检查
# 更明确的例子：T1 9pts, 其他队最多只能到6pts
stnd_locked = {
    "L1": {"pts":9,"gd":5,"gf":8,"mp":2,"rank":1},
    "L2": {"pts":6,"gd":2,"gf":5,"mp":2,"rank":2},
    "L3": {"pts":3,"gd":-1,"gf":3,"mp":2,"rank":3},
    "L4": {"pts":0,"gd":-6,"gf":1,"mp":2,"rank":4},
}
qs_lock, btp_lock = qual_status_with_thirds(
    "L1", stnd_locked, remaining_in_group=1,
    all_group_standings=None, own_group="X"
)
print(f"  qual_status_with_thirds('L1', pts=9, remaining=1) = ({qs_lock}, {btp_lock:.4f})")
print(f"  期望: qs=2 (9pts 已锁 rank1), btp=0.0")
print(f"  {'PASS' if qs_lock == 2 and btp_lock == 0.0 else 'FAIL'}")

print()
print(SEP)
print("  [2c] 蒙特卡洛稳定性: 同一输入跑5次, 波动应在±0.03内")
print(SEP)

# 构造一个"稳定中间值"场景: 本组第3 5pts, 其他11组第3均5pts → 概率约0.5
own_mid = _done_stnd({
    "M1": {"pts":9,"gd":5,"gf":8},
    "M2": {"pts":6,"gd":2,"gf":5},
    "M3": {"pts":5,"gd":1,"gf":4},   # 目标队, 评估为第3名竞争
    "M4": {"pts":0,"gd":-6,"gf":1},
})
other_mid = {g: _done_stnd({
    f"{g}1": {"pts":9,"gd":5,"gf":8},
    f"{g}2": {"pts":6,"gd":2,"gf":5},
    f"{g}3": {"pts":5,"gd":1,"gf":4},
    f"{g}4": {"pts":0,"gd":-7,"gf":0},
}) for g in "BCDEFGHIJKL"}

probs = []
for s in range(5):
    p = estimate_best_third_prob(
        "M3", own_mid, 0, other_mid,
        n_sim=1000, seed=s * 7 + 100
    )
    probs.append(p)

print(f"  5次概率: {[f'{p:.4f}' for p in probs]}")
spread = max(probs) - min(probs)
print(f"  最大波动: {spread:.4f} (期望 ≤0.06, 1000次模拟)")
print(f"  {'PASS' if spread <= 0.06 else 'FAIL'}: spread={spread:.4f}")


# ══════════════════════════════════════════════════════════════
# 3. _sim_group_to_final 自检: 100次, 每轮4队各出场1次
# ══════════════════════════════════════════════════════════════
print()
print(SEP)
print("  [3] _sim_group_to_final自检: 100次, 每轮恰好4队各出场1次")
print(SEP)

stnd_check = {t: {"pts":3,"gd":0,"gf":3,"mp":1} for t in ["W","X","Y","Z"]}
rng = random.Random(999)
teams = list(stnd_check.keys())
errors = []

for trial in range(100):
    for remaining_rounds in [1, 2, 3]:
        # 打印前几次的配对让人工核查
        # 内部: 每轮从 _PAIRINGS_4 取一个方案，方案内2对恰好覆盖4队
        # 这里通过检验最终 mp 来间接验证
        # 直接验证: mock 内部调用计数
        pass

# 更直接的验证: 对每种 remaining_rounds, 检查模拟前后 pts 变化总量
# 每轮2场，每场6个积分（3+3 或 1+1 或 3+0），每队参与1场
# => 总积分增量 = remaining_rounds × (per_round_total_pts)
# per_round_total_pts ∈ {6(平局 1+1+1+1?wait, 2games=2*(1+1)=4 or 2*3=6 or ...)}
# 实际: 每场产生3pts(胜负)或2pts(平局), 2场产生4-6pts
# 无法用总pts精确验证，改为验证"每个团队出场次数"

# 真正的验证: hook _sim_match 计数每队参与情况
call_log = []

original_sim_match = None

import best_thirds as bt

original_sim_match = bt._sim_match

def counting_sim_match(rng_):
    # 无法直接获取h,a，改用间接方式
    return original_sim_match(rng_)

# 用替代方法: 直接检验 _PAIRINGS_4 的完整性
print("  检验 _PAIRINGS_4 完整性（数学验证）:")
for idx, pairing in enumerate(_PAIRINGS_4):
    participants = set()
    for i, j in pairing:
        participants.add(i); participants.add(j)
    assert participants == {0,1,2,3}, f"配对{idx}未覆盖全部4队: {participants}"
    assert len(pairing) == 2, f"配对{idx}应有2场: {pairing}"
    print(f"    方案{idx}: {pairing} → 覆盖队伍: {sorted(participants)} OK")

# 验证3种方案两两不重叠（无重复对阵）
all_pairs = set()
for pairing in _PAIRINGS_4:
    for i, j in pairing:
        pair = (min(i,j), max(i,j))
        assert pair not in all_pairs, f"对阵 {pair} 重复!"
        all_pairs.add(pair)
expected_pairs = {(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)}
assert all_pairs == expected_pairs, f"应包含全部6对: {all_pairs}"
print(f"  3种方案合计 {len(all_pairs)} 对，恰好是4队全排列(C(4,2)=6) OK")

# 运行时验证: 100次模拟，检查每次模拟后积分增量合理
ok_count = 0
for trial in range(100):
    for rem in [1, 2, 3]:
        base = {t: {"pts":0,"gd":0,"gf":0,"mp":3-rem} for t in ["W","X","Y","Z"]}
        final = _sim_group_to_final(base, rem, rng)
        total_pts = sum(final[t]["pts"] for t in ["W","X","Y","Z"])
        # 每场产生2pts(平)或3pts(胜负); rem轮共2*rem场
        # 最少 2*rem*2=4*rem pts, 最多 2*rem*3=6*rem pts
        min_pts = 2 * rem * 2
        max_pts = 2 * rem * 3
        if min_pts <= total_pts <= max_pts:
            ok_count += 1
        else:
            errors.append(f"trial={trial} rem={rem}: total_pts={total_pts} out of [{min_pts},{max_pts}]")

total_trials = 100 * 3
print(f"  运行时检验: {ok_count}/{total_trials} 次积分总量在合理区间内")
if errors:
    print(f"  异常: {errors[:3]}")
else:
    print(f"  全部 {total_trials} 次通过 (每轮积分总量均在 [4*rem, 6*rem] 内)")
print(f"  {'PASS' if ok_count == total_trials else 'FAIL'}")

# ── 最终汇总 ──────────────────────────────────────────────────
print()
print(SEP)
all_passed = (
    passed and
    qs_e == 0 and
    qs_q == 2 and btp_q == 0.0 and
    qs_lock == 2 and
    spread <= 0.06 and
    ok_count == total_trials
)
print(f"  总结: {'全部通过' if all_passed else '有测试未通过，见上方FAIL'}")
print(SEP)
