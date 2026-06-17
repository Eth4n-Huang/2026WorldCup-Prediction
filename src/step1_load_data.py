"""
阶段1: 数据加载与清洗
输出: data/processed/matches_clean.csv
"""
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from team_names import TEAM_NAME_MAP


def _norm(s: str) -> str:
    """去除重音并转小写，用于赛事名匹配（解决 Copa América 等带重音字符的问题）"""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

# ── 路径配置 ──────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# ── 1. 读取原始数据 ──────────────────────────────────────
df = pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])
print(f"原始数据: {len(df)} 场比赛")

# 删除比分缺失行
df = df.dropna(subset=["home_score", "away_score"])
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)

# ── 2. 仅保留1990年后，按时间升序 ────────────────────────
df = df[df["date"] >= "1990-01-01"].sort_values("date").reset_index(drop=True)
print(f"1990年后: {len(df)} 场比赛")

# ── 3. 生成结果标签 H/D/A (基于90分钟比分) ────────────────
# 数据集说明: results.csv 记录的是90分钟比分
# 口径核查见下方第6步
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "H"
    elif row["home_score"] < row["away_score"]:
        return "A"
    else:
        return "D"

df["result"] = df.apply(get_result, axis=1)

# ── 4. K因子分配（根据赛事重要性） ───────────────────────
# K越大 → 该场比赛对Elo更新影响越大
# 注意：所有字符串匹配先经过 _norm() 去重音转小写，避免 Copa América/African 等拼写坑
MAJOR_TOURNAMENTS = [
    "copa america",        # 数据中为 "Copa América"，去重音后匹配
    "uefa euro",
    "afc asian cup",
    "african cup of nations",  # 数据中为 "African Cup of Nations"（含n）
    "gold cup",            # 数据中为 "Gold Cup"（非 CONCACAF Gold Cup）
    "concacaf championship",
    "confederations cup",
    "uefa nations league",
    "concacaf nations league",
]

def assign_k_factor(tournament):
    t = _norm(str(tournament))
    if "fifa world cup" in t and "qualification" not in t:
        return 60   # 世界杯正赛，权重最高
    if any(name in t for name in MAJOR_TOURNAMENTS) and "qualification" not in t:
        return 50   # 洲际决赛圈
    if "qualification" in t or "qualifier" in t:
        return 40   # 资格赛
    if "friendly" in t:
        return 20   # 友谊赛，权重最低
    return 30       # 其他赛事

df["k_factor"] = df["tournament"].apply(assign_k_factor)
print("\nK因子分布:")
print(df["k_factor"].value_counts().sort_index())

# ── 5. 球队名标准化 ───────────────────────────────────────
# TEAM_NAME_MAP 来自 team_names.py（唯一维护点），此处直接使用
df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

# 打印出现次数 < 5 次的球队，供人工判断是否需要追加映射
all_teams = pd.concat([df["home_team"], df["away_team"]])
team_counts = all_teams.value_counts()
rare = team_counts[team_counts < 5]
print(f"\n球队总数: {team_counts.nunique()}")
print(f"出现<5次的球队（共{len(rare)}个，供人工核查异名）:")
# 只打印 ASCII 可显示的部分，避免 Windows 终端乱码
safe_rare = [(name, cnt) for name, cnt in rare.items() if name.isascii()]
for name, cnt in safe_rare[:20]:
    print(f"  {name}: {cnt}")

# ── 6. 口径核查 ──────────────────────────────────────────
# 检查两场关键比赛，确认是90分钟比分还是含加时最终比分
print("\n=== 口径核查 ===")

# 2014决赛: 德国 vs 阿根廷
# 90分钟: 0-0，加时进1球: 最终1-0
final_2014 = df[
    (df["date"] == "2014-07-13") &
    (df["home_team"].isin(["Germany", "Argentina"])) &
    (df["away_team"].isin(["Germany", "Argentina"]))
]
print(f"2014决赛 德国vs阿根廷: {final_2014[['date','home_team','away_team','home_score','away_score','result']].to_string(index=False)}")

# 2022决赛: 阿根廷 vs 法国
# 90分钟: 2-2，加时: 3-3，点球: 阿根廷胜
final_2022 = df[
    (df["date"] == "2022-12-18") &
    (df["home_team"].isin(["Argentina", "France"])) &
    (df["away_team"].isin(["Argentina", "France"]))
]
print(f"2022决赛 阿根廷vs法国: {final_2022[['date','home_team','away_team','home_score','away_score','result']].to_string(index=False)}")

print("""
【口径结论（已核实）】
数据集记录的是"含加时最终比分"，不是纯90分钟比分：
  - 2014决赛: 1-0（德国加时得分），非0-0的90分钟比分
  - 2022决赛: 3-3（加时后比分），非2-2的90分钟比分
影响：
  - 小组赛无加时，H/D/A标签 = 90分钟结果，无歧义
  - 淘汰赛 D 标签 = 120分钟仍平局（随后点球），H/A 可能含加时进球
  - 论文须注明：模型预测的"平局"在淘汰赛语境下等价于"进入点球"
""")

# ── 7. 标记中立场地 ──────────────────────────────────────
# neutral字段: True=中立场，False=主客场
df["neutral"] = df["neutral"].astype(bool)

# ── 8. 输出清洗后的数据 ──────────────────────────────────
output_cols = [
    "date", "home_team", "away_team",
    "home_score", "away_score", "result",
    "tournament", "k_factor", "neutral"
]
df[output_cols].to_csv(PROCESSED_DIR / "matches_clean.csv", index=False)
print(f"\n已保存: data/processed/matches_clean.csv")

# ── 验收检查 ─────────────────────────────────────────────
print("\n=== 验收检查 ===")
print(f"总场次: {len(df):,}")
result_dist = df["result"].value_counts()
print(f"结果分布:\n{result_dist}")
draw_pct = result_dist.get('D', 0) / len(df) * 100
print(f"平局占比: {draw_pct:.1f}% (目标: 20%-25%)")
print(f"时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
