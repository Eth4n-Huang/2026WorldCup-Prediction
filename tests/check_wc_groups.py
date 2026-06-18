"""
检查 1998-2010 WC 分组数据与实际数据的球队名匹配情况
"""
import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wc_data import WC_GROUPS, WC_TEAM_GROUP

df = pd.read_csv(r"e:\worldcup\data\raw\results.csv", parse_dates=["date"])

for year in [1998, 2002, 2006, 2010]:
    wc = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"].dt.year == year)]
    # 1998 WC starts in June 1998
    if year == 2002:
        wc = wc[wc["date"] >= f"{year}-05-01"]

    group_teams = set(t for g in WC_GROUPS[year].values() for t in g)

    # 找实际数据中参与WC的球队
    actual_teams = set(wc["home_team"].tolist() + wc["away_team"].tolist())

    # 在WC_GROUPS中但不在实际数据中
    in_groups_not_data = group_teams - actual_teams
    # 在实际数据中但不在WC_GROUPS中（可能是名称问题）
    in_data_not_groups = actual_teams - group_teams

    print(f"\n=== {year} WC ===")
    print(f"WC比赛总数: {len(wc)}")
    print(f"WC_GROUPS中球队数: {len(group_teams)}")
    if in_groups_not_data:
        print(f"  在分组表但数据中找不到: {sorted(in_groups_not_data)}")
    if in_data_not_groups:
        print(f"  在数据中但不在分组表: {sorted(in_data_not_groups)}")
    if not in_groups_not_data and not in_data_not_groups:
        print("  所有球队名匹配 OK")
