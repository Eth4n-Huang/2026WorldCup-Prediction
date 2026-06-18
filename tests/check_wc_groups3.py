"""对比WC_GROUPS与处理后数据（标准化名称）"""
import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wc_data import WC_GROUPS

# 使用处理后数据（已经过step1标准化）
df = pd.read_csv(r"e:\worldcup\data\processed\matches_with_elo.csv", parse_dates=["date"])

for year in [1998, 2002, 2006, 2010]:
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == year)]
    group_teams = set(t for g in WC_GROUPS[year].values() for t in g)
    actual_teams = set(wc["home_team"].tolist() + wc["away_team"].tolist())

    not_in_data   = group_teams - actual_teams
    not_in_groups = actual_teams - group_teams

    print(f"\n{year} WC (processed data): {len(wc)} games")
    if not_in_data:
        safe = [t.encode('ascii','replace').decode() for t in sorted(not_in_data)]
        print(f"  In groups but NOT in data: {safe}")
    if not_in_groups:
        safe = [t.encode('ascii','replace').decode() for t in sorted(not_in_groups)]
        print(f"  In data but NOT in groups: {safe}")
    if not not_in_data and not not_in_groups:
        print(f"  All team names match OK")
