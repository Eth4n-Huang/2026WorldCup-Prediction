"""检查 WC 分组名匹配（ASCII only 输出避免编码问题）"""
import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wc_data import WC_GROUPS

df = pd.read_csv(r"e:\worldcup\data\raw\results.csv", parse_dates=["date"])

for year in [1998, 2002, 2006, 2010]:
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == year)]
    group_teams = set(t for g in WC_GROUPS[year].values() for t in g)
    actual_teams = set(wc["home_team"].tolist() + wc["away_team"].tolist())

    not_in_data   = group_teams - actual_teams
    not_in_groups = actual_teams - group_teams

    print(f"\n{year} WC: {len(wc)} games")
    if not_in_data:
        # safe print: encode to ascii, replace non-ascii
        safe = [t.encode('ascii','replace').decode() for t in sorted(not_in_data)]
        print(f"  In groups but NOT in data: {safe}")
    if not_in_groups:
        safe = [t.encode('ascii','replace').decode() for t in sorted(not_in_groups)]
        print(f"  In data but NOT in groups: {safe}")
    if not not_in_data and not not_in_groups:
        print(f"  All team names match OK")
