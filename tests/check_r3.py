"""检查哪些 1998-2010 WC R3 场次仍未识别"""
import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wc_data import WC_GROUPS, WC_GROUP_STAGE_END

df = pd.read_csv(r"e:\worldcup\data\processed\features.csv", parse_dates=["date"])

for year in [1998, 2002, 2006, 2010]:
    end = WC_GROUP_STAGE_END[year]
    wc_grp = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"].dt.year == year) &
        (df["date"].astype(str).str[:10] <= end)
    ]
    r3_found = wc_grp["stage_group_r3"].sum()
    r3_exp   = 16  # 8 groups × 2 games per group
    print(f"{year}: group_r3 found={int(r3_found)}, expected=16, missing={16-int(r3_found)}")
    if r3_found < 16:
        # 找哪个组的最后两场被漏掉
        wc_grp_stage_cols = [c for c in df.columns if c.startswith("stage_")]
        non_r3 = wc_grp[wc_grp["stage_group_r3"] == 0]
        # 检查 knockout 标记的场次（应为0）
        ko = non_r3[non_r3["is_knockout"] == 1]
        if len(ko):
            print(f"  Misclassified as knockout: {len(ko)} games")
            print(ko[["date","home_team","away_team","is_knockout"]].to_string(index=False))
