"""
step3 自查：逐条验证 CLAUDE.md 的要求是否满足
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

df = pd.read_csv(r"e:\worldcup\data\processed\features.csv", parse_dates=["date"])
wc_all = df[df["tournament"] == "FIFA World Cup"]
wc_ko  = wc_all[wc_all["is_knockout"] == 1]
wc22_grp = wc_all[(wc_all["date"] >= "2022-11-20") & (wc_all["date"] <= "2022-12-02")]
wc22_ko  = wc_all[(wc_all["date"] >  "2022-12-02") & (wc_all["date"] <= "2022-12-18")]

print("=== 2b: 近期状态特征 ===")
print(f"home_last5_winrate 范围: [{df['home_last5_winrate'].min():.2f}, {df['home_last5_winrate'].max():.2f}]  (期望0-1)")
print(f"rest_diff 范围: [{df['rest_diff'].min()}, {df['rest_diff'].max()}]  (有正有负正常)")
print(f"home_prev_et 为1的行数: {df['home_prev_et'].sum()}  (有加时记录则>0)")
print(f"2022WC淘汰赛中 home_prev_et 均值: {wc22_ko['home_prev_et'].mean():.2f}")
print(f"home_wc_exp 最大值: {df['home_wc_exp'].max()}  (应有团队经验)")

print("\n=== 2c: 情境/交锋特征 ===")
print(f"confed_home_UEFA 列存在: {'confed_home_UEFA' in df.columns}")
print(f"same_confed WC22小组赛均值: {wc22_grp['same_confed'].mean():.2f}")
print(f"h2h_winrate 范围: [{df['h2h_winrate'].min():.2f}, {df['h2h_winrate'].max():.2f}]  (期望0-1)")
print(f"h2h_n=0 的比例: {(df['h2h_n']==0).mean():.2%}  (早期无历史记录正常)")
print(f"is_host_home WC22中: {wc22_grp['is_host_home'].sum()} 场主场东道主(期望0，卡塔尔作客)")
print(f"is_host_home WC14中: {wc_all[(wc_all['date']>='2014-06-12')&(wc_all['date']<='2014-07-13')]['is_host_home'].sum()} 场 (期望>0，巴西主场)")

print("\n=== 2d: 出线形势 ===")
print(f"2022WC小组赛 qual_status 分布:")
print(wc22_grp["qual_status_home"].value_counts().sort_index().to_string())
print(f"2022WC小组赛 must_win_home=1 场数: {wc22_grp['must_win_home'].sum()}  (R3中应有)")
r3 = wc22_grp[wc22_grp["stage_group_r3"]==1]
print(f"  其中R3场数: {len(r3)}, R3中must_win=1场数: {r3['must_win_home'].sum()}")
print(f"2022WC小组赛 rotation_home=1 场数: {wc22_grp['rotation_home'].sum()}")

print("\n=== 2e: 阶段特征 (关键) ===")
print(f"淘汰赛 must_win_home 均值: {wc_ko['must_win_home'].mean():.2f}  (CLAUDE.md要求=1.0)")
print(f"淘汰赛 must_win_away 均值: {wc_ko['must_win_away'].mean():.2f}  (CLAUDE.md要求=1.0)")
print(f"淘汰赛 qual_status_home 均值: {wc_ko['qual_status_home'].mean():.2f}  (期望=1中性值)")
print(f"淘汰赛 rotation_home 均值: {wc_ko['rotation_home'].mean():.2f}  (期望=0)")

print("\n=== WC 1998-2010 阶段标记检查 ===")
wc_old = wc_all[wc_all["date"] < "2014-01-01"]
stage_cols = [c for c in df.columns if c.startswith("stage_")]
print(f"stage列: {stage_cols}")
if wc_old[stage_cols].sum().sum() > 0:
    print("WC 1998-2010 阶段分布:")
    print(wc_old[stage_cols].sum().to_string())
else:
    print("所有WC 1998-2010 阶段标记均为0 (bug: 应有group/knockout区分)")
