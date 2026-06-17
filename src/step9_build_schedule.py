"""
step9_build_schedule.py — 生成 wc2026_schedule.csv (104场)
Wikipedia 核实版: 分组 + 淘汰赛括号 + 已完赛结果
"""
import sys, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT  = Path(__file__).parent.parent / "outputs"
RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

# ══════════════════════════════════════════════
#  1. 分组信息（Wikipedia 核实）
# ══════════════════════════════════════════════
GROUP_TEAMS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}
HOSTS = {'United States', 'Canada', 'Mexico'}

# team → group lookup
TEAM_TO_GROUP = {t: g for g, teams in GROUP_TEAMS.items() for t in teams}

# ══════════════════════════════════════════════
#  2. 读取 results.csv 中的 2026 WC 小组赛
# ══════════════════════════════════════════════
df_raw = pd.read_csv(RAW / 'results.csv', parse_dates=['date'])
gs = (df_raw[(df_raw['tournament'] == 'FIFA World Cup') &
             (df_raw['date'].dt.year == 2026)]
      .sort_values('date').copy().reset_index(drop=True))
assert len(gs) == 72, f"期望72场小组赛，得到{len(gs)}"

# 注入已知比分（Wikipedia 核实，2026-06-11 已完赛）
KNOWN = {
    ('Mexico', 'South Africa', '2026-06-11'):        (2, 0),
    ('South Korea', 'Czech Republic', '2026-06-11'): (2, 1),
    # 2026-06-12 比赛 Wikipedia 尚未更新，待用户提供
}
for (ht, at, dt), (hs, as_) in KNOWN.items():
    mask = ((gs['home_team'] == ht) & (gs['away_team'] == at) &
            (gs['date'] == pd.Timestamp(dt)))
    gs.loc[mask, 'home_score'] = hs
    gs.loc[mask, 'away_score'] = as_
    gs.loc[mask, 'result']     = 'H' if hs > as_ else ('A' if hs < as_ else 'D')

# 分配分组
def get_group(row):
    ht = row['home_team']; at = row['away_team']
    gh = TEAM_TO_GROUP.get(ht)
    ga = TEAM_TO_GROUP.get(at)
    if gh and ga and gh == ga:
        return gh
    return '?'

gs['group'] = gs.apply(get_group, axis=1)

# 确定每组内的轮次（按日期顺序，每组前2场=R1，中2场=R2，后2场=R3）
gs['round'] = ''
for grp in 'ABCDEFGHIJKL':
    idx = gs[gs['group'] == grp].index.tolist()
    assert len(idx) == 6, f"Group {grp} 应有6场，得到{len(idx)}"
    idx_sorted = gs.loc[idx].sort_values('date').index.tolist()
    for rank, i in enumerate(idx_sorted):
        gs.loc[i, 'round'] = f'R{rank // 2 + 1}'

# 补充 venue_country（从 results.csv country 列）
if 'country' in gs.columns:
    gs['venue_country'] = gs['country']
else:
    # 根据中立标记和主队推断
    def infer_venue(row):
        if not row['neutral'] and row['home_team'] in HOSTS:
            return {'United States': 'USA', 'Canada': 'Canada', 'Mexico': 'Mexico'}[row['home_team']]
        # 中立：根据日期范围大致分配（简化）
        return 'USA/Canada/Mexico'
    gs['venue_country'] = gs.apply(infer_venue, axis=1)

# ══════════════════════════════════════════════
#  3. 淘汰赛赛程（Wikipedia 核实，32场）
# ══════════════════════════════════════════════
# 格式: match_id, date, stage, group, home_desc, away_desc, venue, venue_country, neutral
KO_MATCHES = [
    # Round of 32 (Match 73-88)
    (73, '2026-06-28', 'R32', 'Runner-up A', 'Runner-up B', 'SoFi Stadium', 'USA', True),
    (74, '2026-06-29', 'R32', 'Winner E', '3rd C/D/F/H (best)', 'Gillette Stadium', 'USA', True),
    (75, '2026-06-29', 'R32', 'Winner F', 'Runner-up C', 'Estadio BBVA', 'Mexico', True),
    (76, '2026-06-29', 'R32', 'Winner C', 'Runner-up F', 'NRG Stadium', 'USA', True),
    (77, '2026-06-30', 'R32', 'Winner I', '3rd C/D/F/G/H (best)', 'MetLife Stadium', 'USA', True),
    (78, '2026-06-30', 'R32', 'Runner-up E', 'Runner-up I', 'AT&T Stadium', 'USA', True),
    (79, '2026-06-30', 'R32', 'Winner A', '3rd C/E/F/H/I (best)', 'Estadio Azteca', 'Mexico', None),
    # ↑ Match 79 neutral depends: if Mexico wins Group A → neutral=False
    (80, '2026-07-01', 'R32', 'Winner L', '3rd E/H/I/J/K (best)', 'Mercedes-Benz Stadium', 'USA', True),
    (81, '2026-07-01', 'R32', 'Winner D', '3rd B/E/F/I/J (best)', "Levi's Stadium", 'USA', None),
    # ↑ Match 81 neutral: if USA wins Group D → neutral=False
    (82, '2026-07-01', 'R32', 'Winner G', '3rd A/E/H/I/J (best)', 'Lumen Field', 'USA', True),
    (83, '2026-07-02', 'R32', 'Runner-up K', 'Runner-up L', 'BMO Field', 'Canada', True),
    (84, '2026-07-02', 'R32', 'Winner H', 'Runner-up J', 'SoFi Stadium', 'USA', True),
    (85, '2026-07-02', 'R32', 'Winner B', '3rd E/F/G/I/J (best)', 'BC Place', 'Canada', None),
    # ↑ Match 85 neutral: if Canada wins Group B → neutral=False
    (86, '2026-07-03', 'R32', 'Winner J', 'Runner-up H', 'Hard Rock Stadium', 'USA', True),
    (87, '2026-07-03', 'R32', 'Winner K', '3rd D/E/I/J/L (best)', 'Arrowhead Stadium', 'USA', True),
    (88, '2026-07-03', 'R32', 'Runner-up D', 'Runner-up G', 'AT&T Stadium', 'USA', True),
    # Round of 16 (Match 89-96)
    (89, '2026-07-04', 'R16', 'Winner M74', 'Winner M77', 'Lincoln Financial Field', 'USA', True),
    (90, '2026-07-04', 'R16', 'Winner M73', 'Winner M75', 'NRG Stadium', 'USA', True),
    (91, '2026-07-05', 'R16', 'Winner M76', 'Winner M78', 'MetLife Stadium', 'USA', True),
    (92, '2026-07-05', 'R16', 'Winner M79', 'Winner M80', 'Estadio Azteca', 'Mexico', True),
    (93, '2026-07-06', 'R16', 'Winner M83', 'Winner M84', 'AT&T Stadium', 'USA', True),
    (94, '2026-07-06', 'R16', 'Winner M81', 'Winner M82', 'Lumen Field', 'USA', True),
    (95, '2026-07-07', 'R16', 'Winner M86', 'Winner M88', 'Mercedes-Benz Stadium', 'USA', True),
    (96, '2026-07-07', 'R16', 'Winner M85', 'Winner M87', 'BC Place', 'Canada', True),
    # QF (Match 97-100)
    (97, '2026-07-09', 'QF', 'Winner M89', 'Winner M90', 'Gillette Stadium', 'USA', True),
    (98, '2026-07-10', 'QF', 'Winner M93', 'Winner M94', 'SoFi Stadium', 'USA', True),
    (99, '2026-07-11', 'QF', 'Winner M91', 'Winner M92', 'Hard Rock Stadium', 'USA', True),
    (100, '2026-07-11', 'QF', 'Winner M95', 'Winner M96', 'Arrowhead Stadium', 'USA', True),
    # SF (Match 101-102)
    (101, '2026-07-14', 'SF', 'Winner M97', 'Winner M98', 'AT&T Stadium', 'USA', True),
    (102, '2026-07-15', 'SF', 'Winner M99', 'Winner M100', 'Mercedes-Benz Stadium', 'USA', True),
    # 3rd place (103) + Final (104)
    (103, '2026-07-18', '3RD', 'Loser M101', 'Loser M102', 'Hard Rock Stadium', 'USA', True),
    (104, '2026-07-19', 'FINAL', 'Winner M101', 'Winner M102', 'MetLife Stadium', 'USA', True),
]

ko_rows = []
for m in KO_MATCHES:
    mid, dt, stage, h_desc, a_desc, venue, v_country, neutral = m
    ko_rows.append({
        'match_id': mid,
        'date': pd.Timestamp(dt),
        'stage': stage,
        'group': '',
        'round': stage,
        'home_team': h_desc,
        'away_team': a_desc,
        'home_score': float('nan'),
        'away_score': float('nan'),
        'neutral': neutral,
        'venue': venue,
        'venue_country': v_country,
        'result': '',
        'tournament': 'FIFA World Cup',
    })
df_ko = pd.DataFrame(ko_rows)

# ══════════════════════════════════════════════
#  4. 合并并保存
# ══════════════════════════════════════════════
gs['match_id'] = range(1, 73)

out_cols = ['match_id', 'date', 'stage', 'group', 'round', 'home_team', 'away_team',
            'home_score', 'away_score', 'result', 'neutral', 'venue_country', 'tournament']

gs_out = gs.copy()
gs_out['stage'] = 'GROUP'
for c in out_cols:
    if c not in gs_out.columns:
        gs_out[c] = ''
gs_out['venue_country'] = gs_out.get('venue_country', '')

df_ko['stage'] = df_ko['round']
df_full = pd.concat(
    [gs_out[out_cols], df_ko[out_cols]],
    ignore_index=True
)
df_full.to_csv(OUT / 'wc2026_schedule.csv', index=False, encoding='utf-8-sig')

# ══════════════════════════════════════════════
#  5. 展示汇总（供用户人工核对）
# ══════════════════════════════════════════════
print("=" * 65)
print("  wc2026_schedule.csv 汇总（请人工核对）")
print("=" * 65)
print(f"\n总场次: {len(df_full)} (小组72 + 淘汰32)")

print("\n【分组一览（各4队）】")
for g in 'ABCDEFGHIJKL':
    teams = GROUP_TEAMS[g]
    host_mark = [f"*{t}" if t in HOSTS else t for t in teams]
    print(f"  Group {g}: {', '.join(host_mark)}")

print("\n【neutral 标记核对（小组赛）】")
print("  仅主队为东道主(美/加/墨)时 neutral=False:")
non_neutral = gs[gs['neutral'] == False][['date','home_team','away_team','neutral']].head(12)
for _, r in non_neutral.iterrows():
    print(f"    {r['date'].date()}  {r['home_team']:<25} vs {r['away_team']:<25} neutral={r['neutral']}")

print("\n【已注入比分（Wikipedia核实）】")
played = gs[gs['home_score'].notna()]
for _, r in played.iterrows():
    print(f"  {r['date'].date()}  {r['home_team']:<20} {int(r['home_score'])}-{int(r['away_score'])} {r['away_team']}")

print("\n【June 12 比赛（待用户提供比分）】")
j12 = gs[gs['date'] == pd.Timestamp('2026-06-12')]
for _, r in j12.iterrows():
    score = f"{int(r['home_score'])}-{int(r['away_score'])}" if pd.notna(r['home_score']) else "?-?"
    print(f"  {r['date'].date()}  {r['home_team']:<25} {score:<6} {r['away_team']}")

print("\n【淘汰赛关键场次（中立标记待确认）】")
print("  Match 79 (2026-06-30 Azteca): Winner A vs 3rd → neutral=None (取决于墨西哥是否出线)")
print("  Match 81 (2026-07-01 Levi's): Winner D vs 3rd → neutral=None (取决于美国是否出线)")
print("  Match 85 (2026-07-02 BC Place): Winner B vs 3rd → neutral=None (取决于加拿大是否出线)")

print(f"\n已保存: outputs/wc2026_schedule.csv")
print("\n请核对上述内容，确认后回复确认，再执行模型训练与预测。")
