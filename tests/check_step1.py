import pandas as pd

df = pd.read_csv(r'e:\worldcup\data\raw\results.csv')
df = df[df['date'] >= '1990-01-01']

# 验证bug1：Copa América 能否被匹配
copa = df[df['tournament'].str.contains('Copa Am', na=False)]
print(f"Copa América 1990后场次: {len(copa)}")
test = 'Copa América'.lower()
print(f'"copa america" in "{test}": {"copa america" in test}')

# 验证bug2：African Cup 能否被匹配
africa = df[df['tournament'] == 'African Cup of Nations']
print(f"\nAfrican Cup of Nations 1990后场次: {len(africa)}")
test2 = 'African Cup of Nations'.lower()
print(f'"africa cup of nations" in "{test2}": {"africa cup of nations" in test2}')

# 查当前 K=50 实际落到哪些赛事
clean = pd.read_csv(r'e:\worldcup\data\processed\matches_clean.csv')
k50 = clean[clean['k_factor'] == 50]['tournament'].value_counts()
print(f"\n当前 K=50 的赛事（前15）:")
print(k50.head(15).to_string())

k50_total = k50.sum()
print(f"\nK=50 总场次: {k50_total}")
print(f"Copa América 场次: {len(copa)}")
print(f"African Cup 场次: {len(africa)}")
