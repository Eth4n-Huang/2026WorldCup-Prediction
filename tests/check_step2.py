import pandas as pd

df = pd.read_csv(r"e:\worldcup\data\processed\matches_with_elo.csv", parse_dates=["date"])
df_eval = df[df["date"] >= "1998-01-01"]
split = int(len(df_eval) * 0.8)

print(f"训练集: 1998-01-01 ~ {df_eval.iloc[split-1]['date'].date()}")
print(f"测试集: {df_eval.iloc[split]['date'].date()} ~ {df_eval.iloc[-1]['date'].date()}")
print(f"训练:{split}场  测试:{len(df_eval)-split}场")

test = df_eval.iloc[split:]
print("\n测试集K因子分布:")
print(test["k_factor"].value_counts().sort_index())
print("\n测试集赛事类型(K=20 friendly占比):")
total = len(test)
friendly = (test["k_factor"] == 20).sum()
print(f"Friendly: {friendly}/{total} = {friendly/total:.1%}")
