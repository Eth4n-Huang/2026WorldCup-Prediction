"""
阶段2a: 动态Elo系统
- 初始Elo 1500，处理全部1990+比赛（1990-1997仅热身）
- 每行记录赛前 elo_home_pre / elo_away_pre
- 提供可被其他模块导入的 get_elo(team, date) 函数
输出: data/processed/matches_with_elo.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

INITIAL_ELO = 1500
HOME_ADVANTAGE = 100   # 非中立场主场Elo加成


# ══════════════════════════════════════════════
#  工具函数（可被其他模块导入）
# ══════════════════════════════════════════════

def compute_g(home_score: int, away_score: int) -> float:
    """
    净胜球系数G（参照eloratings.net）:
    净胜≤1球 → 1.0，净胜2球 → 1.5，净胜≥3球 → (11+N)/8
    平局 diff=0 按≤1处理，G=1.0
    """
    diff = abs(home_score - away_score)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11 + diff) / 8


def compute_we(elo_home: float, elo_away: float, neutral: bool) -> float:
    """主队期望胜率 We = 1/(1+10^(-dr/400))，dr含主场优势"""
    H = 0 if neutral else HOME_ADVANTAGE
    dr = (elo_home + H) - elo_away
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def get_elo(team: str, date, _cache: dict = {}) -> float:
    """
    返回 team 在 date 之前的最新Elo（上一场赛后值）。
    首次调用自动加载CSV并缓存，后续调用不重复IO。
    用法: from step2_elo import get_elo
    """
    if "df" not in _cache:
        _cache["df"] = pd.read_csv(
            PROCESSED_DIR / "matches_with_elo.csv", parse_dates=["date"]
        )

    df = _cache["df"]
    date = pd.Timestamp(date)

    home_rows = df[df["home_team"] == team][["date", "elo_home_post"]].rename(
        columns={"elo_home_post": "elo"}
    )
    away_rows = df[df["away_team"] == team][["date", "elo_away_post"]].rename(
        columns={"elo_away_post": "elo"}
    )
    candidates = pd.concat([home_rows, away_rows])
    candidates = candidates[candidates["date"] < date]

    if candidates.empty:
        return float(INITIAL_ELO)
    return float(candidates.sort_values("date").iloc[-1]["elo"])


# ══════════════════════════════════════════════
#  主处理流程（运行此脚本时执行）
# ══════════════════════════════════════════════

if __name__ == "__main__":

    # ── 读取清洗后数据 ──────────────────────────
    df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"读取 matches_clean.csv: {len(df)} 行")

    # ── Elo逐场更新 ────────────────────────────
    elo_now: dict[str, float] = {}   # team -> 当前Elo（赛后值）

    elo_home_pre  = np.empty(len(df))
    elo_away_pre  = np.empty(len(df))
    elo_home_post = np.empty(len(df))
    elo_away_post = np.empty(len(df))

    W_MAP = {"H": 1.0, "D": 0.5, "A": 0.0}

    for i, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        eh = elo_now.get(home, INITIAL_ELO)
        ea = elo_now.get(away, INITIAL_ELO)

        We    = compute_we(eh, ea, bool(row["neutral"]))
        G     = compute_g(int(row["home_score"]), int(row["away_score"]))
        K     = float(row["k_factor"])
        W     = W_MAP[row["result"]]

        delta = K * G * (W - We)

        elo_home_pre[i]  = eh
        elo_away_pre[i]  = ea
        elo_now[home]    = eh + delta
        elo_now[away]    = ea - delta
        elo_home_post[i] = elo_now[home]
        elo_away_post[i] = elo_now[away]

    df["elo_home_pre"]  = elo_home_pre
    df["elo_away_pre"]  = elo_away_pre
    df["elo_home_post"] = elo_home_post
    df["elo_away_post"] = elo_away_post
    # elo_diff = 主队Elo优势（含主场加成），是最重要的单一特征
    H_arr = np.where(df["neutral"].values, 0, HOME_ADVANTAGE)
    df["elo_diff"] = df["elo_home_pre"].values + H_arr - df["elo_away_pre"].values

    df.to_csv(PROCESSED_DIR / "matches_with_elo.csv", index=False)
    print(f"已保存: data/processed/matches_with_elo.csv")

    # ── 验收1: 2022-11-20 Elo前十 ───────────────
    print("\n=== 验收1: 2022-11-20（卡塔尔世界杯开赛前）Elo前十 ===")
    cutoff = pd.Timestamp("2022-11-20")
    before = df[df["date"] < cutoff]

    home_ser = before[["date", "home_team", "elo_home_post"]].rename(
        columns={"home_team": "team", "elo_home_post": "elo"}
    )
    away_ser = before[["date", "away_team", "elo_away_post"]].rename(
        columns={"away_team": "team", "elo_away_post": "elo"}
    )
    snapshot = (
        pd.concat([home_ser, away_ser])
        .sort_values("date")
        .groupby("team")["elo"]
        .last()
        .sort_values(ascending=False)
    )
    top10 = snapshot.head(10)
    print(top10.round(1).to_string())
    print("\n参考 eloratings.net 2022-11-20 前十:")
    print("  Brazil, Argentina, France, Belgium, England, Spain,")
    print("  Netherlands, Portugal, Denmark, Germany")

    # ── 验收2: elo_diff单特征逻辑回归准确率 ──────
    print("\n=== 验收2: elo_diff 单特征 LR 准确率 ===")
    df_eval = df[df["date"] >= "1998-01-01"].copy()
    split = int(len(df_eval) * 0.8)

    X_tr = df_eval["elo_diff"].values[:split].reshape(-1, 1)
    y_tr = df_eval["result"].values[:split]
    X_te = df_eval["elo_diff"].values[split:].reshape(-1, 1)
    y_te = df_eval["result"].values[split:]

    lr = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    acc = lr.score(X_te, y_te)
    print(f"测试集准确率: {acc:.3f}  （目标: 0.45-0.50）")

    y_pred = lr.predict(X_te)
    cm = pd.DataFrame(
        confusion_matrix(y_te, y_pred, labels=["H", "D", "A"]),
        index=["真实H", "真实D", "真实A"],
        columns=["预测H", "预测D", "预测A"],
    )
    print(f"\n混淆矩阵（注意平局预测问题）:\n{cm}")
    draw_predicted = cm["预测D"].sum()
    print(f"\n模型共预测了 {draw_predicted} 场平局（总测试 {len(y_te)} 场）")
    print("→ 若平局预测极少，属正常现象，阶段3会专门处理")

    # ── Elo分布健全性检查 ─────────────────────────
    print("\n=== Elo分布健全性检查 ===")
    final_elos = pd.Series(elo_now).sort_values(ascending=False)
    print(f"Elo最高: {final_elos.index[0]} {final_elos.iloc[0]:.0f}")
    print(f"Elo最低: {final_elos.index[-1]} {final_elos.iloc[-1]:.0f}")
    print(f"均值: {final_elos.mean():.0f}  中位数: {final_elos.median():.0f}")
    print(f"（均值应接近1500，零和系统的自然结果）")
