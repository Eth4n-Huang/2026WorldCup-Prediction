"""
step6c_apply_elo.py — 按预注册规则应用新 Elo 参数到持久化文件
预注册规则: CI 跨零时默认采用训练期最优新配置（不要求显著性）
新参数: H_adv=125, K_major=40, K_wc=60, K_qual=40, K_friendly=20, G=original
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from step6_elo_opt import compute_elo_series

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR  = Path(__file__).parent.parent / "outputs"

def main():
    print("=" * 60)
    print("  step6c: 应用新 Elo 参数（预注册规则）")
    print("=" * 60)

    with open(OUT_DIR / "elo_best_params.json") as f:
        best = json.load(f)
    H_ADV = float(best["H_adv"])

    new_params = {
        "H_adv": best["H_adv"], "K_wc": best["K_wc"],
        "K_major": best["K_major"], "K_qual": best["K_qual"],
        "K_friendly": best["K_friendly"], "G_func": best["G_func"],
    }
    print(f"\n新参数: {new_params}")

    # ── 1. 更新 matches_with_elo.csv ─────────────────────
    print("\n重算 Elo 序列...")
    df_clean = pd.read_csv(PROC_DIR / "matches_clean.csv", parse_dates=["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    df_elo = compute_elo_series(df_clean, new_params)
    H_arr = np.where(df_elo["neutral"].values.astype(bool), 0.0, H_ADV)
    df_elo["elo_diff"] = (df_elo["elo_home_pre"].values + H_arr
                          - df_elo["elo_away_pre"].values)
    df_elo.to_csv(PROC_DIR / "matches_with_elo.csv", index=False)
    print(f"  已保存: data/processed/matches_with_elo.csv  ({len(df_elo)} 行)")

    # ── 2. 更新 features.csv（只替换 Elo 相关列）────────
    print("\n更新 features.csv ...")
    df_feat = pd.read_csv(PROC_DIR / "features.csv", parse_dates=["date"])
    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    key_cols = ["date", "home_team", "away_team"]
    elo_sub  = df_elo[key_cols + ["elo_home_pre", "elo_away_pre"]].copy()

    df_feat = df_feat.drop(
        columns=["elo_home_pre", "elo_away_pre", "elo_diff"], errors="ignore")
    df_feat = df_feat.merge(elo_sub, on=key_cols, how="left")
    H_arr2  = np.where(df_feat["neutral"].values.astype(bool), 0.0, H_ADV)
    df_feat["elo_diff"] = (df_feat["elo_home_pre"].values + H_arr2
                           - df_feat["elo_away_pre"].values)
    df_feat.to_csv(PROC_DIR / "features.csv", index=False)
    print(f"  已保存: data/processed/features.csv  ({len(df_feat)} 行)")

    # ── 3. 更新 elo_best_params.json 标注采用状态 ────────
    best["adopted"] = True
    best["adoption_rule"] = "CI跨零时默认采用训练期最优（预注册规则）"
    best["dev_ll_diff_full"] = -0.00060
    best["dev_ll_diff_wec"]  = -0.00621
    best["dev_p_full"]  = 0.912
    best["dev_p_wec"]   = 0.182
    with open(OUT_DIR / "elo_best_params.json", "w") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print("\n  已更新: outputs/elo_best_params.json (adopted=True)")

    # ── 4. 更新 final_model_spec.md ──────────────────────
    spec_path = OUT_DIR / "final_model_spec.md"
    existing  = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""

    addendum = """

## Elo 超参最终采纳记录 (step6c, 预注册规则)

**采纳参数**: H_adv=125, K_wc=60, K_major=40, K_qual=40, K_friendly=20, G=original
**选参依据**: 1998~2014-06 训练期滚动 BLa LogLoss（训练期 ΔLL=−0.00207）
**dev 配对检验**: 方向一致但未达显著（全dev p=0.91, WC+Copa p=0.18）
**采纳理由**: 按预注册规则——增益CI跨零时默认采用训练期最优新配置
**文件更新**: matches_with_elo.csv / features.csv 均已切换到新 Elo
"""
    spec_path.write_text(existing + addendum, encoding="utf-8")
    print("  已更新: outputs/final_model_spec.md")

    # ── 5. 健全性检查 ─────────────────────────────────────
    print("\n健全性检查（2022-11-20 Elo前5）:")
    cutoff = pd.Timestamp("2022-11-20")
    before = df_elo[df_elo["date"] < cutoff]
    home_s = before[["date","home_team","elo_home_post"]].rename(
        columns={"home_team":"team","elo_home_post":"elo"})
    away_s = before[["date","away_team","elo_away_post"]].rename(
        columns={"away_team":"team","elo_away_post":"elo"})
    snap = (pd.concat([home_s, away_s])
              .sort_values("date").groupby("team")["elo"].last()
              .sort_values(ascending=False).head(5))
    print(snap.round(1).to_string())

    print("\n[step6c 完成] 新 Elo 已全面写入; 下游模型从 features.csv 读取")


if __name__ == "__main__":
    main()
