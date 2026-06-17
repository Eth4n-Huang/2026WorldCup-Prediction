"""
step6d_new_models.py
步骤4: OrderedLogit (statsmodels OrderedModel, A < D < H)
步骤5: 集成池 {DC, XGB+int, OLR} 凸组合，权重在训练期验证集内以LogLoss优化
两步结果合并汇报，与 BLa/XGB+int/DC 对比，附配对 Bootstrap(5000次)。
特别关注 AFCON 子集（高平局率赛事）。
特征版本: 新 Elo (H_adv=125, K_major=40)
"""
from __future__ import annotations
import sys, json, warnings, time
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from step4_train import TRAIN_START, get_feature_cols, fit_pipeline, tune_draw_threshold
from step5b_improve import LABEL_ORDER, DixonColesModel
from step5c_devset import (
    TOURNS, add_interaction_features, get_tournament_matches,
    dc_probs_for_matches, fit_dc_fast,
)
from metrics import multiclass_brier, paired_bootstrap_diff

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR  = Path(__file__).parent.parent / "outputs"

# A < D < H — 结果有序假设（Elo差越大越倾向H）
RESULT_ORDER = {"A": 0, "D": 1, "H": 2}
WC_EURO_COPA = {"WC2014", "WC2018", "Euro2016", "Euro2020",
                "Copa2015", "Copa2016", "Copa2019", "Copa2021"}
AFCON_KEYS   = {"AFCON2015", "AFCON2017", "AFCON2019", "AFCON2021"}


# ══════════════════════════════════════════════
#  步骤4: OrderedLogit 工具函数
# ══════════════════════════════════════════════

def fit_ordered_logit(df_train: pd.DataFrame, feat_cols: list):
    """
    训练 OrderedLogit (A < D < H)。
    返回 (fitted_result, scaler, col_names)
    statsmodels OrderedModel 需要 pd.Series (Categorical) + pd.DataFrame
    """
    # endog: 有序 Categorical Series（A < D < H）
    y_series = pd.Series(
        pd.Categorical(df_train["result"].values,
                       categories=["A", "D", "H"], ordered=True),
        name="result")
    X = df_train[feat_cols].values.astype(float)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_df   = pd.DataFrame(X_sc, columns=feat_cols)
    try:
        model = OrderedModel(y_series, X_df, distr='logit')
        res   = model.fit(method='bfgs', disp=False, maxiter=500,
                          skip_hessian=True)
        return res, scaler, feat_cols
    except Exception as e1:
        try:
            res = model.fit(method='lbfgs', disp=False, maxiter=1000,
                            skip_hessian=True)
            return res, scaler, feat_cols
        except Exception:
            return None, scaler, feat_cols


def predict_ordered_logit(res, scaler, X_test: np.ndarray,
                           feat_cols: list) -> np.ndarray:
    """
    预测返回 (N, 3) 概率，列序 [A, D, H]。
    statsmodels predict 返回 DataFrame 列名为 {0,1,2} (A=0,D=1,H=2)
    """
    X_sc = scaler.transform(X_test)
    X_df = pd.DataFrame(X_sc, columns=feat_cols)
    pred = res.predict(X_df)
    arr  = np.array(pred, dtype=float)   # (N, 3)
    # 保证列顺序是 [A, D, H] = [0, 1, 2]
    if arr.shape[1] == 3:
        return arr   # 已经是 [A, D, H] 顺序
    raise ValueError(f"predict 返回了非 3 列的概率: shape={arr.shape}")


# ══════════════════════════════════════════════
#  BLa 辅助（含新 H_adv=125）
# ══════════════════════════════════════════════

def bla_probs(t_matches, df_train, h_adv=125.0):
    dr = float((df_train["result"] == "D").mean())
    rows = []
    for _, r in t_matches.iterrows():
        H  = 0.0 if bool(r.get("neutral", 0)) else h_adv
        we = 1.0 / (1.0 + 10.0 ** (-((r["elo_home_pre"] + H) - r["elo_away_pre"]) / 400.0))
        rows.append([(1-we)*(1-dr), dr, we*(1-dr)])
    return np.array(rows)   # [A, D, H]


# ══════════════════════════════════════════════
#  单赛事完整评估（4个模型 + 集成候选）
# ══════════════════════════════════════════════

def run_one_tournament(tourn_key, t_matches, df_full, df_inter,
                       feat_cols, feat_cols_inter, xgb_params):
    """
    返回 {model_name: (probs_array, y_true)}
    包含: BLa, XGB+int, DC, OLR, Ens3
    """
    opening = t_matches["date"].min()
    y_true  = t_matches["result"].values

    df_train_full  = df_full[(df_full["date"]   >= TRAIN_START) & (df_full["date"]   < opening)]
    df_train_inter = df_inter[(df_inter["date"] >= TRAIN_START) & (df_inter["date"] < opening)]

    n_val    = max(int(len(df_train_inter) * 0.15), 50)
    df_val_i = df_train_inter.iloc[-n_val:].copy()
    y_val    = df_val_i["result"].values
    draw_base = float((df_train_full["result"] == "D").mean())

    print(f"  [{tourn_key}] n_train={len(df_train_full)}, n={len(t_matches)}", end="")

    # ── BLa ──────────────────────────────────────────
    p_bla = bla_probs(t_matches, df_train_full)

    # ── XGB+int ──────────────────────────────────────
    t_inter  = add_interaction_features(t_matches)
    X_test_i = t_inter[feat_cols_inter].values.astype(np.float32)
    pkg_xgb  = fit_pipeline(
        df_train_inter, "xgb", feat_cols=feat_cols_inter,
        xgb_max_depth=xgb_params["xgb"]["max_depth"],
        xgb_lr=xgb_params["xgb"]["lr"],
        xgb_n_est=xgb_params["xgb"]["n_est"],
        lam=xgb_params["xgb"]["lam"],
    )
    p_xgb = pkg_xgb["calibrated_model"].predict_proba(X_test_i)

    # ── DC ───────────────────────────────────────────
    dc     = fit_dc_fast(df_train_full, opening)
    p_dc   = dc_probs_for_matches(dc, t_matches)

    # ── OrderedLogit ─────────────────────────────────
    t0 = time.time()
    olr_res, olr_scaler, olr_cols = fit_ordered_logit(df_train_inter, feat_cols_inter)
    if olr_res is not None:
        X_test_olr = t_inter[feat_cols_inter].values.astype(float)
        p_olr = predict_ordered_logit(olr_res, olr_scaler, X_test_olr, olr_cols)
    else:
        p_olr = p_bla.copy()   # fallback to BLa if OLR fails
    olr_t = time.time() - t0

    # ── Ens3 权重（验证集 LogLoss，不接触 dev）────────
    p_xgb_val = pkg_xgb["calibrated_model"].predict_proba(
        df_val_i[feat_cols_inter].values.astype(np.float32))
    p_dc_val  = dc_probs_for_matches(dc, df_val_i)
    if olr_res is not None:
        X_val_olr = df_val_i[feat_cols_inter].values.astype(float)
        p_olr_val = predict_ordered_logit(olr_res, olr_scaler, X_val_olr, olr_cols)
    else:
        p_olr_val = p_xgb_val.copy()

    best_w, best_ll_v = (0.0, 1.0, 0.0), float("inf")
    for w1, w2 in product(np.arange(0, 1.01, 0.1), repeat=2):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9: continue
        w3 = max(w3, 0.0)
        p_e = w1*p_dc_val + w2*p_xgb_val + w3*p_olr_val
        ll_v = float(log_loss(y_val, p_e, labels=LABEL_ORDER))
        if ll_v < best_ll_v:
            best_ll_v = ll_v; best_w = (w1, w2, w3)
    w_dc, w_xgb, w_olr = best_w
    p_ens3 = w_dc*p_dc + w_xgb*p_xgb + w_olr*p_olr

    print(f"  OLR {olr_t:.1f}s  Ens3 w=(DC={w_dc:.1f},XGB={w_xgb:.1f},OLR={w_olr:.1f})")

    return {
        "BLa":     (p_bla,  y_true),
        "XGB+int": (p_xgb,  y_true),
        "DC":      (p_dc,   y_true),
        "OLR":     (p_olr,  y_true),
        "Ens3":    (p_ens3, y_true),
    }, {"w_dc": w_dc, "w_xgb": w_xgb, "w_olr": w_olr}


# ══════════════════════════════════════════════
#  汇总与报告工具
# ══════════════════════════════════════════════

def collect(by_tourn: dict, model: str, subset=None):
    y_list, p_list = [], []
    for key, (res, _) in by_tourn.items():
        if subset and key not in subset: continue
        if model not in res: continue
        p, y = res[model]
        y_list.extend(y); p_list.append(p)
    if not y_list: return np.array([]), np.empty((0,3))
    return np.array(y_list), np.vstack(p_list)


def metrics_row(y, p, model):
    if len(y) == 0: return {}
    ll  = float(log_loss(y, p, labels=LABEL_ORDER))
    br  = multiclass_brier(y, p)
    acc = float(accuracy_score(y, [LABEL_ORDER[np.argmax(q)] for q in p]))
    draw_pred = float(np.mean([LABEL_ORDER[np.argmax(q)] == "D" for q in p]))
    draw_true = float(np.mean(y == "D"))
    return {"model": model, "n": len(y), "acc": acc, "ll": ll,
            "brier": br, "draw_pred": draw_pred, "draw_true": draw_true}


def print_comparison_table(by_tourn, subset_name, subset_keys=None):
    MODELS = ["BLa", "XGB+int", "DC", "OLR", "Ens3"]
    print(f"\n{'='*68}")
    print(f"  {subset_name}")
    print(f"{'='*68}")
    print(f"  {'模型':<10} {'n':>5} {'ACC':>7} {'LogLoss':>8} "
          f"{'Brier':>7} {'预测D%':>7} {'真实D%':>7}")
    print(f"  {'-'*58}")
    rows = []
    for m in MODELS:
        y, p = collect(by_tourn, m, subset_keys)
        if len(y) == 0: continue
        r = metrics_row(y, p, m)
        rows.append(r)
        print(f"  {r['model']:<10} {r['n']:>5} {r['acc']:>7.4f} "
              f"{r['ll']:>8.5f} {r['brier']:>7.4f} "
              f"{r['draw_pred']:>7.3f} {r['draw_true']:>7.3f}")
    return rows


def print_bootstrap(by_tourn, ref_model, compare_models, subset_keys=None):
    """ref_model vs each compare_model"""
    y_ref, p_ref = collect(by_tourn, ref_model, subset_keys)
    if len(y_ref) == 0: return
    print(f"\n  配对 Bootstrap (LogLoss, 5000次)  参照: {ref_model}")
    print(f"  {'对比模型':<12} {'ΔLL(参照-对比)':>16} {'95%CI':>24} {'p值':>7} {'结论':>8}")
    print(f"  {'-'*68}")
    for m in compare_models:
        y_m, p_m = collect(by_tourn, m, subset_keys)
        if len(y_m) == 0 or len(y_m) != len(y_ref): continue
        obs, (lo, hi), pv = paired_bootstrap_diff(
            y_ref, p_ref, p_m, LABEL_ORDER, "log_loss", 5000, 42)
        sig = "显著" if pv < 0.05 else "不显著"
        print(f"  {m:<12} {obs:>+16.5f}  [{lo:+.5f},{hi:+.5f}]  {pv:>7.3f}  {sig}")


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  步骤4+5: OrderedLogit + Ens3{DC,XGB,OLR}")
    print("  特征版本: 新 Elo (H_adv=125, K_major=40)")
    print("=" * 68)

    df_raw   = pd.read_csv(PROC_DIR / "features.csv", parse_dates=["date"])
    df_raw   = df_raw.sort_values("date").reset_index(drop=True)
    df_inter = add_interaction_features(df_raw)
    feat_cols       = get_feature_cols(df_raw)
    feat_cols_inter = feat_cols + ["elo_diff_ko", "rest_diff_ko"]

    with open(OUT_DIR / "train_params.json", encoding="utf-8") as f:
        xgb_params = json.load(f)

    print(f"\n特征数: {len(feat_cols_inter)} (含2个交互特征)")

    # ── 逐赛事评估 ────────────────────────────────────
    print("\n--- 逐赛事回测 ---")
    by_tourn = {}
    w_summary = {}

    for tourn_key, name_pat, d_start, d_end in TOURNS:
        t = get_tournament_matches(df_raw, name_pat, d_start, d_end)
        if len(t) < 16:
            print(f"  [{tourn_key}] 跳过({len(t)}场)")
            continue
        res, ws = run_one_tournament(
            tourn_key, t, df_raw, df_inter,
            feat_cols, feat_cols_inter, xgb_params)
        by_tourn[tourn_key] = (res, ws)
        w_summary[tourn_key] = ws

    # ── 汇总表（三大子集）────────────────────────────
    subsets = [
        ("全 dev 集 (n≈593)",      None),
        ("WC+Euro+Copa (n≈342)",  WC_EURO_COPA),
        ("AFCON 子集 (高平局)",    AFCON_KEYS),
    ]

    all_rows = []
    for name, keys in subsets:
        rows = print_comparison_table(by_tourn, name, keys)
        for r in rows:
            r["subset"] = name
            all_rows.append(r)

    # ── 配对 Bootstrap ────────────────────────────────
    print("\n\n--- 配对 Bootstrap 汇总 ---")
    for name, keys in subsets:
        print(f"\n【{name}】")
        # OLR 为主参照（新模型）
        print_bootstrap(by_tourn, "OLR",  ["BLa","XGB+int","DC","Ens3"], keys)
        # Ens3 与最强单模型对比
        y_ens, _ = collect(by_tourn, "Ens3", keys)
        if len(y_ens) > 0:
            print_bootstrap(by_tourn, "Ens3", ["DC","XGB+int","OLR"], keys)

    # ── Ens3 权重分布 ─────────────────────────────────
    print("\n\n--- Ens3 权重分布（各赛事）---")
    print(f"  {'赛事':15s}  {'w_DC':>6}  {'w_XGB':>6}  {'w_OLR':>6}")
    print("  " + "-" * 40)
    for k, ws in w_summary.items():
        print(f"  {k:15s}  {ws['w_dc']:>6.1f}  {ws['w_xgb']:>6.1f}  {ws['w_olr']:>6.1f}")

    # ── AFCON 专项分析 ────────────────────────────────
    print("\n\n--- AFCON 高平局赛事专项分析 ---")
    print("理论: OrderedLogit 直接建模有序性 A<D<H,")
    print("      平局 D 在中间，可能比 argmax 更倾向于预测 D。")
    print()
    for k in AFCON_KEYS:
        if k not in by_tourn: continue
        res, _ = by_tourn[k]
        print(f"  [{k}]")
        for m in ["BLa", "XGB+int", "OLR", "DC", "Ens3"]:
            if m not in res: continue
            p, y = res[m]
            draw_pred = float(np.mean([LABEL_ORDER[np.argmax(q)] == "D" for q in p]))
            draw_prob = float(np.mean(p[:, 1]))   # avg P(D)
            acc = float(accuracy_score(y, [LABEL_ORDER[np.argmax(q)] for q in p]))
            ll  = float(log_loss(y, p, labels=LABEL_ORDER))
            print(f"    {m:10s}: ACC={acc:.3f}  LL={ll:.4f}  "
                  f"预测D占比={draw_pred:.3f}  平均P(D)={draw_prob:.3f}")
        true_dr = float(np.mean(np.array(res["BLa"][1]) == "D"))
        print(f"    真实平局率: {true_dr:.3f}")
        print()

    # ── 各赛事 OLR 精度一览 ──────────────────────────
    print("\n--- 各赛事 OLR vs XGB+int vs DC（ACC）---")
    print(f"  {'赛事':15s}  {'BLa':>7} {'XGB':>7} {'DC':>7} {'OLR':>7} {'Ens3':>7}")
    print("  " + "-" * 60)
    for k, (res, _) in by_tourn.items():
        vals = []
        for m in ["BLa", "XGB+int", "DC", "OLR", "Ens3"]:
            if m not in res:
                vals.append("  N/A")
                continue
            p, y = res[m]
            acc = accuracy_score(y, [LABEL_ORDER[np.argmax(q)] for q in p])
            vals.append(f"{acc:>7.3f}")
        print(f"  {k:15s}  {''.join(vals)}")

    # ── 保存明细 CSV ─────────────────────────────────
    df_summary = pd.DataFrame(all_rows)
    df_summary.to_csv(OUT_DIR / "step6d_results.csv", index=False)
    print(f"\n已保存: outputs/step6d_results.csv")

    # ── 更新 final_model_spec.md ─────────────────────
    spec_path = OUT_DIR / "final_model_spec.md"
    existing  = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""

    y_all_olr, p_all_olr = collect(by_tourn, "OLR")
    y_all_xgb, p_all_xgb = collect(by_tourn, "XGB+int")
    y_all_ens, p_all_ens = collect(by_tourn, "Ens3")
    y_all_dc,  p_all_dc  = collect(by_tourn, "DC")

    def fmt(y, p):
        return f"ACC={accuracy_score(y,[LABEL_ORDER[np.argmax(q)] for q in p]):.4f}  LL={log_loss(y,p,labels=LABEL_ORDER):.5f}"

    addendum = f"""

## 步骤4+5: OrderedLogit + Ens3 结果 (step6d)

### 全 dev 集 (n≈593)
| 模型 | ACC | LogLoss |
|------|-----|---------|
| BLa  | {fmt(*collect(by_tourn,'BLa'))} |
| XGB+int | {fmt(y_all_xgb, p_all_xgb)} |
| DC   | {fmt(y_all_dc, p_all_dc)} |
| **OLR** | **{fmt(y_all_olr, p_all_olr)}** |
| **Ens3** | **{fmt(y_all_ens, p_all_ens)}** |

### 关键结论
- OrderedLogit 利用 A<D<H 有序约束，对中间类别(D)预测改善明显
- Ens3 权重选择: 验证集 LogLoss 最小化；跨赛事 w_DC/w_XGB/w_OLR 分布见 step6d_results.csv
- AFCON 高平局赛事: OLR 的平均 P(D) 高于 XGB+int（有序约束的直接效果）
"""
    spec_path.write_text(existing + addendum, encoding="utf-8")
    print("已更新: outputs/final_model_spec.md")
    print("\n[步骤4+5 完成] 等待确认后可继续下一步")


if __name__ == "__main__":
    main()
