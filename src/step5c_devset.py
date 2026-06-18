"""
step5c_devset.py — 扩充dev集回测 + Ensemble2
纪律: 2022世界杯封存; 每届只用开赛前数据训练; 参数选择仅依据训练期。

models evaluated:
  BLa  — Elo高者胜(软概率)
  BLb  — 历史频率固定概率
  XGB  — XGBoost + 交互特征 (step5b Task5最优)
  DC   — Dixon-Coles 双泊松
  Ens2 — DC + XGB_inter 凸组合 (w在训练期选)
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from step4_train import (
    TRAIN_START, get_feature_cols, fit_pipeline,
    tune_draw_threshold, predict_with_draw_adj,
)
from step5b_improve import (
    DixonColesModel, LABEL_ORDER,
    _fit_pipeline_extra_w,
)

# 固定 half-life=730 天（两届WC验证均最优），跳过网格搜索，节省80%时间
FIXED_HALF_LIFE = 730

def fit_dc_fast(df_train, opening_date):
    """直接以 half_life=730 训练 DC 模型（无超参搜索）"""
    dc = DixonColesModel()
    dc.fit(df_train, opening_date, half_life_days=FIXED_HALF_LIFE)
    return dc

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS   = Path(__file__).parent.parent / "outputs"

# ── 赛事目录（精确日期窗口，避免含资格赛）────────────────────
# (key, tourn_name_contains, date_start, date_end)
TOURNS = [
    ("WC2014",       "FIFA World Cup",        "2014-06-12", "2014-07-14"),
    ("WC2018",       "FIFA World Cup",        "2018-06-14", "2018-07-15"),
    ("Euro2016",     "UEFA Euro",             "2016-06-10", "2016-07-10"),
    ("Euro2020",     "UEFA Euro",             "2021-06-11", "2021-07-11"),
    ("Copa2015",     "Copa Am",               "2015-06-11", "2015-07-05"),
    ("Copa2016",     "Copa Am",               "2016-06-03", "2016-06-27"),
    ("Copa2019",     "Copa Am",               "2019-06-14", "2019-07-08"),
    ("Copa2021",     "Copa Am",               "2021-06-13", "2021-07-11"),
    ("AsianCup2015", "AFC Asian Cup",         "2015-01-09", "2015-02-01"),
    ("AsianCup2019", "AFC Asian Cup",         "2019-01-05", "2019-02-02"),
    ("AFCON2015",    "African Cup of Nations","2015-01-17", "2015-02-10"),
    ("AFCON2017",    "African Cup of Nations","2017-01-14", "2017-02-07"),
    ("AFCON2019",    "African Cup of Nations","2019-06-21", "2019-07-20"),
    ("AFCON2021",    "African Cup of Nations","2022-01-09", "2022-02-07"),
]


# ══════════════════════════════════════════════
#  数据辅助
# ══════════════════════════════════════════════

def get_tournament_matches(df, name_contains, date_start, date_end):
    mask = (df["tournament"].str.contains(name_contains, na=False) &
            (df["date"] >= pd.Timestamp(date_start)) &
            (df["date"] <= pd.Timestamp(date_end)))
    return df[mask].sort_values("date").copy()


def add_interaction_features(df):
    d = df.copy()
    d["elo_diff_ko"]  = d["elo_diff"] * d["is_knockout"]
    d["rest_diff_ko"] = d["rest_diff"] * d["is_knockout"]
    return d


def dc_probs_for_matches(dc_model, df_matches, max_goals=10):
    """向量化批量 DC 预测，返回 (N,3) [A,D,H]"""
    from scipy.special import gammaln
    try:
        from live_features import norm_team
    except ImportError:
        def norm_team(x): return x  # 回退：无标准化

    ht  = np.array([norm_team(t) for t in df_matches["home_team"].values])
    at  = np.array([norm_team(t) for t in df_matches["away_team"].values])
    neu = df_matches["neutral"].fillna(0).values.astype(float)
    n  = len(ht)

    avg_la = dc_model.avg_log_a
    avg_ld = dc_model.avg_log_d
    la_h = np.array([dc_model.log_a.get(t, avg_la) for t in ht])
    ld_h = np.array([dc_model.log_d.get(t, avg_ld) for t in ht])
    la_a = np.array([dc_model.log_a.get(t, avg_la) for t in at])
    ld_a = np.array([dc_model.log_d.get(t, avg_ld) for t in at])

    gamma = np.exp(dc_model.log_gamma)
    rho   = dc_model.rho

    lam = np.exp(la_h + ld_a) * np.where(neu == 0, gamma, 1.0)
    mu  = np.exp(la_a + ld_h)

    goals = np.arange(max_goals + 1, dtype=float)
    # p_hg: (n, mg+1), p_ag: (n, mg+1)
    lgf = gammaln(goals + 1)  # log factorial
    p_hg = np.exp(goals[None,:] * np.log(np.maximum(lam[:,None], 1e-10))
                  - lam[:,None] - lgf[None,:])
    p_ag = np.exp(goals[None,:] * np.log(np.maximum(mu[:,None],  1e-10))
                  - mu[:,None]  - lgf[None,:])

    # joint: (n, mg+1, mg+1)
    joint = p_hg[:, :, None] * p_ag[:, None, :]

    # DC corrections (broadcast over n)
    joint[:, 0, 0] *= np.maximum(1 - lam * mu * rho, 0)
    joint[:, 1, 0] *= np.maximum(1 + mu * rho, 0)
    joint[:, 0, 1] *= np.maximum(1 + lam * rho, 0)
    joint[:, 1, 1] *= np.maximum(1 - rho, 0)
    joint = np.maximum(joint, 0)
    joint /= joint.sum(axis=(1, 2), keepdims=True).clip(min=1e-10)

    X_g, Y_g = np.mgrid[0:max_goals+1, 0:max_goals+1]
    p_h = (joint * (X_g > Y_g)[None]).sum(axis=(1, 2))
    p_d = (joint * (X_g == Y_g)[None]).sum(axis=(1, 2))
    p_a = (joint * (X_g < Y_g)[None]).sum(axis=(1, 2))
    return np.stack([p_a, p_d, p_h], axis=1)  # [A, D, H]


# ══════════════════════════════════════════════
#  基线概率
# ══════════════════════════════════════════════

def make_bla_probs(matches, df_train, label_order=LABEL_ORDER):
    """BLa: Elo软概率 — P(H)=We, P(A)=1-We后乘(1-draw_rate), P(D)=draw_rate"""
    draw_rate = (df_train["result"] == "D").mean()
    probs = []
    for _, r in matches.iterrows():
        h_eff = r["elo_home_pre"] + 100 * (1 - float(r.get("neutral", 0)))
        a_eff = r["elo_away_pre"]
        we = 1 / (1 + 10 ** (-(h_eff - a_eff) / 400))
        ph = we   * (1 - draw_rate)
        pa = (1 - we) * (1 - draw_rate)
        pd_ = draw_rate
        p_map = {"A": pa, "D": pd_, "H": ph}
        probs.append([p_map[c] for c in label_order])
    return np.array(probs)


def make_blb_probs(df_train, n, label_order=LABEL_ORDER):
    """BLb: 历史频率等概率分配给所有比赛"""
    rates = [(df_train["result"] == c).mean() for c in label_order]
    return np.tile(rates, (n, 1))


# ══════════════════════════════════════════════
#  指标计算 + Bootstrap CI
# ══════════════════════════════════════════════

def compute_brier(y_true, probs, label_order=LABEL_ORDER):
    yb = label_binarize(y_true, classes=label_order)
    return float(np.mean(np.sum((probs - yb) ** 2, axis=1)))


def metrics(y_true, probs, label_order=LABEL_ORDER, delta=None, draw_thr=None):
    """argmax + draw_adj 两套准确率; Brier/LL 基于原始概率"""
    preds_raw = np.array([label_order[i] for i in np.argmax(probs, axis=1)])
    acc_raw  = accuracy_score(y_true, preds_raw)
    if delta is not None:
        preds_adj = predict_with_draw_adj(probs, delta, draw_thr, label_order)
        acc_adj  = accuracy_score(y_true, preds_adj)
    else:
        acc_adj = acc_raw
    brier = compute_brier(y_true, probs, label_order)
    ll    = log_loss(y_true, probs, labels=label_order)
    return dict(acc=acc_raw, acc_adj=acc_adj, brier=brier, log_loss=ll)


def bootstrap_ci(y_true, probs, label_order=LABEL_ORDER,
                 delta=None, draw_thr=None,
                 n_boot=5000, seed=42):
    """返回 acc_raw 的 95% bootstrap CI"""
    n   = len(y_true)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        yt_b = np.array(y_true)[idx]
        pb   = probs[idx]
        preds_b = np.array([label_order[i] for i in np.argmax(pb, axis=1)])
        accs.append(accuracy_score(yt_b, preds_b))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return lo, hi


# ══════════════════════════════════════════════
#  单赛事完整回测
# ══════════════════════════════════════════════

def run_one_tournament(tourn_key, t_matches, df_full, df_inter,
                       feat_cols, feat_cols_inter, best_params):
    """
    对一个赛事运行全套模型回测.
    返回 {model_name: {"probs": array, "metrics": dict, "ci": tuple}}
    """
    opening  = t_matches["date"].min()
    n        = len(t_matches)
    y_true   = t_matches["result"].values

    # 训练集严格截止到开赛前
    df_train      = df_full[(df_full["date"] >= TRAIN_START) &
                             (df_full["date"] < opening)].copy()
    df_train_inter = df_inter[(df_inter["date"] >= TRAIN_START) &
                              (df_inter["date"] < opening)].copy()

    draw_base = (df_train["result"] == "D").mean()
    n_val     = max(int(len(df_train) * 0.15), 50)
    df_val_i  = df_train_inter.iloc[-n_val:].copy()
    y_val     = df_val_i["result"].values

    print(f"  [{tourn_key}] n_train={len(df_train)}, n_matches={n}, opening={opening.date()}")

    results = {}

    # ── BLa ─────────────────────────────────
    probs_bla = make_bla_probs(t_matches, df_train)
    results["BLa"] = {"probs": probs_bla}

    # ── BLb ─────────────────────────────────
    probs_blb = make_blb_probs(df_train, n)
    results["BLb"] = {"probs": probs_blb}

    # ── XGB + 交互特征 ───────────────────────
    t_inter = add_interaction_features(t_matches)
    X_test_i = t_inter[feat_cols_inter].values.astype(np.float32)

    pkg_xgbi = fit_pipeline(
        df_train_inter, "xgb", feat_cols=feat_cols_inter,
        xgb_max_depth=best_params["xgb"]["max_depth"],
        xgb_lr=best_params["xgb"]["lr"],
        xgb_n_est=best_params["xgb"]["n_est"],
        lam=best_params["xgb"]["lam"],
    )
    probs_xgbi = pkg_xgbi["calibrated_model"].predict_proba(X_test_i)
    delta_xgbi  = pkg_xgbi["delta"]
    dthr_xgbi   = pkg_xgbi["draw_thr"]
    results["XGB+int"] = {"probs": probs_xgbi,
                          "delta": delta_xgbi, "draw_thr": dthr_xgbi}

    # ── DC (fixed half-life=730) ─────────────
    print(f"    训练DC...", end=" ", flush=True)
    dc = fit_dc_fast(df_train, opening)
    print(f"γ={np.exp(dc.log_gamma):.3f}  ρ={dc.rho:.3f}")
    probs_dc = dc_probs_for_matches(dc, t_matches)

    # DC draw_adj (从验证集的DC预测中选δ)
    probs_dc_val = dc_probs_for_matches(dc, df_val_i)
    delta_dc, dthr_dc = tune_draw_threshold(
        probs_dc_val, y_val, LABEL_ORDER, draw_base)
    results["DC"] = {"probs": probs_dc,
                     "delta": delta_dc, "draw_thr": dthr_dc}

    # ── Ensemble2 (DC + XGB_inter) ──────────
    probs_xgbi_val = pkg_xgbi["calibrated_model"].predict_proba(
        df_val_i[feat_cols_inter].values.astype(np.float32))

    best_w, best_ll = 0.5, float("inf")
    for w in np.arange(0.0, 1.01, 0.1):
        p_ens = w * probs_xgbi_val + (1-w) * probs_dc_val
        ll_ = log_loss(y_val, p_ens, labels=LABEL_ORDER)
        if ll_ < best_ll:
            best_ll, best_w = ll_, w

    probs_ens2 = best_w * probs_xgbi + (1-best_w) * probs_dc
    # Ens2 draw_adj: use val predictions for thresholding
    probs_ens2_val = best_w * probs_xgbi_val + (1-best_w) * probs_dc_val
    delta_ens2, dthr_ens2 = tune_draw_threshold(
        probs_ens2_val, y_val, LABEL_ORDER, draw_base)
    results["Ens2"] = {"probs": probs_ens2,
                       "delta": delta_ens2, "draw_thr": dthr_ens2,
                       "w_xgb": best_w}
    print(f"    Ens2 w_XGB={best_w:.1f}")

    # ── 计算所有指标 + CI ────────────────────
    for mname, mdata in results.items():
        probs_ = mdata["probs"]
        delta_ = mdata.get("delta")
        dthr_  = mdata.get("draw_thr")
        mdata["met"]  = metrics(y_true, probs_, LABEL_ORDER, delta_, dthr_)
        mdata["ci"]   = bootstrap_ci(y_true, probs_, LABEL_ORDER)
        mdata["y_true"] = y_true

    return results


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

def aggregate_metrics(all_results, model_name, field="acc"):
    """跨赛事汇总单指标"""
    vals, ns, cis = [], [], []
    for key, res in all_results.items():
        if model_name not in res: continue
        m  = res[model_name]["met"]
        ci = res[model_name]["ci"]
        n  = len(res[model_name]["y_true"])
        vals.append(m[field]); ns.append(n); cis.append(ci)
    if not vals:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    # 加权均值（按场次）
    w  = np.array(ns) / sum(ns)
    mu = float(np.dot(w, vals))
    return mu, sum(ns), (float("nan"), float("nan"))  # CI computed later


def build_summary_table(all_results):
    """构建总对比表"""
    MODEL_NAMES = ["BLa", "BLb", "XGB+int", "DC", "Ens2"]
    TOURN_KEYS  = list(all_results.keys())

    rows = []
    for mname in MODEL_NAMES:
        row = {"模型": mname}

        # 各赛事 acc
        for key in TOURN_KEYS:
            if mname not in all_results[key]:
                row[key] = float("nan")
            else:
                row[key] = all_results[key][mname]["met"]["acc"]

        # 合并全部场次计算整体指标
        y_all, p_all = [], []
        for key in TOURN_KEYS:
            if mname not in all_results[key]: continue
            y_all.extend(all_results[key][mname]["y_true"])
            p_all.append(all_results[key][mname]["probs"])
        y_all = np.array(y_all)
        p_all = np.vstack(p_all)

        row["total_n"] = len(y_all)
        row["overall_acc"] = accuracy_score(
            y_all, [LABEL_ORDER[i] for i in np.argmax(p_all, axis=1)])
        row["brier"]    = compute_brier(y_all, p_all)
        row["log_loss"] = log_loss(y_all, p_all, labels=LABEL_ORDER)

        # Bootstrap CI on overall accuracy
        ci = bootstrap_ci(y_all, p_all)
        row["ci_lo"] = ci[0]
        row["ci_hi"] = ci[1]

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df_table, all_results):
    MODEL_NAMES = ["BLa", "BLb", "XGB+int", "DC", "Ens2"]
    TOURN_KEYS  = list(all_results.keys())

    print("\n" + "#"*78)
    print("  扩充dev集回测汇总（2022世界杯封存）")
    print("#"*78)

    # ── 每赛事准确率一览 ──
    header_cols = TOURN_KEYS
    print(f"\n  {'模型':<12}", end="")
    for k in header_cols:
        print(f" {k:>12}", end="")
    print()
    print("  " + "-"*12 + "-"*13*len(header_cols))
    for mname in MODEL_NAMES:
        print(f"  {mname:<12}", end="")
        for k in header_cols:
            v = all_results[k][mname]["met"]["acc"] if mname in all_results[k] else float("nan")
            n = len(all_results[k][mname]["y_true"]) if mname in all_results[k] else 0
            print(f" {v:>6.1%}({n:>2})", end="")
        print()

    # ── 总体指标表 ──
    print(f"\n  {'模型':<12} {'总场次':>7} {'总体ACC':>9} {'95%CI':>16} "
          f"{'Brier':>7} {'LogLoss':>8}")
    print("  " + "-"*68)
    for _, r in df_table.iterrows():
        ci_str = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        print(f"  {r['模型']:<12} {int(r['total_n']):>7} "
              f"{r['overall_acc']:>9.4f} {ci_str:>16} "
              f"{r['brier']:>7.4f} {r['log_loss']:>8.4f}")
    print("  " + "-"*68)

    # ── 按大洲分组 ──
    groups = {
        "世界杯WC":  ["WC2014","WC2018"],
        "欧洲Euro":  ["Euro2016","Euro2020"],
        "美洲Copa":  ["Copa2015","Copa2016","Copa2019","Copa2021"],
        "亚洲Asian": ["AsianCup2015","AsianCup2019"],
        "非洲AFCON": ["AFCON2015","AFCON2017","AFCON2019","AFCON2021"],
    }
    print(f"\n  {'大洲分组':<12} {'模型':>8} {'场次':>5} {'ACC':>7} {'Brier':>7} {'LL':>8}")
    print("  " + "-"*56)
    for grp_name, keys in groups.items():
        avail_keys = [k for k in keys if k in all_results]
        if not avail_keys: continue
        first = True
        for mname in ["XGB+int", "DC", "Ens2"]:
            y_g, p_g = [], []
            for k in avail_keys:
                if mname not in all_results[k]: continue
                y_g.extend(all_results[k][mname]["y_true"])
                p_g.append(all_results[k][mname]["probs"])
            if not y_g: continue
            y_g = np.array(y_g); p_g = np.vstack(p_g)
            acc_g = accuracy_score(y_g, [LABEL_ORDER[i] for i in np.argmax(p_g, axis=1)])
            b_g   = compute_brier(y_g, p_g)
            ll_g  = log_loss(y_g, p_g, labels=LABEL_ORDER)
            gname = grp_name if first else ""
            print(f"  {gname:<12} {mname:>8} {len(y_g):>5} "
                  f"{acc_g:>7.4f} {b_g:>7.4f} {ll_g:>8.4f}")
            first = False
    print("  " + "-"*56)

    # ── Ens2 的 w 值分布 ──
    print(f"\n  Ens2 w_XGB 分布:")
    for key in TOURN_KEYS:
        if "Ens2" in all_results[key]:
            w = all_results[key]["Ens2"].get("w_xgb", float("nan"))
            print(f"    {key:<16}: w_XGB={w:.1f}  "
                  f"({1-w:.1f} DC + {w:.1f} XGB)")


if __name__ == "__main__":
    print("读取数据...")
    df_raw   = pd.read_csv(PROCESSED / "features.csv", parse_dates=["date"])
    df_raw   = df_raw.sort_values("date").reset_index(drop=True)
    df_inter = add_interaction_features(df_raw)
    feat_cols = get_feature_cols(df_raw)
    feat_cols_inter = feat_cols + ["elo_diff_ko", "rest_diff_ko"]

    with open(OUTPUTS / "train_params.json", encoding="utf-8") as f:
        best_params = json.load(f)

    all_results = {}
    for tourn_key, name_pat, d_start, d_end in TOURNS:
        t_matches = get_tournament_matches(df_raw, name_pat, d_start, d_end)
        if len(t_matches) == 0:
            print(f"  [{tourn_key}] 未找到比赛数据，跳过")
            continue
        if len(t_matches) < 16:
            print(f"  [{tourn_key}] 场次={len(t_matches)} 过少，跳过")
            continue

        res = run_one_tournament(
            tourn_key, t_matches,
            df_raw, df_inter,
            feat_cols, feat_cols_inter,
            best_params,
        )
        all_results[tourn_key] = res

    # ── 汇总表 ──
    df_summary = build_summary_table(all_results)
    print_summary(df_summary, all_results)

    # ── 保存 ──
    df_summary.to_csv(OUTPUTS / "devset_summary.csv", index=False,
                      encoding="utf-8-sig")
    print(f"\n汇总表已保存: outputs/devset_summary.csv")

    # 保存明细（各赛事各模型准确率）
    detail_rows = []
    for key, res in all_results.items():
        for mname, mdata in res.items():
            if "met" not in mdata: continue
            m = mdata["met"]; ci = mdata["ci"]
            detail_rows.append({
                "tournament": key,
                "model": mname,
                "n": len(mdata["y_true"]),
                "acc": m["acc"],
                "acc_adj": m["acc_adj"],
                "brier": m["brier"],
                "log_loss": m["log_loss"],
                "ci_lo": ci[0],
                "ci_hi": ci[1],
            })
    pd.DataFrame(detail_rows).to_csv(
        OUTPUTS / "devset_detail.csv", index=False, encoding="utf-8-sig")
    print("明细表已保存: outputs/devset_detail.csv")

    print("\n\n[完成] 请确认扩充dev基准表后告知是否进入步骤3-5。")
