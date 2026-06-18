"""
模型改进实验 (step5b_improve.py)
严格纪律：仅使用2014/2018回测数据，2022封存。

Task 1: Bootstrap + McNemar 显著性检验
Task 2: Dixon-Coles 双泊松模型
Task 3: XGB + DC 集成
Task 4: 友谊赛降权实验
Task 5: 交互特征
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from step4_train import (
    TRAIN_START, get_feature_cols, predict_with_draw_adj,
    fit_pipeline, tune_draw_threshold, XGBIsotonicCalibrated,
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"

LABEL_ORDER = ["A", "D", "H"]   # 与 fit_pipeline 保持一致

WC_OPENING = {
    2014: pd.Timestamp("2014-06-12"),
    2018: pd.Timestamp("2018-06-14"),
    # 2022: SEALED — 不在本文件中使用
}
EVAL_YEARS = [2014, 2018]


# ══════════════════════════════════════════════
#  通用评估工具
# ══════════════════════════════════════════════

def metrics_from_probs(y_true, probs, label_order=LABEL_ORDER,
                       delta=None, draw_thr=None, tag=""):
    """返回 argmax 和 draw_adj 双口径的指标字典"""
    preds_raw = np.array([label_order[i] for i in np.argmax(probs, axis=1)])
    if delta is not None:
        preds_adj = predict_with_draw_adj(probs, delta, draw_thr, label_order)
    else:
        preds_adj = preds_raw

    def _m(preds):
        acc  = accuracy_score(y_true, preds)
        mf1  = f1_score(y_true, preds, average="macro", zero_division=0)
        yb   = label_binarize(y_true, classes=label_order)
        brier = float(np.mean([brier_score_loss(yb[:,i], probs[:,i])
                                for i in range(len(label_order))]))
        ll   = log_loss(y_true, probs, labels=label_order)
        return dict(accuracy=acc, macro_f1=mf1, brier=brier, log_loss=ll)

    raw = _m(preds_raw)
    adj = _m(preds_adj)
    return raw, adj


def get_wc_data(df: pd.DataFrame, year: int):
    """获取该届WC比赛数据（特征+元数据）"""
    return (df[(df["tournament"] == "FIFA World Cup") &
               (df["date"].dt.year == year)]
            .sort_values("date").copy())


def run_xgb_backtest(year, df, feat_cols, lr_params, xgb_params,
                     extra_weights_fn=None, extra_feat_cols=None):
    """
    训练XGB（可带额外sample weight和额外特征），返回 (probs, y_true, pkg)
    extra_weights_fn: df_train -> np.array 乘在时间衰减权重上
    extra_feat_cols: list of new feature column names in df
    """
    opening = WC_OPENING[year]
    df_train = df[(df["date"] >= TRAIN_START) & (df["date"] < opening)].copy()
    df_wc    = get_wc_data(df, year)

    all_feat  = (feat_cols if extra_feat_cols is None
                 else feat_cols + extra_feat_cols)
    draw_base = (df_train["result"] == "D").mean()

    # 计算额外权重（乘法）
    extra_w = None
    if extra_weights_fn is not None:
        extra_w = extra_weights_fn(df_train)

    # 使用 fit_pipeline，通过注入额外权重临时 patch
    if extra_w is not None:
        pkg = _fit_pipeline_extra_w(df_train, xgb_params, extra_w, all_feat, draw_base)
    else:
        pkg = fit_pipeline(
            df_train, "xgb", feat_cols=all_feat,
            xgb_max_depth=xgb_params["max_depth"],
            xgb_lr=xgb_params["lr"],
            xgb_n_est=xgb_params["n_est"],
            lam=xgb_params["lam"],
        )

    X_wc  = df_wc[all_feat].values.astype(np.float32)
    probs = pkg["calibrated_model"].predict_proba(X_wc)
    y_true = df_wc["result"].values
    return probs, y_true, pkg


def _fit_pipeline_extra_w(df_train, xgb_params, extra_w, feat_cols, draw_base):
    """内部：带额外乘法权重的 XGB 训练（不修改 step4_train.py）"""
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.isotonic import IsotonicRegression
    from step4_train import time_decay_weights, tune_draw_threshold

    df_train = df_train.sort_values("date").reset_index(drop=True)
    n = len(df_train); n_cal = max(int(n * 0.15), 50)
    df_fit = df_train.iloc[:n - n_cal].copy()
    df_cal = df_train.iloc[n - n_cal:].copy()

    X_fit = df_fit[feat_cols].values.astype(np.float32)
    y_fit = df_fit["result"].values
    X_cal = df_cal[feat_cols].values.astype(np.float32)
    y_cal = df_cal["result"].values

    ref   = df_fit["date"].max()
    lam   = xgb_params["lam"]
    w_td  = time_decay_weights(df_fit["date"], ref, lam)
    w_ex  = extra_w[:len(df_fit)]
    w_fit = w_td * w_ex
    w_fit = w_fit / w_fit.max()   # normalize to prevent extreme scale

    label_order = sorted(np.unique(df_train["result"]))
    le = LabelEncoder().fit(label_order)
    y_fit_enc = le.transform(y_fit)

    base = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(label_order),
        max_depth=xgb_params["max_depth"], learning_rate=xgb_params["lr"],
        n_estimators=xgb_params["n_est"], subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, eval_metric="mlogloss",
    )
    base.fit(X_fit, y_fit_enc, sample_weight=w_fit)

    raw_cal  = base.predict_proba(X_cal)
    iso_regs = []
    for i, cls in enumerate(label_order):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_cal[:, i], (y_cal == cls).astype(float))
        iso_regs.append(iso)

    cal_model = XGBIsotonicCalibrated(base, iso_regs, label_order)
    probs_cal = cal_model.predict_proba(X_cal)
    delta, draw_thr = tune_draw_threshold(probs_cal, y_cal, label_order, draw_base)

    return dict(calibrated_model=cal_model, feat_cols=feat_cols,
                label_order=label_order, delta=delta, draw_thr=draw_thr,
                lam=lam, model_type="xgb")


# ══════════════════════════════════════════════
#  Task 1: Bootstrap + McNemar 显著性检验
# ══════════════════════════════════════════════

def task1_significance(verbose=True):
    print("\n" + "="*60)
    print("  Task 1: 显著性检验（2014+2018 合并，共128场）")
    print("="*60)

    rows = []
    for yr in EVAL_YEARS:
        df = pd.read_csv(OUTPUTS_DIR / f"backtest_{yr}.csv")
        rows.append(df)
    df_all = pd.concat(rows, ignore_index=True)
    n = len(df_all)

    xgb_ok  = (df_all["xgb_argmax"] == df_all["result"]).values.astype(int)
    bla_ok  = (df_all["bla_pred"]   == df_all["result"]).values.astype(int)

    # ── Bootstrap (10000次) ───────────────────
    rng   = np.random.default_rng(42)
    diffs = []
    for _ in range(10_000):
        idx  = rng.integers(0, n, size=n)
        diffs.append(xgb_ok[idx].mean() - bla_ok[idx].mean())
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    obs_diff = xgb_ok.mean() - bla_ok.mean()

    print(f"\n  XGB_argmax acc  = {xgb_ok.mean():.4f}")
    print(f"  BLa(Elo高者胜) acc = {bla_ok.mean():.4f}")
    print(f"  观测差值 = {obs_diff:+.4f}")
    print(f"  Bootstrap 95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    p_val_bs = min(np.mean(diffs >= 0), np.mean(diffs <= 0)) * 2
    print(f"  Bootstrap p-value (双侧) ≈ {p_val_bs:.3f}")

    # ── McNemar 检验 ──────────────────────────
    b = np.sum((xgb_ok == 1) & (bla_ok == 0))  # XGB对，BLa错
    c = np.sum((xgb_ok == 0) & (bla_ok == 1))  # XGB错，BLa对
    if (b + c) > 0:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_mc = 1 - chi2_dist.cdf(stat, df=1)
    else:
        stat, p_mc = 0, 1.0

    print(f"\n  McNemar chi2={stat:.3f}  p={p_mc:.3f}")
    print(f"  b(XGB对BLa错)={b}  c(XGB错BLa对)={c}")
    print(f"  结论: {'差异' if p_mc < 0.05 else '差异不'}显著 (alpha=0.05)")

    # ── 意见不一致场次明细 ──────────────────────
    df_all["xgb_ok"]  = xgb_ok
    df_all["bla_ok"]  = bla_ok
    disagree = df_all[df_all["xgb_argmax"] != df_all["bla_pred"]].copy()
    print(f"\n  意见不一致场次: {len(disagree)}/{n}")
    print(f"  按真实结果分类:")

    stats = []
    for res in ["H", "D", "A"]:
        sub = disagree[disagree["result"] == res]
        if len(sub) == 0: continue
        xgb_win = sub["xgb_ok"].sum()
        bla_win = sub["bla_ok"].sum()
        neither = len(sub) - xgb_win - bla_win
        stats.append((res, len(sub), xgb_win, bla_win, neither))
        print(f"    真实={res}: {len(sub)}场  XGB对={xgb_win}  BLa对={bla_win}  "
              f"两者都错={neither}")

    draw_dis = disagree[disagree["result"] == "D"]
    total_draw_err = (df_all["result"] == "D").sum()
    pct = len(draw_dis) / max(1, total_draw_err) * 100
    print(f"\n  平局场次中意见不一致: {len(draw_dis)}/{int(total_draw_err)} ({pct:.1f}%)")
    print(f"  → 差距{'几乎全部' if pct > 70 else '部分'}来自平局场次")

    return {"obs_diff": obs_diff, "ci": (ci_lo, ci_hi), "p_mc": p_mc}


# ══════════════════════════════════════════════
#  Task 2: Dixon-Coles 双泊松模型
# ══════════════════════════════════════════════

class DixonColesModel:
    """Dixon-Coles (1997) 双泊松模型，含低比分修正ρ和时间衰减"""

    def __init__(self):
        self.log_a  = {}   # team → log(attack)
        self.log_d  = {}   # team → log(defense)
        self.log_gamma = 0.2  # log(home advantage)
        self.rho    = -0.1
        self.avg_log_a = 0.0
        self.avg_log_d = 0.0
        self.half_life = 365

    def fit(self, df_matches: pd.DataFrame, ref_date, half_life_days: int = 365):
        """加权最大似然估计"""
        self.half_life = half_life_days
        decay  = np.log(2) / half_life_days

        df = (df_matches
              .dropna(subset=["home_score", "away_score"])
              .copy())
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        days    = (pd.Timestamp(ref_date) - df["date"]).dt.days.clip(lower=0).values
        weights = np.exp(-decay * days)

        all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        n_t  = len(all_teams)
        tidx = {t: i for i, t in enumerate(all_teams)}

        ht = df["home_team"].map(tidx).values
        at = df["away_team"].map(tidx).values
        hg = df["home_score"].values
        ag = df["away_score"].values
        neu = df["neutral"].fillna(0).values.astype(float)

        # 预计算掩码（固定）
        m00 = (hg == 0) & (ag == 0)
        m10 = (hg == 1) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m11 = (hg == 1) & (ag == 1)
        is_hv = (neu == 0)  # home venue

        def nll_and_grad(params):
            la   = params[:n_t]
            ld   = params[n_t:2*n_t]
            lg   = params[2*n_t]
            rho_ = params[2*n_t + 1]

            a_   = np.exp(np.clip(la, -4, 4))
            d_   = np.exp(np.clip(ld, -4, 4))
            g_   = float(np.exp(np.clip(lg, -0.7, 0.7)))

            lam_ = a_[ht] * d_[at] * np.where(is_hv, g_, 1.0)
            mu_  = a_[at] * d_[ht]

            tau  = np.ones(len(hg))
            tau[m00] = 1 - lam_[m00] * mu_[m00] * rho_
            tau[m10] = 1 + mu_[m10] * rho_
            tau[m01] = 1 + lam_[m01] * rho_
            tau[m11] = 1 - rho_
            tau = np.maximum(tau, 1e-8)

            log_p = (np.log(tau) +
                     hg * np.log(np.maximum(lam_, 1e-8)) - lam_ - gammaln(hg + 1) +
                     ag * np.log(np.maximum(mu_,  1e-8)) - mu_  - gammaln(ag + 1))

            reg  = 0.01 * (np.sum(la**2) + np.sum(ld**2))
            nll_val = -np.dot(weights, log_p) + reg

            # ── 解析梯度 ──────────────────────────
            # d(log τ)/d(λ)*λ and d(log τ)/d(μ)*μ
            dtau_dloglam = np.zeros(len(hg))
            dtau_dloglam[m00] = -lam_[m00] * mu_[m00] * rho_ / tau[m00]
            dtau_dloglam[m01] =  lam_[m01] * rho_ / tau[m01]

            dtau_dlogmu = np.zeros(len(hg))
            dtau_dlogmu[m00] = -lam_[m00] * mu_[m00] * rho_ / tau[m00]
            dtau_dlogmu[m10] =  mu_[m10] * rho_ / tau[m10]

            # d(log_p)/d(log λ) = dtau_dloglam + (x - λ)
            dphi_dloglam = dtau_dloglam + (hg - lam_)
            dphi_dlogmu  = dtau_dlogmu  + (ag - mu_)

            w_phi_lam = weights * dphi_dloglam
            w_phi_mu  = weights * dphi_dlogmu

            g_la = np.zeros(n_t)
            np.add.at(g_la, ht, w_phi_lam)   # team as home: enters λ
            np.add.at(g_la, at, w_phi_mu)    # team as away: enters μ
            g_la = -g_la + 2 * 0.01 * la

            g_ld = np.zeros(n_t)
            np.add.at(g_ld, at, w_phi_lam)   # away defense enters λ
            np.add.at(g_ld, ht, w_phi_mu)    # home defense enters μ
            g_ld = -g_ld + 2 * 0.01 * ld

            g_lg  = float(-np.sum(weights * is_hv * dphi_dloglam))

            dtau_drho = np.zeros(len(hg))
            dtau_drho[m00] = -lam_[m00] * mu_[m00] / tau[m00]
            dtau_drho[m10] =  mu_[m10] / tau[m10]
            dtau_drho[m01] =  lam_[m01] / tau[m01]
            dtau_drho[m11] = -1.0 / tau[m11]
            g_rho = float(-np.sum(weights * dtau_drho))

            grad = np.concatenate([g_la, g_ld, [g_lg, g_rho]])
            return nll_val, grad

        x0     = np.zeros(2*n_t + 2)
        x0[2*n_t]   = 0.2
        x0[2*n_t+1] = -0.1
        bounds = [(-4, 4)] * (2*n_t) + [(-0.7, 0.7), (-0.4, 0.4)]

        res = minimize(nll_and_grad, x0, method="L-BFGS-B",
                       jac=True, bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-6})

        params_opt = res.x
        la_opt = params_opt[:n_t]
        ld_opt = params_opt[n_t:2*n_t]

        self.log_a = {t: la_opt[i] for i, t in enumerate(all_teams)}
        self.log_d = {t: ld_opt[i] for i, t in enumerate(all_teams)}
        self.log_gamma = params_opt[2*n_t]
        self.rho       = params_opt[2*n_t + 1]
        self.avg_log_a = float(np.mean(la_opt))
        self.avg_log_d = float(np.mean(ld_opt))
        return self

    def predict_proba(self, home: str, away: str,
                      neutral: bool = False, max_goals: int = 10):
        """返回 (P(H), P(D), P(A))"""
        la_h = self.log_a.get(home, self.avg_log_a)
        ld_h = self.log_d.get(home, self.avg_log_d)
        la_a = self.log_a.get(away, self.avg_log_a)
        ld_a = self.log_d.get(away, self.avg_log_d)

        a_h = np.exp(la_h); d_h = np.exp(ld_h)
        a_a = np.exp(la_a); d_a = np.exp(ld_a)
        gamma = np.exp(self.log_gamma)
        rho   = self.rho

        lam = a_h * d_a * (gamma if not neutral else 1.0)
        mu  = a_a * d_h

        X, Y = np.mgrid[0:max_goals+1, 0:max_goals+1]
        from scipy.stats import poisson
        probs = poisson.pmf(X, lam) * poisson.pmf(Y, mu)

        probs[0, 0] *= max(1 - lam * mu * rho, 0)
        probs[1, 0] *= max(1 + mu * rho, 0)
        probs[0, 1] *= max(1 + lam * rho, 0)
        probs[1, 1] *= max(1 - rho, 0)

        probs = np.maximum(probs, 0)
        total = probs.sum()
        probs /= total if total > 0 else 1

        p_h = float(probs[X > Y].sum())
        p_d = float(probs[X == Y].sum())
        p_a = float(probs[X < Y].sum())
        return p_h, p_d, p_a

    def val_log_loss(self, df_val: pd.DataFrame) -> float:
        """在验证集上计算LogLoss"""
        probs = []
        for _, r in df_val.iterrows():
            ph, pd_, pa = self.predict_proba(r["home_team"], r["away_team"],
                                              bool(r["neutral"]))
            probs.append([pa, pd_, ph])  # label_order = [A,D,H]
        return log_loss(df_val["result"].values, np.array(probs), labels=LABEL_ORDER)


def fit_dc_model(df_train: pd.DataFrame, opening_date,
                 half_lives=(90, 180, 365, 730), val_ratio=0.15):
    """选最佳 half-life，最终在全训练数据上训练 DC 模型"""
    n     = len(df_train)
    n_val = max(int(n * val_ratio), 50)
    df_f  = df_train.iloc[:n - n_val].copy()
    df_v  = df_train.iloc[n - n_val:].copy()

    best_hl, best_ll = 365, float("inf")
    for hl in half_lives:
        dc = DixonColesModel()
        dc.fit(df_f, opening_date, half_life_days=hl)
        ll = dc.val_log_loss(df_v)
        if ll < best_ll:
            best_ll, best_hl = ll, hl

    dc_final = DixonColesModel()
    dc_final.fit(df_train, opening_date, half_life_days=best_hl)
    return dc_final, best_hl


def dc_probs_for_wc(dc_model: DixonColesModel,
                    df_wc: pd.DataFrame) -> np.ndarray:
    """返回 (N, 3) 概率矩阵，列顺序 = LABEL_ORDER=['A','D','H']"""
    out = []
    for _, r in df_wc.iterrows():
        ph, pd_, pa = dc_model.predict_proba(
            r["home_team"], r["away_team"], bool(r.get("neutral", 0)))
        out.append([pa, pd_, ph])   # [A, D, H] order
    return np.array(out)


def task2_dc(df, best_params, verbose=True):
    print("\n" + "="*60)
    print("  Task 2: Dixon-Coles 模型")
    print("="*60)

    results = {}
    for year in EVAL_YEARS:
        opening  = WC_OPENING[year]
        df_train = df[(df["date"] >= TRAIN_START) & (df["date"] < opening)].copy()
        df_wc    = get_wc_data(df, year)
        y_true   = df_wc["result"].values

        print(f"\n  {year}: 训练DC模型 ({len(df_train)} 场)...")
        dc, best_hl = fit_dc_model(df_train, opening)
        print(f"    最佳半衰期: {best_hl} 天  "
              f"γ={np.exp(dc.log_gamma):.3f}  ρ={dc.rho:.3f}")

        probs = dc_probs_for_wc(dc, df_wc)

        # 在校准集上调平局阈值
        draw_base = (df_train["result"] == "D").mean()
        n_val     = max(int(len(df_train) * 0.15), 50)
        df_val    = df_train.iloc[-n_val:]
        probs_val = dc_probs_for_wc(dc, df_val)
        delta, draw_thr = tune_draw_threshold(
            probs_val, df_val["result"].values, LABEL_ORDER, draw_base)

        raw, adj = metrics_from_probs(y_true, probs, LABEL_ORDER,
                                       delta, draw_thr)
        results[year] = {"raw": raw, "adj": adj, "probs": probs,
                         "y_true": y_true, "delta": delta, "draw_thr": draw_thr,
                         "dc_model": dc}
        print(f"    ACC(raw)={raw['accuracy']:.4f}  ACC(adj)={adj['accuracy']:.4f}  "
              f"Brier={raw['brier']:.4f}  LL={raw['log_loss']:.4f}")
    return results


# ══════════════════════════════════════════════
#  Task 3: 集成模型
# ══════════════════════════════════════════════

def task3_ensemble(df, feat_cols, best_params, dc_results, verbose=True):
    print("\n" + "="*60)
    print("  Task 3: 集成模型 P_ens = w*XGB + (1-w)*DC")
    print("="*60)

    results = {}
    for year in EVAL_YEARS:
        opening  = WC_OPENING[year]
        df_train = df[(df["date"] >= TRAIN_START) & (df["date"] < opening)].copy()
        df_wc    = get_wc_data(df, year)
        y_true   = df_wc["result"].values

        # 训练 XGB
        pkg_xgb = fit_pipeline(
            df_train, "xgb", feat_cols=feat_cols,
            xgb_max_depth=best_params["xgb"]["max_depth"],
            xgb_lr=best_params["xgb"]["lr"],
            xgb_n_est=best_params["xgb"]["n_est"],
            lam=best_params["xgb"]["lam"],
        )
        X_wc       = df_wc[feat_cols].values.astype(np.float32)
        probs_xgb  = pkg_xgb["calibrated_model"].predict_proba(X_wc)

        # DC 预测（已在 task2 训练好）
        dc_model  = dc_results[year]["dc_model"]
        probs_dc  = dc_probs_for_wc(dc_model, df_wc)

        # 在校准集上选 w（最小化 LogLoss）
        n_val     = max(int(len(df_train) * 0.15), 50)
        df_val    = df_train.iloc[-n_val:]
        X_val     = df_val[feat_cols].values.astype(np.float32)
        y_val     = df_val["result"].values

        probs_xgb_val = pkg_xgb["calibrated_model"].predict_proba(X_val)
        probs_dc_val  = dc_probs_for_wc(dc_model, df_val)

        best_w, best_ll_val = 0.5, float("inf")
        for w in np.arange(0.0, 1.01, 0.1):
            p_ens_val = w * probs_xgb_val + (1 - w) * probs_dc_val
            ll_val = log_loss(y_val, p_ens_val, labels=LABEL_ORDER)
            if ll_val < best_ll_val:
                best_ll_val, best_w = ll_val, w

        print(f"\n  {year}: 最佳 w(XGB)={best_w:.1f}  "
              f"(val LogLoss={best_ll_val:.4f})")

        probs_ens = best_w * probs_xgb + (1 - best_w) * probs_dc

        draw_base = (df_train["result"] == "D").mean()
        p_ens_val_best = best_w * probs_xgb_val + (1 - best_w) * probs_dc_val
        delta, draw_thr = tune_draw_threshold(
            p_ens_val_best, y_val, LABEL_ORDER, draw_base)

        raw, adj = metrics_from_probs(y_true, probs_ens, LABEL_ORDER,
                                       delta, draw_thr)
        results[year] = {"raw": raw, "adj": adj, "w": best_w}
        print(f"    ACC(raw)={raw['accuracy']:.4f}  ACC(adj)={adj['accuracy']:.4f}  "
              f"Brier={raw['brier']:.4f}  LL={raw['log_loss']:.4f}")
    return results


# ══════════════════════════════════════════════
#  Task 4: 友谊赛降权
# ══════════════════════════════════════════════

def task4_friendly_weight(df, feat_cols, best_params, verbose=True):
    print("\n" + "="*60)
    print("  Task 4: 友谊赛降权 (w_f ∈ {0.2, 0.3, 0.5, 1.0})")
    print("="*60)

    TOURNAMENT_W = {
        "is_world_cup": 1.0,
        "continental": 0.9,   # 非WC、非预选赛、非友谊赛
        "is_qualifier": 0.8,
        "is_friendly": None,  # grid search
    }
    W_F_GRID = [0.2, 0.3, 0.5, 1.0]

    def make_w_fn(w_f):
        def fn(df_train):
            w = np.ones(len(df_train))
            wc_mask  = df_train["is_world_cup"].astype(bool).values
            q_mask   = df_train["is_qualifier"].astype(bool).values
            fr_mask  = df_train["is_friendly"].astype(bool).values
            con_mask = (~wc_mask) & (~q_mask) & (~fr_mask)
            w[wc_mask]  = 1.0
            w[con_mask] = 0.9
            w[q_mask]   = 0.8
            w[fr_mask]  = w_f
            return w
        return fn

    # 在2014训练期内选 w_f（用校准集 LogLoss）
    opening_2014 = WC_OPENING[2014]
    df_tr_2014   = df[(df["date"] >= TRAIN_START) & (df["date"] < opening_2014)].copy()
    n_val        = max(int(len(df_tr_2014) * 0.15), 50)
    df_val_2014  = df_tr_2014.iloc[-n_val:]
    y_val_2014   = df_val_2014["result"].values

    best_wf, best_ll = 0.5, float("inf")
    for wf in W_F_GRID:
        w_arr  = make_w_fn(wf)(df_tr_2014)
        pkg_tmp = _fit_pipeline_extra_w(
            df_tr_2014, best_params["xgb"], w_arr, feat_cols,
            (df_tr_2014["result"] == "D").mean())
        X_v = df_val_2014[feat_cols].values.astype(np.float32)
        p_v = pkg_tmp["calibrated_model"].predict_proba(X_v)
        ll  = log_loss(y_val_2014, p_v, labels=LABEL_ORDER)
        print(f"    w_f={wf}: val_LL={ll:.4f}")
        if ll < best_ll:
            best_ll, best_wf = ll, wf

    print(f"  最佳 w_f={best_wf} (val LL={best_ll:.4f})")

    results = {}
    for year in EVAL_YEARS:
        df_wc  = get_wc_data(df, year)
        y_true = df_wc["result"].values
        probs, _, pkg = run_xgb_backtest(
            year, df, feat_cols, best_params["lr"], best_params["xgb"],
            extra_weights_fn=make_w_fn(best_wf))
        raw, adj = metrics_from_probs(y_true, probs, LABEL_ORDER,
                                       pkg["delta"], pkg["draw_thr"])
        results[year] = {"raw": raw, "adj": adj}
        print(f"  {year}: ACC(raw)={raw['accuracy']:.4f}  "
              f"ACC(adj)={adj['accuracy']:.4f}  Brier={raw['brier']:.4f}")
    return results, best_wf


# ══════════════════════════════════════════════
#  Task 5: 交互特征
# ══════════════════════════════════════════════

def task5_interaction(df, feat_cols, best_params, verbose=True):
    print("\n" + "="*60)
    print("  Task 5: 交互特征 (elo_diff×is_knockout, rest_diff×is_knockout)")
    print("="*60)

    df2 = df.copy()
    df2["elo_diff_ko"]  = df2["elo_diff"] * df2["is_knockout"]
    df2["rest_diff_ko"] = df2["rest_diff"] * df2["is_knockout"]
    new_feats = ["elo_diff_ko", "rest_diff_ko"]

    results = {}
    for year in EVAL_YEARS:
        df_wc  = get_wc_data(df2, year)
        y_true = df_wc["result"].values
        probs, _, pkg = run_xgb_backtest(
            year, df2, feat_cols, best_params["lr"], best_params["xgb"],
            extra_feat_cols=new_feats)
        raw, adj = metrics_from_probs(y_true, probs, LABEL_ORDER,
                                       pkg["delta"], pkg["draw_thr"])
        results[year] = {"raw": raw, "adj": adj}
        print(f"  {year}: ACC(raw)={raw['accuracy']:.4f}  "
              f"ACC(adj)={adj['accuracy']:.4f}  Brier={raw['brier']:.4f}")
    return results


# ══════════════════════════════════════════════
#  汇总对比表
# ══════════════════════════════════════════════

def build_comparison_table(base_results: dict, model_results: dict):
    """
    base_results: {year: {"xgb_raw": {acc,...}, "xgb_adj":..., "bla":..., "blb":...}}
    model_results: {model_name: {year: {"raw": {...}, "adj": {...}}}}
    """
    rows = []

    def avg2(d, key):
        vals = [d[yr][key] for yr in EVAL_YEARS if yr in d]
        return float(np.mean(vals)) if vals else float("nan")

    def add_row(name, m14, m18):
        if m14 is None and m18 is None: return
        acc14 = m14["accuracy"] if m14 else float("nan")
        acc18 = m18["accuracy"] if m18 else float("nan")
        avg_a = float(np.mean([v for v in [acc14, acc18] if not np.isnan(v)]))
        avg_f = float(np.mean([v for v in [m14.get("macro_f1", float("nan")),
                                            m18.get("macro_f1", float("nan"))]
                                if not np.isnan(v)]))
        avg_b = float(np.mean([v for v in [m14.get("brier", float("nan")),
                                            m18.get("brier", float("nan"))]
                                if not np.isnan(v)]))
        avg_l = float(np.mean([v for v in [m14.get("log_loss", float("nan")),
                                            m18.get("log_loss", float("nan"))]
                                if not np.isnan(v)]))
        rows.append({
            "模型": name,
            "2014": acc14, "2018": acc18,
            "均值acc": avg_a, "均值mF1": avg_f,
            "均值Brier": avg_b, "均值LL": avg_l,
        })

    # 基线
    for yr in EVAL_YEARS:
        br = base_results.get(yr, {})
    add_row("BLa Elo高者胜",
            base_results.get(2014, {}).get("bla"),
            base_results.get(2018, {}).get("bla"))
    add_row("BLb 历史频率随机",
            base_results.get(2014, {}).get("blb"),
            base_results.get(2018, {}).get("blb"))

    # 原始XGB（从step5 CSV读取，已有真实结果）
    add_row("XGB(argmax) [原始]",
            base_results.get(2014, {}).get("xgb_raw"),
            base_results.get(2018, {}).get("xgb_raw"))
    add_row("XGB(+draw_adj) [原始]",
            base_results.get(2014, {}).get("xgb_adj"),
            base_results.get(2018, {}).get("xgb_adj"))
    add_row("LR(argmax)",
            base_results.get(2014, {}).get("lr_raw"),
            base_results.get(2018, {}).get("lr_raw"))

    # 新模型
    for mname, mres in model_results.items():
        add_row(f"{mname}(raw)",
                mres.get(2014, {}).get("raw"),
                mres.get(2018, {}).get("raw"))
        add_row(f"{mname}(adj)",
                mres.get(2014, {}).get("adj"),
                mres.get(2018, {}).get("adj"))

    df_table = pd.DataFrame(rows)

    # 计算相对原始XGB的增减
    ref_acc = df_table.loc[df_table["模型"] == "XGB(argmax) [原始]", "均值acc"].values
    ref_acc = ref_acc[0] if len(ref_acc) > 0 else float("nan")

    df_table["Δacc vs XGB"] = df_table["均值acc"].apply(
        lambda x: f"{x - ref_acc:+.4f}" if not np.isnan(x) else "—")

    return df_table


def print_big_table(df_table: pd.DataFrame):
    print("\n" + "#"*80)
    print("  大对比表（行=模型，列=指标）")
    print("#"*80)
    fmt_cols = ["模型", "2014", "2018", "均值acc", "Δacc vs XGB",
                "均值mF1", "均值Brier", "均值LL"]
    hdr = (f"{'模型':<30} {'2014':>7} {'2018':>7} {'均值acc':>8} "
           f"{'Δacc':>8} {'mF1':>7} {'Brier':>7} {'LL':>8}")
    print(hdr)
    print("-"*80)
    for _, r in df_table.iterrows():
        def _f(v, fmt=".4f"):
            if isinstance(v, float) and np.isnan(v): return "  —  "
            return format(v, fmt) if isinstance(v, float) else str(v)
        print(f"{str(r['模型']):<30} {_f(r['2014']):>7} {_f(r['2018']):>7} "
              f"{_f(r['均值acc']):>8} {str(r['Δacc vs XGB']):>8} "
              f"{_f(r['均值mF1']):>7} {_f(r['均值Brier']):>7} {_f(r['均值LL']):>8}")
    print("-"*80)


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("读取数据...")
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feat_cols = get_feature_cols(df)

    with open(OUTPUTS_DIR / "train_params.json", encoding="utf-8") as f:
        best_params = json.load(f)

    # ── 从 step5 的 CSV 读取已有结果（仅2014/2018）──────
    base_results = {}
    for year in EVAL_YEARS:
        csv = pd.read_csv(OUTPUTS_DIR / f"backtest_{year}.csv")
        y   = csv["result"].values
        lo  = LABEL_ORDER

        def _mk(preds, probs_cols):
            p = csv[probs_cols].values
            acc  = accuracy_score(y, csv[preds].values)
            mf1  = f1_score(y, csv[preds].values, average="macro", zero_division=0)
            yb   = label_binarize(y, classes=lo)
            brier = float(np.mean([brier_score_loss(yb[:,i], p[:,i])
                                   for i in range(len(lo))]))
            ll   = log_loss(y, p, labels=lo)
            return dict(accuracy=acc, macro_f1=mf1, brier=brier, log_loss=ll)

        lo_csv = [f"xgb_prob_{c}" for c in lo]
        base_results[year] = {
            "xgb_raw": _mk("xgb_argmax",   lo_csv),
            "xgb_adj": _mk("xgb_draw_adj", lo_csv),
            "lr_raw":  _mk("lr_argmax",    [f"lr_prob_{c}" for c in lo]),
            "bla":     _mk("bla_pred",     [f"bla_prob_{c}" for c in lo]),
            "blb":     _mk("blb_pred",     [f"blb_prob_{c}" for c in lo]),
        }

    # ── Task 1 ────────────────────────────────
    t1_res = task1_significance()

    # ── Task 2 ────────────────────────────────
    t2_res = task2_dc(df, best_params)

    # ── Task 3 ────────────────────────────────
    t3_res = task3_ensemble(df, feat_cols, best_params, t2_res)

    # ── Task 4 ────────────────────────────────
    t4_res, best_wf = task4_friendly_weight(df, feat_cols, best_params)

    # ── Task 5 ────────────────────────────────
    t5_res = task5_interaction(df, feat_cols, best_params)

    # ── 汇总大表 ──────────────────────────────
    model_results = {
        "DC":      t2_res,
        "Ensemble":t3_res,
        f"XGB_wf{best_wf}": t4_res,
        "XGB+inter": t5_res,
    }
    df_table = build_comparison_table(base_results, model_results)
    print_big_table(df_table)

    # 保存表格
    df_table.to_csv(OUTPUTS_DIR / "improvement_table.csv",
                    index=False, encoding="utf-8-sig")
    print(f"\n对比表已保存: outputs/improvement_table.csv")
