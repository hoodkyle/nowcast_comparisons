"""Scalar AR(1) Kalman factor with training-only EM calibration."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def _filter(obs: np.ndarray, rho: float, q: float, r: float, p0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(obs); mp = np.zeros(n); pp = np.zeros(n); mf = np.zeros(n); pf = np.zeros(n); ll = 0.0
    for t in range(n):
        if t == 0: mp[t], pp[t] = 0.0, p0
        else: mp[t], pp[t] = rho * mf[t - 1], rho * rho * pf[t - 1] + q
        if np.isfinite(obs[t]):
            var = max(pp[t] + r, 1e-12); gain = pp[t] / var
            mf[t], pf[t] = mp[t] + gain * (obs[t] - mp[t]), max((1.0 - gain) * pp[t], 1e-12)
            ll += -.5 * (np.log(2 * np.pi * var) + (obs[t] - mp[t]) ** 2 / var)
        else: mf[t], pf[t] = mp[t], pp[t]
    return mp, pp, mf, pf, float(ll)


def _smooth(mp: np.ndarray, pp: np.ndarray, mf: np.ndarray, pf: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(mf); ms, ps = mf.copy(), pf.copy(); cross = np.zeros(max(n - 1, 0))
    for t in range(n - 2, -1, -1):
        j = pf[t] * rho / max(pp[t + 1], 1e-12)
        ms[t] = mf[t] + j * (ms[t + 1] - mp[t + 1])
        ps[t] = max(pf[t] + j * j * (ps[t + 1] - pp[t + 1]), 1e-12)
        cross[t] = ms[t + 1] * ms[t] + j * ps[t + 1]
    return ms, ps, cross


def _fit_em(obs: np.ndarray, spec: dict[str, Any]) -> tuple[float, float, float, np.ndarray, dict[str, Any]]:
    floor = float(spec.get("variance_floor", 1e-4)); p0 = float(spec.get("P", 1.0))
    rho = float(spec.get("initial_rho", .8)); q = float(spec.get("Q", 1.0)); r = float(spec.get("R", .5))
    _, _, _, _, initial_ll = _filter(obs, rho, q, r, p0); ll_trace = [initial_ll]; converged = False
    for iteration in range(1, int(spec.get("max_em_iterations", 50)) + 1):
        mp, pp, mf, pf, _ = _filter(obs, rho, q, r, p0); ms, ps, cross = _smooth(mp, pp, mf, pf, rho)
        if len(ms) > 1:
            denom = np.sum(ps[:-1] + ms[:-1] ** 2)
            new_rho = float(np.clip(np.sum(cross) / max(denom, floor), -.98, .98))
            new_q = max(float(np.mean(ps[1:] + ms[1:] ** 2 - 2 * new_rho * cross + new_rho ** 2 * (ps[:-1] + ms[:-1] ** 2))), floor)
        else: new_rho, new_q = rho, q
        finite = np.isfinite(obs)
        new_r = max(float(np.mean((obs[finite] - ms[finite]) ** 2 + ps[finite])) if finite.any() else r, floor)
        rho, q, r = new_rho, new_q, new_r
        ll = _filter(obs, rho, q, r, p0)[-1]; ll_trace.append(ll)
        if abs(ll_trace[-1] - ll_trace[-2]) <= float(spec.get("em_tolerance", 1e-6)):
            converged = True; break
    filt = _filter(obs, rho, q, r, p0)[2]
    return rho, q, r, filt, {"initial_loglik": ll_trace[0], "final_loglik": ll_trace[-1], "em_iterations": len(ll_trace) - 1, "em_converged": converged, "em_loglik_trace": ll_trace}


def fit_state_space(ds: Any, spec: dict[str, Any], seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    tx = np.asarray(ds.train["x"], float); y = np.asarray(ds.train["y"], float); qx = np.asarray(ds.score["x"], float); eligible = np.asarray(ds.score["eligible"], bool)
    flat = tx.reshape(len(tx), -1); qflat = qx.reshape(len(qx), -1); mean, scale = flat.mean(0), np.where(flat.std(0) > 1e-12, flat.std(0), 1.0)
    flat = (flat - mean) / scale; pca = PCA(n_components=1, random_state=seed).fit(flat); factors = pca.transform(flat).ravel()
    score_factor = np.full(len(qflat), np.nan)
    if eligible.any(): score_factor[eligible] = pca.transform((qflat[eligible] - mean) / scale).ravel()
    train_dates = pd.to_datetime(ds.train["keys"]["date"]).dt.to_period("M").dt.to_timestamp()
    score_dates = pd.to_datetime(ds.score["keys"]["date"]).dt.to_period("M").dt.to_timestamp()
    origin = score_dates.min(); calendar = pd.date_range(train_dates.min(), origin - pd.DateOffset(months=1), freq="MS")
    obs = np.asarray([np.mean(factors[train_dates.eq(d)]) if np.any(train_dates.eq(d)) else np.nan for d in calendar])
    rho, q, r, filtered, emmeta = _fit_em(obs, spec)
    train_factor = np.asarray([filtered[np.flatnonzero(calendar == d)[0]] for d in train_dates])
    states = np.asarray(ds.train["states"], str); score_states = np.asarray(ds.score["states"], str); levels = np.unique(states)
    def effects(values: np.ndarray) -> np.ndarray: return np.column_stack([values == level for level in levels[1:]]).astype(float) if len(levels) > 1 else np.empty((len(values), 0))
    design = np.column_stack([train_factor, effects(states), np.ones(len(states))])
    penalty = np.diag([1.0] + [0.0] * (design.shape[1] - 1))
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    # Append one aggregate score-month measurement and filter it with frozen
    # parameters. This permits the current covariate measurement only.
    pred_factor = rho * filtered[-1] if len(filtered) else 0.0
    _, _, filtered_all, pf_all, _ = _filter(np.r_[obs, np.nanmean(score_factor[eligible]) if eligible.any() else np.nan], rho, q, r, float(spec.get("P", 1.0)))
    pred_factor = filtered_all[-1]
    score_design = np.column_stack([np.full(len(score_states), pred_factor), effects(score_states), np.ones(len(score_states))]); pred = np.full(len(score_states), np.nan)
    if eligible.any(): pred[eligible] = score_design[eligible] @ coef
    meta = {"model": spec.get("name", "scalar_AR1_state_space"), "rows": len(y), "regression_and_dynamic_parameter_count": int(len(coef) + 3), "rho": rho, "Q": q, "R": r, "factor_months": len(calendar), "features": ["PC1", "state_effects"], **emmeta}
    return pred, meta
