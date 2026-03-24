# -*- coding: utf-8 -*-
"""
src/metrics/distribution.py — Distribution-fidelity metrics on I/Q residuals.

Reusable functions for comparing residual distributions between real
and model-predicted data.  Used by the protocol runner (Commit 3O)
for both deterministic baseline and cVAE evaluations.

Canonical δ definition
----------------------
δ = Y − X (element-wise).  X and Y are the float32 (N, 2) arrays stored per
experiment; their signal-chain semantics are defined by the dataset preparation
pipeline (see src/data/channel_dataset.py), not asserted here.

Functions
---------
moment_deltas          Δ mean (L2), Δ covariance (Frobenius), Δ skew (L2), Δ kurtosis (L2)
psd_distance           Welch PSD L2 distance (log-domain)
acf_distance           Normalised ACF L2 distance on complex residuals
gaussianity_tests      Jarque–Bera per I/Q component → p-values + reject flag
residual_fidelity_metrics  All-in-one convenience wrapper

Commit 3O.
"""

import math
import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Moment deltas
# ---------------------------------------------------------------------------

def moment_deltas(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Moment deltas between two [N, 2] arrays (e.g. real vs predicted residuals).

    Parameters
    ----------
    a : ndarray (N, 2) — reference (real residuals)
    b : ndarray (N, 2) — test     (model residuals)

    Returns
    -------
    dict  delta_mean_l2, delta_cov_fro, delta_skew_l2, delta_kurt_l2
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Mean
    mean_l2 = float(np.linalg.norm(np.mean(b, axis=0) - np.mean(a, axis=0)))

    # Covariance (Frobenius norm of Δ covariance matrix)
    cov_fro = float(np.linalg.norm(np.cov(b.T) - np.cov(a.T), ord="fro"))

    # Skewness & excess kurtosis
    eps = 1e-12

    def _sk(x):
        m = np.mean(x, axis=0)
        v = np.var(x, axis=0)
        s = np.sqrt(v + eps)
        z = (x - m) / s
        skew = np.mean(z ** 3, axis=0)
        kurt = np.mean(z ** 4, axis=0) - 3.0
        return skew, kurt

    skew_a, kurt_a = _sk(a)
    skew_b, kurt_b = _sk(b)
    skew_l2 = float(np.linalg.norm(skew_b - skew_a))
    kurt_l2 = float(np.linalg.norm(kurt_b - kurt_a))

    return {
        "delta_mean_l2": mean_l2,
        "delta_cov_fro": cov_fro,
        "delta_skew_l2": skew_l2,
        "delta_kurt_l2": kurt_l2,
    }


# ---------------------------------------------------------------------------
# PSD distance
# ---------------------------------------------------------------------------

def psd_distance(a: np.ndarray, b: np.ndarray, nfft: int = 2048) -> Dict[str, float]:
    """
    PSD L2 distance (log-domain) between two [N, 2] I/Q arrays.

    Uses the shared `_psd_log` helper from ``src.evaluation.metrics``
    (Welch-like, Hanning-windowed, 4 segments).

    Parameters
    ----------
    a, b : ndarray (N, 2)
    nfft : int

    Returns
    -------
    dict  psd_l2
    """
    from src.evaluation.metrics import _psd_log

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    ca = a[:, 0] + 1j * a[:, 1]
    cb = b[:, 0] + 1j * b[:, 1]

    psd_a = _psd_log(ca, nfft=int(nfft))
    psd_b = _psd_log(cb, nfft=int(nfft))

    psd_l2 = float(np.linalg.norm(psd_b - psd_a) / np.sqrt(max(len(psd_a), 1)))
    return {"psd_l2": psd_l2}


def acf_distance(a: np.ndarray, b: np.ndarray, max_lag: int = 128) -> Dict[str, float]:
    """Normalised ACF distance between two [N, 2] I/Q arrays."""
    from src.evaluation.metrics import _acf_curve_complex

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ca = a[:, 0] + 1j * a[:, 1]
    cb = b[:, 0] + 1j * b[:, 1]
    acf_a = _acf_curve_complex(ca, max_lag=int(max_lag))
    acf_b = _acf_curve_complex(cb, max_lag=int(max_lag))
    acf_l2 = float(np.linalg.norm(acf_b - acf_a) / np.sqrt(max(len(acf_a), 1)))
    return {"delta_acf_l2": acf_l2}


# ---------------------------------------------------------------------------
# Gaussianity tests
# ---------------------------------------------------------------------------

def gaussianity_tests(
    residuals: np.ndarray,
    alpha: float = 0.01,
) -> Dict:
    """
    Jarque–Bera test per I/Q component of [N, 2] residuals.

    Falls back to a manual JB + chi²(2) survival approximation if SciPy
    is not installed.

    Parameters
    ----------
    residuals : ndarray (N, 2)
    alpha     : float — significance level (default 0.01)

    Returns
    -------
    dict
        jb_stat_I, jb_stat_Q, jb_p_I, jb_p_Q, jb_p_min,
        jb_log10p_I, jb_log10p_Q, jb_log10p_min, reject_gaussian
    """
    residuals = np.asarray(residuals, dtype=np.float64)

    def _manual_jb(x):
        """Manual Jarque-Bera statistic + chi²(2) p-value approximation."""
        n = len(x)
        eps = 1e-12
        m = np.mean(x)
        v = np.var(x)
        s = np.sqrt(v + eps)
        z = (x - m) / s
        S = float(np.mean(z ** 3))
        K = float(np.mean(z ** 4) - 3.0)
        jb = (n / 6.0) * (S ** 2 + K ** 2 / 4.0)
        # chi²(2) survival:  P(X > jb) = exp(-jb / 2)
        p = float(np.exp(-jb / 2.0))
        return jb, p

    try:
        from scipy.stats import jarque_bera
        jb_I = jarque_bera(residuals[:, 0])
        jb_Q = jarque_bera(residuals[:, 1])
        stat_I, p_I = float(jb_I.statistic), float(jb_I.pvalue)
        stat_Q, p_Q = float(jb_Q.statistic), float(jb_Q.pvalue)
    except (ImportError, AttributeError):
        stat_I, p_I = _manual_jb(residuals[:, 0])
        stat_Q, p_Q = _manual_jb(residuals[:, 1])

    def _safe_log10p(p: float, stat: float) -> float:
        """Stable log10(p), with chi²(2) tail fallback under underflow."""
        if p > 0.0 and np.isfinite(p):
            return float(math.log10(p))
        # Under H0, JB ~ chi2(df=2) and for large stat:
        # p ≈ exp(-stat/2) => log10(p) ≈ -stat / (2 ln 10)
        return float(-stat / (2.0 * math.log(10.0)))

    p_min = min(p_I, p_Q)
    log10p_I = _safe_log10p(p_I, stat_I)
    log10p_Q = _safe_log10p(p_Q, stat_Q)
    return {
        "jb_stat_I": stat_I,
        "jb_stat_Q": stat_Q,
        "jb_p_I": p_I,
        "jb_p_Q": p_Q,
        "jb_p_min": p_min,
        "jb_log10p_I": log10p_I,
        "jb_log10p_Q": log10p_Q,
        "jb_log10p_min": min(log10p_I, log10p_Q),
        "reject_gaussian": bool(p_min < alpha),
    }


# ---------------------------------------------------------------------------
# Combined convenience
# ---------------------------------------------------------------------------

def residual_fidelity_metrics(
    residuals_real: np.ndarray,
    residuals_pred: np.ndarray,
    psd_nfft: int = 2048,
    gauss_alpha: float = 0.01,
    max_samples: int = 200_000,
    acf_max_lag: int = 128,
    X: Optional[np.ndarray] = None,
    X_pred: Optional[np.ndarray] = None,
) -> Dict:
    """
    All distribution-fidelity metrics between real and predicted residuals.

    Parameters
    ----------
    residuals_real : ndarray (N, 2)  — y − x  (real channel)
    residuals_pred : ndarray (N, 2)  — ŷ − x  (model)
    psd_nfft       : int
    gauss_alpha    : float
    max_samples    : int — cap to bound computation
    X              : ndarray (N, 2), optional — input I/Q paired with residuals_real;
                     if provided, computes rho_hetero_real = Pearson(|X_c|, |δ_real_c|²).
    X_pred         : ndarray (N, 2), optional — input I/Q paired with residuals_pred;
                     if None, falls back to X (only use X_pred when residuals_pred was
                     drawn from a different index than residuals_real, e.g. independent
                     MC sub-samples in the protocol runner).

    Returns
    -------
    dict  combining moment_deltas, psd_distance, acf_distance, gaussianity_tests,
          and optionally rho_hetero_real / rho_hetero_pred.
    """
    n = min(len(residuals_real), len(residuals_pred), max_samples)
    rr = residuals_real[:n]
    rp = residuals_pred[:n]

    result = {}
    result.update(moment_deltas(rr, rp))
    result.update(psd_distance(rr, rp, nfft=psd_nfft))
    result.update(acf_distance(rr, rp, max_lag=acf_max_lag))

    from src.evaluation.metrics import _skew_kurt, _wasserstein_1d

    mean_real = np.mean(rr, axis=0)
    mean_pred = np.mean(rp, axis=0)
    std_real = np.std(rr, axis=0)
    std_pred = np.std(rp, axis=0)
    skew_real, kurt_real = _skew_kurt(rr)
    skew_pred, kurt_pred = _skew_kurt(rp)

    result.update({
        "mean_real_delta_I": float(mean_real[0]),
        "mean_real_delta_Q": float(mean_real[1]),
        "mean_pred_delta_I": float(mean_pred[0]),
        "mean_pred_delta_Q": float(mean_pred[1]),
        "std_real_delta_I": float(std_real[0]),
        "std_real_delta_Q": float(std_real[1]),
        "std_pred_delta_I": float(std_pred[0]),
        "std_pred_delta_Q": float(std_pred[1]),
        "delta_mean_I": float(mean_pred[0] - mean_real[0]),
        "delta_mean_Q": float(mean_pred[1] - mean_real[1]),
        "delta_std_I": float(std_pred[0] - std_real[0]),
        "delta_std_Q": float(std_pred[1] - std_real[1]),
        "delta_skew_I": float(skew_pred[0] - skew_real[0]),
        "delta_skew_Q": float(skew_pred[1] - skew_real[1]),
        "delta_kurt_I": float(kurt_pred[0] - kurt_real[0]),
        "delta_kurt_Q": float(kurt_pred[1] - kurt_real[1]),
        "delta_wasserstein_I": _wasserstein_1d(rr[:, 0], rp[:, 0]),
        "delta_wasserstein_Q": _wasserstein_1d(rr[:, 1], rp[:, 1]),
    })

    try:
        from src.evaluation.stat_tests.jsd import jsd_2d
        result["stat_jsd"] = jsd_2d(rr, rp)
    except ImportError:
        result["stat_jsd"] = float("nan")

    g_real = gaussianity_tests(rr, alpha=gauss_alpha)
    result.update({
        "jb_real_stat_I": g_real["jb_stat_I"],
        "jb_real_stat_Q": g_real["jb_stat_Q"],
        "jb_real_p_I": g_real["jb_p_I"],
        "jb_real_p_Q": g_real["jb_p_Q"],
        "jb_real_p_min": g_real["jb_p_min"],
        "jb_real_log10p_I": g_real["jb_log10p_I"],
        "jb_real_log10p_Q": g_real["jb_log10p_Q"],
        "jb_real_log10p_min": g_real["jb_log10p_min"],
        "jb_real_reject_gaussian": g_real["reject_gaussian"],
    })
    result.update(gaussianity_tests(rp, alpha=gauss_alpha))
    for axis in ("I", "Q"):
        pred = float(result.get(f"jb_log10p_{axis}", float("nan")))
        real = float(result.get(f"jb_real_log10p_{axis}", float("nan")))
        if np.isfinite(pred) and np.isfinite(real):
            delta = abs(pred - real)
            rel = delta / abs(real) if abs(real) > 0 else float("nan")
        else:
            delta = float("nan")
            rel = float("nan")
        result[f"delta_jb_log10p_{axis}"] = float(delta)
        result[f"delta_jb_stat_rel_{axis}"] = float(rel)

    if X is not None:
        def _rho(x_arr: np.ndarray, d_arr: np.ndarray) -> float:
            xc = x_arr[:, 0] + 1j * x_arr[:, 1]
            dc = d_arr[:, 0] + 1j * d_arr[:, 1]
            amp_x = np.abs(xc)
            amp_d2 = np.abs(dc) ** 2
            if np.std(amp_x) < 1e-12 or np.std(amp_d2) < 1e-12:
                return float("nan")
            return float(np.corrcoef(amp_x, amp_d2)[0, 1])

        Xn_real = np.asarray(X)[:n]
        result["rho_hetero_real"] = _rho(Xn_real, rr)

        # Use X_pred when the predicted residuals are paired with a different X
        # sub-sample (e.g. independent MC draws in the protocol runner).
        Xn_pred = np.asarray(X_pred)[:n] if X_pred is not None else Xn_real
        result["rho_hetero_pred"] = _rho(Xn_pred, rp)

    return result
