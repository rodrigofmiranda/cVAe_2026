# -*- coding: utf-8 -*-
"""
src/metrics/distribution.py — Distribution-fidelity metrics on I/Q residuals.

Reusable functions for comparing residual distributions between real
and model-predicted data.  Used by the protocol runner (Commit 3O)
for both deterministic baseline and cVAE evaluations.

Functions
---------
moment_deltas          Δ mean (L2), Δ covariance (Frobenius), Δ skew (L2), Δ kurtosis (L2)
psd_distance           Welch PSD L2 distance (log-domain)
gaussianity_tests      Jarque–Bera per I/Q component → p-values + reject flag
residual_fidelity_metrics  All-in-one convenience wrapper

Commit 3O.
"""

import math
import numpy as np
from typing import Dict


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

    Returns
    -------
    dict  combining moment_deltas, psd_distance, gaussianity_tests
    """
    n = min(len(residuals_real), len(residuals_pred), max_samples)
    rr = residuals_real[:n]
    rp = residuals_pred[:n]

    result = {}
    result.update(moment_deltas(rr, rp))
    result.update(psd_distance(rr, rp, nfft=psd_nfft))
    result.update(gaussianity_tests(rp, alpha=gauss_alpha))
    return result
