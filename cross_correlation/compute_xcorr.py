#!/usr/bin/env python3
"""Cross-correlation channel-identification metrics for the VLC digital twin.

Scientific basis: under the broadband full-square excitation the input is
approximately white (R_xx(tau) ~= delta), so the input-output cross-correlation
R_xy(tau) estimates the channel linear impulse response h(tau) (Wiener-Hopf /
cross-correlation system identification). For the VLC channel's Hammerstein
structure (static LED nonlinearity -> linear filter; dang_2022), R_xy(tau)
identifies the LINEAR block. Comparing R_xy_real vs R_xy_twin at unseen
distances tests whether the learned channel reproduces the channel memory.
See TESE/06_validacao_do_gemeo/cross_correlation_fundamento_2026-06-13.md.

This module implements and unit-tests the estimator core. The model/data wiring
(producing y_twin = E[y|x,d,c] from the saved cross-dist model) is in
`run_xcorr_crossdist.py` and must be verified against the real eval artifact.
"""
from __future__ import annotations

import numpy as np


def normalized_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int = 64) -> np.ndarray:
    """Normalized input-output cross-correlation R_xy(tau) for tau in [-L, L].

    R_xy(tau) = mean[(x[n]-xbar)(y[n+tau]-ybar)] / (sigma_x sigma_y)

    1-D real signals of equal length. Returns array of length 2*max_lag+1,
    index L is tau=0. Unbiased by overlap count per lag.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(len(x), len(y))
    x = x[:n] - x[:n].mean()
    y = y[:n] - y[:n].mean()
    sx = x.std() + 1e-12
    sy = y.std() + 1e-12
    L = int(max_lag)
    out = np.empty(2 * L + 1, dtype=np.float64)
    for i, tau in enumerate(range(-L, L + 1)):
        if tau >= 0:
            a, b = x[: n - tau], y[tau:]
        else:
            a, b = x[-tau:], y[: n + tau]
        out[i] = np.mean(a * b) / (sx * sy) if len(a) else 0.0
    return out


def xcorr_discrepancy(r_real: np.ndarray, r_twin: np.ndarray) -> dict:
    """Compare two R_xy(tau) curves (same lag grid, index center = tau=0).

    Returns the L2 discrepancy over lags, the peak-magnitude error (gain), and
    the peak-lag error (delay/phase, in samples).
    """
    r_real = np.asarray(r_real, dtype=np.float64)
    r_twin = np.asarray(r_twin, dtype=np.float64)
    L = (len(r_real) - 1) // 2
    lags = np.arange(-L, L + 1)
    return {
        "xcorr_l2": float(np.sqrt(np.mean((r_real - r_twin) ** 2))),
        "xcorr_linf": float(np.max(np.abs(r_real - r_twin))),
        "peak_real": float(r_real.max()),
        "peak_twin": float(r_twin.max()),
        "peak_mag_err": float(abs(r_real.max() - r_twin.max())),
        "peak_lag_real": int(lags[np.argmax(r_real)]),
        "peak_lag_twin": int(lags[np.argmax(r_twin)]),
        "peak_lag_err": int(abs(lags[np.argmax(r_real)] - lags[np.argmax(r_twin)])),
    }


def input_whiteness(x: np.ndarray, max_lag: int = 64) -> float:
    """Whiteness check of the input: ratio of off-lag autocorrelation energy to
    the tau=0 peak. ~0 => white (R_xx ~ delta => R_xy ~ h holds). Documents the
    validity precondition of the cross-correlation identification.
    """
    r = normalized_xcorr(x, x, max_lag)
    L = (len(r) - 1) // 2
    off = np.concatenate([r[:L], r[L + 1:]])
    return float(np.sqrt(np.mean(off ** 2)) / (abs(r[L]) + 1e-12))


# ---------------------------------------------------------------------------
# Self-test: white input through a known LTI filter h -> R_xy must recover h.
# ---------------------------------------------------------------------------
def _selftest() -> bool:
    rng = np.random.default_rng(33)
    N = 200_000
    x = rng.standard_normal(N)                      # white input
    h = np.array([0.2, 1.0, 0.5, -0.3, 0.1])        # known impulse response
    # causal convolution y[n] = sum_k h[k] x[n-k]  => R_xy(tau) ∝ h[tau], tau>=0
    y = np.convolve(x, h)[:N]
    # add signal-independent noise (does not bias the cross-correlation)
    y = y + 0.3 * rng.standard_normal(N)

    L = 8
    r = normalized_xcorr(x, y, max_lag=L)
    # under white causal LTI, R_xy(tau) ∝ h[tau] at non-negative lags;
    # check the recovered shape at tau=0..4 correlates ~1.0 with h.
    seg = r[L: L + len(h)]
    seg = seg / (np.linalg.norm(seg) + 1e-12)
    hn = h / (np.linalg.norm(h) + 1e-12)
    shape_corr = float(np.dot(seg, hn))

    ok = True
    def ck(name, cond, info=""):
        nonlocal ok
        print(f"{'OK ' if cond else 'FAIL'} {name} {info}")
        ok = ok and cond

    ck("recupera forma de h (corr>0.97)", shape_corr > 0.97, f"corr={shape_corr:.4f}")

    # whiteness of a white input ~ 0; of a strongly colored input >> 0
    w_white = input_whiteness(x, 32)
    xc = np.convolve(rng.standard_normal(N), np.ones(20) / 20, mode="same")  # colored
    w_color = input_whiteness(xc, 32)
    ck("whiteness branco baixo", w_white < 0.05, f"w={w_white:.4f}")
    ck("whiteness colorido alto", w_color > 0.3, f"w={w_color:.4f}")

    # discrepancy of identical curves is zero; of shifted curve is positive
    d0 = xcorr_discrepancy(r, r)
    rshift = np.roll(r, 2)
    d1 = xcorr_discrepancy(r, rshift)
    ck("discrepância idêntica = 0", d0["xcorr_l2"] < 1e-9, f"l2={d0['xcorr_l2']:.2e}")
    ck("discrepância deslocada detecta lag", d1["peak_lag_err"] == 2, f"lag_err={d1['peak_lag_err']}")

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
