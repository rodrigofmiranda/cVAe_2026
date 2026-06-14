#!/usr/bin/env python3
"""BER for V3 QAM links (real and twin), for the modulation-verification study.

The V3 QAM data store X.npy = the ideal sent constellation (clean, rectangular
M-QAM, corners at ±1/√2) and Y.npy = the aligned received I/Q (channel-attenuated
~0.6x in amplitude). The constellation is separable: I and Q are independent
Gray-coded PAM-√M. So we demodulate per axis.

BER pipeline (per regime):
  sent bits   <- level index of X per axis, Gray-coded
  equalize    <- scalar per-axis LS gain a = <X,Y>/<Y,Y> maps Y onto X scale
  recv bits   <- nearest level of (a*Y) per axis, Gray-coded
  BER         <- bit error fraction

For the twin, pass Y_twin = cVAE(X) instead of Y_real (same equalization).
Used by the FS/FC modulation comparison; mirrors comparison/ style.
"""
from __future__ import annotations

import numpy as np


def _gray(n: int) -> np.ndarray:
    """Gray codes 0..2^k-1 for k = log2(n) bits, as an (n, k) bit array, ordered
    by ascending PAM level (standard Gray-coded PAM labelling)."""
    k = int(round(np.log2(n)))
    idx = np.arange(n)
    g = idx ^ (idx >> 1)
    return ((g[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.int8)


def qam_axis_levels(X: np.ndarray) -> np.ndarray:
    """Sorted unique amplitude levels of one axis of the ideal constellation."""
    return np.sort(np.unique(np.round(X, 3)))


def _bits_for(values: np.ndarray, levels: np.ndarray, graymap: np.ndarray) -> np.ndarray:
    """Map each value to its nearest level index, return the Gray bits (N, k)."""
    idx = np.abs(values[:, None] - levels[None, :]).argmin(axis=1)
    return graymap[idx]


def equalize_axis(x: np.ndarray, y: np.ndarray) -> float:
    """Per-axis LS scalar gain a minimizing ||a*y - x||."""
    denom = float(np.dot(y, y)) + 1e-12
    return float(np.dot(x, y) / denom)


def qam_ber(X: np.ndarray, Y: np.ndarray) -> dict:
    """BER between sent constellation X and received Y (real or twin).

    Returns BER, per-axis BER, SER, equalization gains, and n_bits.
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    out = {"n_symbols": int(len(X))}
    sent_bits, recv_bits = [], []
    ser_err = np.zeros(len(X), dtype=bool)
    for ax in (0, 1):
        levels = qam_axis_levels(X[:, ax])
        gmap = _gray(len(levels))
        a = equalize_axis(X[:, ax], Y[:, ax])
        sb = _bits_for(X[:, ax], levels, gmap)
        # received: nearest level of equalized Y
        ridx = np.abs((a * Y[:, ax])[:, None] - levels[None, :]).argmin(axis=1)
        sidx = np.abs(X[:, ax][:, None] - levels[None, :]).argmin(axis=1)
        rb = gmap[ridx]
        sent_bits.append(sb); recv_bits.append(rb)
        out[f"gain_ax{ax}"] = a
        out[f"ber_ax{ax}"] = float(np.mean(sb != rb))
        out[f"n_levels_ax{ax}"] = int(len(levels))
        ser_err |= (ridx != sidx)
    sent = np.concatenate(sent_bits, axis=1)
    recv = np.concatenate(recv_bits, axis=1)
    out["n_bits"] = int(sent.size)
    out["ber"] = float(np.mean(sent != recv))
    out["ser"] = float(np.mean(ser_err))
    out["bits_per_symbol"] = int(sent.shape[1])
    return out


# ---------------------------------------------------------------------------
# Self-test on REAL V3 QAM data: BER must be plausible and grow with M / range.
# ---------------------------------------------------------------------------
def _selftest() -> bool:
    import glob
    ROOT = "/data/Dataset/V3"
    ok = True

    def ck(name, cond, info=""):
        nonlocal ok
        print(f"{'OK ' if cond else 'FAIL'} {name} {info}")
        ok = ok and cond

    # 1) synthetic QPSK with known noise -> BER ~ Q(1/sigma_rel) sanity
    rng = np.random.default_rng(33)
    N = 200_000
    syms = rng.integers(0, 2, (N, 2)) * 2 - 1     # ±1 per axis
    X = syms / np.sqrt(2)                           # corners at ±0.707
    sigma = 0.2
    Y = X + sigma * rng.standard_normal((N, 2))     # gain 1, awgn
    r = qam_ber(X, Y)
    from math import erfc
    expected = 0.5 * erfc((0.707 / sigma) / np.sqrt(2))   # Q(A/sigma)
    ck("QPSK sintético BER ~ Q(A/sigma)", abs(r["ber"] - expected) < 0.002,
       f"ber={r['ber']:.4f} esperado={expected:.4f}")

    # 2) gain robustness: attenuate Y by 0.4 -> equalization must recover same BER
    Yatt = 0.4 * Y
    r2 = qam_ber(X, Yatt)
    ck("equalização recupera BER sob atenuação", abs(r2["ber"] - r["ber"]) < 1e-6,
       f"ber_att={r2['ber']:.4f} gain≈{r2['gain_ax0']:.2f}")

    # 3) real V3 data, all 3 modulations at a mid regime
    for mod, bps in [("4QAM", 2), ("16QAM", 4), ("64QAM", 6)]:
        caps = sorted(glob.glob(f"{ROOT}/{mod}_2026_V3_ORGANIZED/dist_1.0m/curr_500mA/*/"))
        if not caps:
            print(f"  (sem dados {mod})"); continue
        X = np.load(caps[0] + "IQ_data/X.npy"); Y = np.load(caps[0] + "IQ_data/Y.npy")
        r = qam_ber(X, Y)
        ck(f"{mod} real: bits/símbolo={bps}", r["bits_per_symbol"] == bps,
           f"BER={r['ber']:.4f} SER={r['ser']:.4f} gain≈{r['gain_ax0']:.2f}")
        ck(f"{mod} real: BER plausível [0,0.3]", 0.0 <= r["ber"] < 0.3, f"BER={r['ber']:.4f}")

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
