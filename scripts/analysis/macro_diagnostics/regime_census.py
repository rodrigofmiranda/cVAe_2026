#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer 0 — model-free regime census of the REAL channel (runs in docker, numpy).

For every regime (dist x curr) of each base model's training dataset, loads the
measured X/Y arrays and characterises the REAL noise residual delta = Y - a·X
(per-axis linear gain a removed, so moments reflect the NOISE, not the attenuated
input shape). This is the foundation of the data-separation decision: it
describes the *data itself*
(non-Gaussianity, tail, heteroscedastic spread, SNR, sample count) independently
of any twin — so the recommendation rests on the measured distribution, not the
model.

No GPU, no model load. Reads dataset roots from the awgn manifests.

Reports BOTH residual views (user choice):
  canonical  δ = Y - X    -> *_canon columns (project-consistent, heteroscedastic;
                             the signature of the static nonlinear gain map per
                             dang_2022 Hammerstein / bojarczuk_2023 Volterra).
  equalized  δ = Y - a·X  -> *_eq columns (linear gain removed; isolates the noise,
                             which is ~Gaussian & ~homoscedastic in V3).

Variance budget (option 1b): a per-regime static nonlinear gain map g(P)·x is
fitted (train half) and the residual measured (test half) to decompose Var(Y-X)
into frac_linear / frac_nonlinear / frac_noise. This answers the training-redesign
question: is the per-regime difficulty the deterministic (nonlinear) map, or the
noise distribution? If the nonlinear-equalised residual is Gaussian (kurt_nl≈0),
adding MDN tail capacity is the wrong target.

Output: <out>/regime_census.csv  (one row per model, dist_m, curr_mA)
Columns: real_channel_snr_db, n_samples, real_var_mag, delta_p50..delta_max,
  real_skew_canon, real_kurt_canon, het_slope_canon,
  real_skew_eq, real_kurt_eq, het_slope_eq,
  frac_linear, frac_nonlinear, frac_noise, real_kurt_nl, real_skew_nl, het_slope_nl
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

DEFAULT_MANIFESTS = {
    "FC": "/home/rodrigo/comparison_v3/awgn/fc/manifest.json",
    "FS": "/home/rodrigo/comparison_v3/awgn/fs/manifest.json",
}


def _dataset_root(manifest_path: str) -> str:
    root = json.load(open(manifest_path))["dataset_root"]
    # host path may say /data/... (docker mount). Try as-is, then host fallback.
    if Path(root).exists():
        return root
    alt = root.replace("/data/", "/home/rodrigo/1-Data/")
    return alt if Path(alt).exists() else root


def _equalized_residual(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Noise residual AFTER removing the per-axis linear gain: δ = Y - a·X.

    Without this, δ = Y - X = (a-1)·X + noise is dominated by the attenuated
    INPUT shape (circle/square), which is platykurtic and makes the residual
    look non-Gaussian for the wrong reason (gain, not noise). Equalising by the
    LS gain a = <X,Y>/<X,X> per axis isolates the actual additive noise — the
    same convention used by modulation_check.equalize_axis and the xcorr driver.
    """
    out = np.empty_like(Y)
    for ax in (0, 1):
        denom = float(np.sum(X[:, ax] ** 2))
        a = float(np.sum(X[:, ax] * Y[:, ax]) / denom) if denom > 1e-18 else 1.0
        out[:, ax] = Y[:, ax] - a * X[:, ax]
    return out


def _nl_features(X: np.ndarray, ax: int) -> np.ndarray:
    """Amplitude-dependent-gain features for axis `ax`: [1, x, x·P, x·P²] with
    P=|X|² the transmitted power. Captures a static memoryless nonlinear gain
    g(P)·x (Hammerstein saturation / Volterra squarer), 4 params per axis."""
    x = X[:, ax]
    P = X[:, 0] ** 2 + X[:, 1] ** 2
    return np.column_stack([np.ones_like(x), x, x * P, x * P * P])


def _residual_budget(Xtr, Ytr, Xte, Yte) -> dict:
    """Variance budget of the apparent residual Y-X, fit on train / measured on
    test (overfit-proof). Decomposes how much of Var(Y-X) is explained by:
      - a per-axis linear gain a·x          -> frac_linear
      - the extra static nonlinear gain map -> frac_nonlinear
      - the irreducible residual (noise)    -> frac_noise
    Also returns the moments of the nonlinear-equalised residual (test half):
    if it is Gaussian (kurt≈0) and homoscedastic, the channel is a deterministic
    map + benign Gaussian noise -> chasing MDN tail capacity is the wrong target.
    """
    res_lin = np.empty_like(Yte)
    res_nl = np.empty_like(Yte)
    for ax in (0, 1):
        den = float(np.sum(Xtr[:, ax] ** 2))
        a = float(np.sum(Xtr[:, ax] * Ytr[:, ax]) / den) if den > 1e-18 else 1.0
        res_lin[:, ax] = Yte[:, ax] - a * Xte[:, ax]
        coef, *_ = np.linalg.lstsq(_nl_features(Xtr, ax), Ytr[:, ax], rcond=None)
        res_nl[:, ax] = Yte[:, ax] - _nl_features(Xte, ax) @ coef
    v_canon = float(np.var(Yte[:, 0] - Xte[:, 0]) + np.var(Yte[:, 1] - Xte[:, 1]))
    v_lin = float(np.var(res_lin[:, 0]) + np.var(res_lin[:, 1]))
    v_nl = float(np.var(res_nl[:, 0]) + np.var(res_nl[:, 1]))
    v_canon = max(v_canon, 1e-18)
    sk_nl, ku_nl, _ = _moments(res_nl)
    het_nl = _het_slope(Xte, res_nl)
    return {
        "frac_linear": (v_canon - v_lin) / v_canon,
        "frac_nonlinear": (v_lin - v_nl) / v_canon,
        "frac_noise": v_nl / v_canon,
        "real_kurt_nl": ku_nl,
        "real_skew_nl": sk_nl,
        "het_slope_nl": het_nl,
    }


def _moments(d: np.ndarray) -> tuple[float, float, float]:
    """Per-axis skew / excess-kurt averaged over I/Q + var of |delta|."""
    mean = d.mean(0)
    var = d.var(0)
    s = np.sqrt(var + 1e-12)
    z = (d - mean) / s
    skew = float(np.mean(np.abs(np.mean(z ** 3, 0))))
    kurt = float(np.mean(np.mean(z ** 4, 0) - 3.0))
    mag = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    return skew, kurt, float(np.var(mag))


def _het_slope(X: np.ndarray, d: np.ndarray, n_bins: int = 20) -> float:
    """Heteroscedastic slope: Var(|δ|) vs transmitted power |X|^2 across amplitude
    bins. This is the real between-regime discriminator (shot noise: variance
    grows with amplitude) and exactly what a homoscedastic AWGN channel cannot
    reproduce. Returns the linear slope (0 => homoscedastic)."""
    power = X[:, 0] ** 2 + X[:, 1] ** 2
    order = np.argsort(power)
    power, ds = power[order], d[order]
    edges = np.linspace(0, len(power), n_bins + 1, dtype=int)
    px, vy = [], []
    for i in range(n_bins):
        s, e = edges[i], edges[i + 1]
        if e - s < 50:
            continue
        px.append(float(np.mean(power[s:e])))
        vy.append(float(np.var(ds[s:e, 0]) + np.var(ds[s:e, 1])))  # noise var in this power bin
    if len(px) < 3:
        return float("nan")
    return float(np.polyfit(px, vy, 1)[0])


def _snr_db(X: np.ndarray, Y: np.ndarray) -> float:
    sp = float(np.mean(X[:, 0] ** 2 + X[:, 1] ** 2))
    npow = float(np.mean((Y[:, 0] - X[:, 0]) ** 2 + (Y[:, 1] - X[:, 1]) ** 2))
    return float(10.0 * np.log10(max(sp / max(npow, 1e-18), 1e-18)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output dir for regime_census.csv")
    ap.add_argument("--n-cap", type=int, default=60000)
    ap.add_argument("--manifests", default=None,
                    help="JSON dict model->manifest path (default: awgn fc/fs manifests)")
    args = ap.parse_args()

    manifests = json.loads(args.manifests) if args.manifests else DEFAULT_MANIFESTS
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for model, man in manifests.items():
        root = _dataset_root(man)
        print(f"[{model}] dataset_root={root}", flush=True)
        for dd in sorted(glob.glob(f"{root}/dist_*")):
            md = re.search(r"dist_([0-9.]+)m", os.path.basename(dd))
            if not md:
                continue
            dist = float(md.group(1))
            for cd in sorted(glob.glob(f"{dd}/curr_*mA")):
                mc = re.search(r"curr_([0-9]+)mA", os.path.basename(cd))
                if not mc:
                    continue
                curr = int(mc.group(1))
                caps = sorted(glob.glob(f"{cd}/*/IQ_data/"))
                if not caps:
                    continue
                X = np.load(caps[0] + "X.npy")[: args.n_cap].astype(np.float64)
                Y = np.load(caps[0] + "Y.npy")[: args.n_cap].astype(np.float64)
                n = min(len(X), len(Y))
                X, Y = X[:n], Y[:n]
                # Two views (user choice "report both"):
                #  canonical δ = Y - X       -> project-consistent (gates/shot_noise);
                #                               heteroscedastic = signature of the static
                #                               nonlinear gain map (Hammerstein/Volterra).
                #  equalized δ = Y - a·X      -> removes per-axis linear gain; isolates noise.
                d_canon = Y - X
                d_eq = _equalized_residual(X, Y)
                sk_c, ku_c, var_c = _moments(d_canon)
                sk_e, ku_e, _ = _moments(d_eq)
                het_c = _het_slope(X, d_canon)
                het_e = _het_slope(X, d_eq)
                # Variance budget via nonlinear-map fit (train/test split, overfit-proof)
                h = n // 2
                bud = _residual_budget(X[:h], Y[:h], X[h:], Y[h:])
                mag = np.sqrt(d_canon[:, 0] ** 2 + d_canon[:, 1] ** 2)
                p = np.percentile(mag, [50, 90, 99, 99.9])
                rows.append({
                    "model": model, "dist_m": dist, "curr_mA": curr,
                    "n_samples": n,
                    "real_channel_snr_db": round(_snr_db(X, Y), 4),
                    "real_skew_canon": round(sk_c, 5),
                    "real_kurt_canon": round(ku_c, 5),
                    "het_slope_canon": round(het_c, 8),
                    "real_skew_eq": round(sk_e, 5),
                    "real_kurt_eq": round(ku_e, 5),
                    "het_slope_eq": round(het_e, 8),
                    "frac_linear": round(bud["frac_linear"], 5),
                    "frac_nonlinear": round(bud["frac_nonlinear"], 5),
                    "frac_noise": round(bud["frac_noise"], 5),
                    "real_kurt_nl": round(bud["real_kurt_nl"], 5),
                    "real_skew_nl": round(bud["real_skew_nl"], 5),
                    "het_slope_nl": round(bud["het_slope_nl"], 8),
                    "real_var_mag": round(var_c, 8),
                    "delta_p50": round(float(p[0]), 6),
                    "delta_p90": round(float(p[1]), 6),
                    "delta_p99": round(float(p[2]), 6),
                    "delta_p999": round(float(p[3]), 6),
                    "delta_max": round(float(mag.max()), 6),
                })
                print(f"  {model} {dist:g}m/{curr}mA: N={n} | budget lin={bud['frac_linear']:.3f} "
                      f"nl={bud['frac_nonlinear']:.3f} noise={bud['frac_noise']:.3f} "
                      f"| kurt_nl={bud['real_kurt_nl']:.2f} | snr={rows[-1]['real_channel_snr_db']:.1f}dB",
                      flush=True)
                del X, Y, d_canon, d_eq, mag

    if rows:
        out_csv = os.path.join(args.out, "regime_census.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nescrito: {out_csv} ({len(rows)} regimes)")


if __name__ == "__main__":
    main()
