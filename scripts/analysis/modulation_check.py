#!/usr/bin/env python3
"""Lightweight modulation verification: run an FS/FC base twin on QAM signals.

Loads the saved channel twin ONCE, generates a stochastic Y_twin ~ p(y|x,d,c)
for each QAM regime (one realisation, with noise — comparable to the single real
measurement), and reports BER/EVM real-vs-twin plus a constellation overlay
image. Memory-light: model held once, one capped regime at a time. Avoids the
heavy per-regime eval (dashboard/stat/reanalysis) that OOM'd.

Scientific intent: the channel is physical (Hammerstein), independent of the
modulation. A twin trained on full-square/full-circle excitation should predict
QAM transmission at a trained distance. BER is the application-level verdict.

Usage:
  modulation_check.py --model <train_dir> --modulation 4QAM_2026_V3_ORGANIZED \
    --dists 0.75,1.0,1.35,1.5 --currents 100,300,500,700 --n-cap 50000 \
    --label FS --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("CVAE_REPO", "/workspace/2026/feat_seq_bigru_residual_cvae"))


def _windows(X: np.ndarray, W: int) -> np.ndarray:
    """Centered edge-padded sliding windows: out[i] = X[i-h .. i+h], h=W//2."""
    h = W // 2
    Xp = np.pad(X, ((h, h), (0, 0)), mode="edge")
    idx = np.arange(len(X))[:, None] + np.arange(W)[None, :]
    return Xp[idx]  # (N, W, 2)


def _evm(x: np.ndarray, y: np.ndarray) -> float:
    """EVM% of y vs the ideal sent constellation x (per-axis LS gain removed)."""
    from scripts.analysis.compute_ber import equalize_axis
    err = np.zeros_like(x)
    for ax in (0, 1):
        a = equalize_axis(x[:, ax], y[:, ax])
        err[:, ax] = a * y[:, ax] - x[:, ax]
    return 100.0 * float(np.sqrt(np.mean(np.sum(err**2, 1)) / np.mean(np.sum(x**2, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="train dir with models/best_model_full.keras")
    ap.add_argument("--modulation", required=True, help="e.g. 4QAM_2026_V3_ORGANIZED")
    ap.add_argument("--data-root", default="/data/Dataset/V3")
    ap.add_argument("--dists", default="0.75,1.0,1.35,1.5")
    ap.add_argument("--currents", default="100,300,500,700")
    ap.add_argument("--n-cap", type=int, default=50000)
    ap.add_argument("--label", default="FS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=33)
    args = ap.parse_args()

    import tensorflow as tf
    from src.models.cvae import create_inference_model_from_full
    from src.models.cvae_sequence import load_seq_model
    from src.data.normalization import apply_condition_norm, load_normalization_from_state
    from scripts.analysis.compute_ber import qam_ber
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(args.seed); tf.random.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    model_path = os.path.join(args.model, "models", "best_model_full.keras")
    vae = load_seq_model(model_path)
    inf = create_inference_model_from_full(vae, deterministic=False)  # stochastic: 1 noisy draw
    W = int(vae.get_layer("prior_net").inputs[0].shape[1])
    state = json.load(open(os.path.join(args.model, "state_run.json")))
    norm = load_normalization_from_state(state)
    print(f"modelo {args.label} | W={W} | norm={norm}")

    dists = [float(x) for x in args.dists.split(",")]
    currs = [int(x) for x in args.currents.split(",")]
    # map float distance -> real on-disk folder name (1.0 -> "1.0", not "%g"="1")
    import re as _re
    dist_dirname = {}
    for dd in glob.glob(f"{args.data_root}/{args.modulation}/dist_*"):
        m = _re.search(r"dist_([0-9.]+)m", os.path.basename(dd))
        if m:
            dist_dirname[float(m.group(1))] = m.group(1)
    rows = []
    for d in dists:
        ds = dist_dirname.get(d, "%g" % d)
        for c in currs:
            caps = sorted(glob.glob(f"{args.data_root}/{args.modulation}/dist_{ds}m/curr_{c}mA/*/"))
            if not caps:
                print(f"  ! sem dados {ds}m/{c}mA"); continue
            X = np.load(caps[0] + "IQ_data/X.npy")[: args.n_cap].astype(np.float32)
            Yr = np.load(caps[0] + "IQ_data/Y.npy")[: args.n_cap].astype(np.float32)
            Xw = _windows(X, W)
            Dn, Cn = apply_condition_norm(np.full(len(X), d), np.full(len(X), c), norm)
            Yt = inf.predict([Xw, Dn.reshape(-1, 1).astype(np.float32),
                              Cn.reshape(-1, 1).astype(np.float32)], batch_size=16384, verbose=0)
            Yt = np.asarray(Yt)[:, :2].astype(np.float32)
            br = qam_ber(X, Yr); bt = qam_ber(X, Yt)
            row = {"label": args.label, "modulation": args.modulation.split("_")[0],
                   "dist_m": d, "curr_mA": c, "n": len(X),
                   "ber_real": round(br["ber"], 5), "ber_twin": round(bt["ber"], 5),
                   "ber_abs_err": round(abs(br["ber"] - bt["ber"]), 5),
                   "evm_real_pct": round(_evm(X, Yr), 3), "evm_twin_pct": round(_evm(X, Yt), 3)}
            rows.append(row)
            print(f"  {ds}m/{c}mA: BER real={row['ber_real']:.4f} twin={row['ber_twin']:.4f} "
                  f"| EVM real={row['evm_real_pct']:.1f}% twin={row['evm_twin_pct']:.1f}%")

            # overlay: real vs twin constellation (subsample for plot)
            s = np.random.default_rng(0).choice(len(X), size=min(4000, len(X)), replace=False)
            fig, ax = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
            ax[0].plot(Yr[s, 0], Yr[s, 1], ".", ms=1, alpha=.3, color="#1f77b4")
            ax[0].set_title(f"{row['modulation']} REAL — {ds}m/{c}mA\nBER={row['ber_real']:.4f}")
            ax[1].plot(Yt[s, 0], Yt[s, 1], ".", ms=1, alpha=.3, color="#d62728")
            ax[1].set_title(f"cVAE {args.label} twin\nBER={row['ber_twin']:.4f}")
            for a in ax: a.set_aspect("equal"); a.grid(alpha=.2)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "overlays", f"{row['modulation']}_{ds}m_{c}mA_{args.label}.png"), dpi=110)
            plt.close()
            del X, Yr, Xw, Yt

    if rows:
        with open(os.path.join(args.out, f"ber_table_{args.label}_{rows[0]['modulation']}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nescrito: {args.out} ({len(rows)} regimes)")


if __name__ == "__main__":
    main()
