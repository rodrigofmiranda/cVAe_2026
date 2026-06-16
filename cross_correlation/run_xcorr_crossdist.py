#!/usr/bin/env python3
"""Cross-correlation channel-ID driver: R_xy(tau) real vs twin across distances.

For a base twin (FS or FC), at each distance (trained + unseen), computes the
input-output cross-correlation R_xy(tau) of the REAL channel and of the twin's
DETERMINISTIC conditional mean E[y|x] (isolates the linear response from noise),
per axis I/Q. Reports the discrepancy (L2, peak-lag), the input whiteness
(validity precondition R_xx~=delta), and a figure. See the scientific basis in
TESE/06_validacao_do_gemeo/cross_correlation_fundamento_2026-06-13.md.

Light: model loaded once, one capped regime at a time.

Usage:
  run_xcorr_crossdist.py --model <train_dir> --data <DATASET_ORGANIZED dir name>
    --label FS --currents 500 --n-cap 60000 --max-lag 48 --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("CVAE_REPO", "/workspace/2026/feat_seq_bigru_residual_cvae"))


def _windows(X, W):
    h = W // 2
    Xp = np.pad(X, ((h, h), (0, 0)), mode="edge")
    idx = np.arange(len(X))[:, None] + np.arange(W)[None, :]
    return Xp[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="modulation/geom dir name under /data/Dataset/V3")
    ap.add_argument("--data-root", default="/data/Dataset/V3")
    ap.add_argument("--label", default="FS")
    ap.add_argument("--trained", default="0.75,1.0,1.35,1.5")
    ap.add_argument("--currents", default="500")
    ap.add_argument("--n-cap", type=int, default=60000)
    ap.add_argument("--max-lag", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import tensorflow as tf
    from src.models.cvae import create_inference_model_from_full
    from src.models.cvae_sequence import load_seq_model
    from src.data.normalization import apply_condition_norm, load_normalization_from_state
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from compute_xcorr import normalized_xcorr, xcorr_discrepancy, input_whiteness
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    vae = load_seq_model(os.path.join(args.model, "models", "best_model_full.keras"))
    inf = create_inference_model_from_full(vae, deterministic=True)  # conditional mean E[y|x]
    W = int(vae.get_layer("prior_net").inputs[0].shape[1])
    norm = load_normalization_from_state(json.load(open(os.path.join(args.model, "state_run.json"))))

    trained = {float(x) for x in args.trained.split(",")}
    # discover all distances on disk
    dist_dirname = {}
    for dd in glob.glob(f"{args.data_root}/{args.data}/dist_*"):
        m = re.search(r"dist_([0-9.]+)m", os.path.basename(dd))
        if m:
            dist_dirname[float(m.group(1))] = m.group(1)
    dists = sorted(dist_dirname)
    currs = [int(x) for x in args.currents.split(",")]
    L = args.max_lag

    rows, curves = [], {}
    for d in dists:
        for c in currs:
            caps = sorted(glob.glob(f"{args.data_root}/{args.data}/dist_{dist_dirname[d]}m/curr_{c}mA/*/"))
            if not caps:
                continue
            X = np.load(caps[0] + "IQ_data/X.npy")[: args.n_cap].astype(np.float32)
            Yr = np.load(caps[0] + "IQ_data/Y.npy")[: args.n_cap].astype(np.float32)
            Dn, Cn = apply_condition_norm(np.full(len(X), d), np.full(len(X), c), norm)
            Yt = np.asarray(inf.predict([_windows(X, W), Dn.reshape(-1, 1).astype(np.float32),
                                         Cn.reshape(-1, 1).astype(np.float32)],
                                        batch_size=16384, verbose=0))[:, :2]
            wx = input_whiteness(X[:, 0], L)
            seen = "treinada" if d in trained else "NAO-vista"
            row = {"label": args.label, "dist_m": d, "curr_mA": c, "seen": seen,
                   "input_whiteness": round(wx, 4)}
            for ax, axn in ((0, "I"), (1, "Q")):
                rr = normalized_xcorr(X[:, ax], Yr[:, ax], L)
                rt = normalized_xcorr(X[:, ax], Yt[:, ax], L)
                disc = xcorr_discrepancy(rr, rt)
                row[f"xcorr_l2_{axn}"] = round(disc["xcorr_l2"], 5)
                row[f"peak_lag_err_{axn}"] = disc["peak_lag_err"]
                if c == currs[0] and ax == 0:
                    curves[d] = (rr, rt, seen)
            rows.append(row)
            print(f"  {dist_dirname[d]}m/{c}mA [{seen}]: xcorr_l2 I={row['xcorr_l2_I']} Q={row['xcorr_l2_Q']} "
                  f"| peak_lag_err I={row['peak_lag_err_I']} | whiteness={row['input_whiteness']}")
            del X, Yr, Yt

    if rows:
        with open(os.path.join(args.out, f"xcorr_table_{args.label}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # figure: R_xy real vs twin per distance (axis I), unseen highlighted
    if curves:
        lags = np.arange(-L, L + 1)
        nd = len(curves)
        fig, axes = plt.subplots(1, nd, figsize=(2.6 * nd, 3.2), sharey=True)
        if nd == 1: axes = [axes]
        for axp, d in zip(axes, sorted(curves)):
            rr, rt, seen = curves[d]
            axp.plot(lags, rr, color="#1f77b4", lw=1.5, label="real")
            axp.plot(lags, rt, color="#d62728", lw=1.2, ls="--", label="twin")
            ttl = f"{d:g} m" + ("  (NAO-vista)" if seen != "treinada" else "")
            axp.set_title(ttl, fontsize=9, color=("#b00" if seen != "treinada" else "k"))
            axp.grid(alpha=.2); axp.set_xlabel("lag")
        axes[0].set_ylabel("R_xy(τ)"); axes[0].legend(fontsize=8)
        plt.suptitle(f"Cross-correlation entrada-saida R_xy(τ) — {args.label} (real vs twin)\n"
                     f"twin reproduz a resposta linear do canal?", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(os.path.join(args.out, f"xcorr_curves_{args.label}.png"), dpi=120)
        plt.close()
    print(f"\nescrito: {args.out} ({len(rows)} regimes)")


if __name__ == "__main__":
    main()
