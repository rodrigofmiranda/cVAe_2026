#!/usr/bin/env python3
"""AWGN baseline vs cVAE twin — V3, from existing cross-dist eval columns.

The eval already fits a matched-AWGN channel (X + Gaussian noise, power matched
to Y-X) and stores its errors as baseline_* alongside cvae_*. A matched AWGN can
reproduce EVM/SNR and 2nd-moment scale, but NOT the residual SHAPE (skew/kurt)
nor the signal-dependent noise (rho_hetero). This quantifies where the
generative twin beats the naive AWGN model. Trained distances only (twin valid).
"""
from __future__ import annotations
import csv, glob, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np

TRAINED = {0.75, 1.0, 1.35, 1.5}
# (label, cvae_col, baseline_col, "lower is better")
METRICS = [
    ("EVM err", "cvae_rel_evm_error", "baseline_rel_evm_error"),
    ("sigma err", "cvae_mean_rel_sigma", "baseline_mean_rel_sigma"),
    ("cov err", "cvae_cov_rel_var", "baseline_cov_rel_var"),
    ("skew L2", "cvae_delta_skew_l2", "baseline_delta_skew_l2"),
    ("kurt L2", "cvae_delta_kurt_l2", "baseline_delta_kurt_l2"),
    ("PSD L2", "cvae_psd_l2", "baseline_psd_l2"),
]


def _hetero_err(r, who):
    try:
        return abs(float(r[f"{who}_rho_hetero_pred"]) - float(r[f"{who}_rho_hetero_real"]))
    except (KeyError, ValueError):
        return np.nan


def load(run_glob):
    p = sorted(glob.glob(run_glob))[-1]
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            d = float(r["dist_target_m"])
        except (KeyError, ValueError):
            continue
        if d in TRAINED:
            rows.append(r)
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/rodrigo/comparison_v3/awgn"
    os.makedirs(out, exist_ok=True)
    base = "/home/rodrigo/cVAe_2026_full_square_v3det/outputs"
    runs = {"FS": f"{base}/v3fs_crossdist_20260613/exp_*/tables/summary_by_regime.csv",
            "FC": f"{base}/v3fc_crossdist_20260613/exp_*/tables/summary_by_regime.csv"}

    summary = {}
    for lbl, g in runs.items():
        rows = load(g)
        m = {}
        for name, cc, bc in METRICS:
            cv = np.nanmean([float(r[cc]) for r in rows if r.get(cc) not in (None, "", "nan")])
            bl = np.nanmean([float(r[bc]) for r in rows if r.get(bc) not in (None, "", "nan")])
            m[name] = (cv, bl)
        m["hetero err"] = (np.nanmean([_hetero_err(r, "cvae") for r in rows]),
                           np.nanmean([_hetero_err(r, "baseline") for r in rows]))
        summary[lbl] = m

    # CSV
    names = [n for n, *_ in METRICS] + ["hetero err"]
    with open(os.path.join(out, "awgn_vs_cvae_v3.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "metric", "cVAE", "AWGN", "cVAE_melhor?", "razao_AWGN/cVAE"])
        for lbl, m in summary.items():
            for n in names:
                cv, bl = m[n]
                ratio = bl / cv if cv else float("inf")
                w.writerow([lbl, n, f"{cv:.4f}", f"{bl:.4f}", "sim" if cv < bl else "nao", f"{ratio:.2f}"])

    # figure: grouped bars per model, log-y, cVAE vs AWGN
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, lbl in zip(axes, ["FS", "FC"]):
        m = summary[lbl]; x = np.arange(len(names)); wbar = 0.38
        cvs = [m[n][0] for n in names]; bls = [m[n][1] for n in names]
        ax.bar(x - wbar/2, cvs, wbar, label="cVAE twin", color="#2ca02c")
        ax.bar(x + wbar/2, bls, wbar, label="AWGN baseline", color="#7f7f7f")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{lbl} — cVAE vs AWGN (menor = melhor, distancias treinadas)"); ax.legend(); ax.grid(alpha=.2, axis="y")
    plt.suptitle("Twin generativo vs canal AWGN casado — onde o cVAE ganha (forma/heterocedasticidade)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out, "awgn_vs_cvae_v3.png"), dpi=120); plt.close()

    print("=== AWGN vs cVAE (média, distâncias treinadas) ===")
    for lbl, m in summary.items():
        print(f"--- {lbl} ---")
        for n in names:
            cv, bl = m[n]
            print(f"  {n:12s}: cVAE={cv:.4f}  AWGN={bl:.4f}  -> {'cVAE' if cv<bl else 'AWGN'} ganha ({bl/cv if cv else 0:.1f}x)")
    print(f"\nescrito: {out}/awgn_vs_cvae_v3.{{csv,png}}")


if __name__ == "__main__":
    main()
