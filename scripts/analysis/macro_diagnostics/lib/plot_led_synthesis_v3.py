#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer 3 — physical synthesis: LED regime ↔ noise structure.

Three stacked panels sharing x = bias current (mA), one line per distance:
  (top)    real channel SNR  snr(X, Y_real)            — link budget
  (mid)    heteroscedastic slope Var(δ) vs |X|^2       — shot-noise amplitude
           dependence; what a homoscedastic AWGN channel cannot match
  (bottom) excess kurtosis of the EQUALISED residual δ — Gaussianity check
           (≈0 everywhere ⇒ V3 noise is Gaussian after gain removal)

LED operating regions (joelho / quasi-linear / droop) are shaded across panels.
Corrected physical reading: the noise is Gaussian (bottom ≈ 0), so the AWGN
baseline does NOT fail on tails — it fails because it is homoscedastic (flat),
missing the amplitude-dependent variance shown in the middle panel.

Model-free: reads only regime_census.csv (no model, no xlsx required).
Ported/condensed from comparison/plot_led_synthesis.py (V2, frozen).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

DIST_COLORS = {
    0.75: "#0ea5e9", 0.9: "#38bdf8", 1.0: "#2563eb", 1.16: "#6366f1",
    1.25: "#7c3aed", 1.35: "#1e3a8a", 1.5: "#0f172a",
}
MODEL_LS = {"FC": "-", "FS": "--"}
MODEL_MK = {"FC": "o", "FS": "^"}
REGIONS = [(60, 250, "#fde68a", "Joelho"),
           (250, 700, "#bbf7d0", "Quasi-linear"),
           (700, 1000, "#fecaca", "Droop / térmico")]


def _load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load(args.census)
    # index: (model, dist) -> [(curr, snr, het, kurt)]
    series: dict[tuple, list[tuple]] = defaultdict(list)
    for r in rows:
        try:
            series[(r["model"], float(r["dist_m"]))].append(
                (int(float(r["curr_mA"])), float(r["real_channel_snr_db"]),
                 float(r["het_slope_canon"]), float(r["real_kurt_eq"])))
        except (KeyError, ValueError):
            continue
    for k in series:
        series[k].sort()

    dists = sorted({float(r["dist_m"]) for r in rows})
    models = sorted({r["model"] for r in rows})
    sampled = sorted({int(float(r["curr_mA"])) for r in rows})

    fig, (ax_snr, ax_het, ax_kurt) = plt.subplots(3, 1, figsize=(13.0, 12.5), sharex=True,
                                                  gridspec_kw={"hspace": 0.10}, facecolor="white")
    panels = (ax_snr, ax_het, ax_kurt)

    for x0, x1, color, label in REGIONS:
        for ax in panels:
            ax.axvspan(x0, x1, color=color, alpha=0.22, zorder=0)
        ax_snr.text(0.5 * (x0 + x1), 0.96, label, transform=ax_snr.get_xaxis_transform(),
                    ha="center", va="top", fontsize=10, fontweight="bold", color="#1f2937",
                    bbox=dict(facecolor="white", edgecolor=color, lw=0.8, alpha=0.92, pad=2.5))
    for c in sampled:
        for ax in panels:
            ax.axvline(c, color="#374151", lw=0.6, ls=":", alpha=0.45, zorder=1)

    for model in models:
        for dist in dists:
            pts = series.get((model, dist))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            color = DIST_COLORS.get(dist, "#334155")
            kw = dict(color=color, ls=MODEL_LS.get(model, "-"), lw=2.0,
                      marker=MODEL_MK.get(model, "o"), markersize=5.5,
                      markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7)
            ax_snr.plot(xs, [p[1] for p in pts], label=f"{model} {dist:g} m", **kw)
            ax_het.plot(xs, [p[2] for p in pts], **kw)
            ax_kurt.plot(xs, [p[3] for p in pts], **kw)

    ax_kurt.axhline(0.0, color="#b91c1c", lw=1.4, ls="--", alpha=0.85, zorder=3)
    ax_kurt.text(sampled[-1], 0.0, " Gaussiano (kurt=0)", color="#b91c1c",
                 fontsize=9.5, va="bottom", ha="right", fontweight="bold")

    ax_snr.set_ylabel("SNR físico do canal (dB)\n= snr(X, Y_real)", fontsize=11, fontweight="bold")
    ax_het.set_ylabel("Var(Y−X) ~ |X|²\n(mapa de ganho não-linear)", fontsize=11, fontweight="bold")
    ax_kurt.set_ylabel("Excess kurtosis de δ\n(equalizado ⇒ ≈0)", fontsize=11, fontweight="bold")
    ax_kurt.set_xlabel("Corrente de bias (mA)", fontsize=11, fontweight="bold")
    ax_snr.set_title("Síntese física — regime do LED ↔ estrutura do ruído "
                     "(falha do AWGN = heterocedasticidade, não cauda)",
                     fontsize=12.5, fontweight="bold", color="#1f2937", pad=10)
    for ax in panels:
        ax.grid(True, alpha=0.25, lw=0.5)
    ax_snr.legend(loc="upper right", fontsize=8.5, ncol=2, frameon=True, framealpha=0.95)

    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"escrito: {args.out}")


if __name__ == "__main__":
    main()
