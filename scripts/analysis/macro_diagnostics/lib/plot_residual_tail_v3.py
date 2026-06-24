#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer 0/2 — noise-structure map of the REAL channel per regime.

Two heatmaps per model over the (distance x current) grid:
  (left)  heteroscedastic slope Var(δ) vs |X|^2 — the REAL discriminator
          (shot noise: variance grows with amplitude; a homoscedastic AWGN
          channel cannot reproduce this). Hot cells = strongest amplitude
          dependence = where the Gaussian-AWGN assumption is most wrong.
  (right) excess kurtosis of the EQUALISED δ     — Gaussianity check. It is ~0
          everywhere (the V3 noise is Gaussian once the linear gain is removed),
          so the twin's distributional failures are about SCALE/heteroscedasticity,
          NOT tail shape.

Model-free: reads only regime_census.csv (equalised residual; see regime_census.py).
Replaces the per-regime KDE grid (comparison/plot_residual_kde_grid.py, V2) with
a compact regime-level map (no raw samples needed).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np


def _load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _grid(rows: list[dict], model: str, value_fn):
    dists = sorted({float(r["dist_m"]) for r in rows if r["model"] == model})
    currs = sorted({int(float(r["curr_mA"])) for r in rows if r["model"] == model})
    M = np.full((len(dists), len(currs)), np.nan)
    di = {d: i for i, d in enumerate(dists)}
    ci = {c: j for j, c in enumerate(currs)}
    for r in rows:
        if r["model"] != model:
            continue
        try:
            v = value_fn(r)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        M[di[float(r["dist_m"])], ci[int(float(r["curr_mA"]))]] = v
    return dists, currs, M


def _draw(ax, dists, currs, M, title, cmap, fmt):
    im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(currs)))
    ax.set_xticklabels(currs, fontsize=8)
    ax.set_yticks(range(len(dists)))
    ax.set_yticklabels([f"{d:g}" for d in dists], fontsize=9)
    ax.set_xlabel("Corrente (mA)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Distância (m)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=7, color="#111827")
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load(args.census)
    models = sorted({r["model"] for r in rows})

    fig, axes = plt.subplots(len(models), 2, figsize=(13.0, 4.4 * len(models)),
                             squeeze=False, facecolor="white")
    for i, model in enumerate(models):
        d, c, Mh = _grid(rows, model, lambda r: float(r["het_slope_canon"]))
        im0 = _draw(axes[i][0], d, c, Mh, f"{model} — Var(Y−X)~|X|² (mapa de ganho não-linear)", "magma_r", "{:.3f}")
        fig.colorbar(im0, ax=axes[i][0], fraction=0.046, pad=0.04)

        d, c, Mk = _grid(rows, model, lambda r: float(r["real_kurt_eq"]))
        im1 = _draw(axes[i][1], d, c, Mk, f"{model} — kurtose de δ equalizado (≈0 ⇒ ruído Gaussiano)", "coolwarm", "{:.2f}")
        fig.colorbar(im1, ax=axes[i][1], fraction=0.046, pad=0.04)

    fig.suptitle("Estrutura do canal real por regime — distorção determinística (esq., Var∝|X|², "
                 "discriminador do AWGN) vs ruído (dir., kurt≈0 ⇒ Gaussiano)",
                 fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"escrito: {args.out}")


if __name__ == "__main__":
    main()
