#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Option 1b — variance-budget decomposition of the apparent residual Y-X.

Per model, two panels that answer the training-redesign question directly:
  (left)  stacked budget vs current (mean over distances): how much of Var(Y-X)
          is deterministic LINEAR gain, extra NONLINEAR gain (saturation), and
          irreducible NOISE. If the nonlinear slice grows with current, the LED
          saturation is real and amplitude-conditioning/capacity is the lever.
  (right) heatmap of the NONLINEAR fraction over (distance x current): WHERE the
          static nonlinear map dominates — the regimes to give capacity / oversample.

A small nonlinear slice + Gaussian final residual => the per-regime difficulty is
gain/interpolation, not noise tails => chasing MDN tail capacity is misdirected.

Model-free: reads only regime_census.csv (columns frac_linear/frac_nonlinear/frac_noise)."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

C_LIN, C_NL, C_NOISE = "#2563eb", "#ea580c", "#94a3b8"


def _load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _budget_by_current(rows, model):
    by_c: dict[int, list[tuple]] = defaultdict(list)
    for r in rows:
        if r["model"] != model:
            continue
        try:
            by_c[int(float(r["curr_mA"]))].append(
                (float(r["frac_linear"]), float(r["frac_nonlinear"]), float(r["frac_noise"])))
        except (KeyError, ValueError):
            continue
    currs = sorted(by_c)
    lin = [np.mean([t[0] for t in by_c[c]]) for c in currs]
    nl = [np.mean([t[1] for t in by_c[c]]) for c in currs]
    nz = [np.mean([t[2] for t in by_c[c]]) for c in currs]
    return currs, np.array(lin), np.array(nl), np.array(nz)


def _nl_grid(rows, model):
    dists = sorted({float(r["dist_m"]) for r in rows if r["model"] == model})
    currs = sorted({int(float(r["curr_mA"])) for r in rows if r["model"] == model})
    M = np.full((len(dists), len(currs)), np.nan)
    di = {d: i for i, d in enumerate(dists)}
    ci = {c: j for j, c in enumerate(currs)}
    for r in rows:
        if r["model"] != model:
            continue
        try:
            M[di[float(r["dist_m"])], ci[int(float(r["curr_mA"]))]] = float(r["frac_nonlinear"])
        except (KeyError, ValueError):
            continue
    return dists, currs, M


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
    fig, axes = plt.subplots(len(models), 2, figsize=(13.5, 4.6 * len(models)),
                             squeeze=False, facecolor="white")

    for i, model in enumerate(models):
        currs, lin, nl, nz = _budget_by_current(rows, model)
        ax = axes[i][0]
        ax.bar(range(len(currs)), lin, color=C_LIN, label="ganho linear", width=0.8)
        ax.bar(range(len(currs)), nl, bottom=lin, color=C_NL, label="extra não-linear", width=0.8)
        ax.bar(range(len(currs)), nz, bottom=lin + nl, color=C_NOISE, label="ruído irredutível", width=0.8)
        ax.set_xticks(range(len(currs)))
        ax.set_xticklabels(currs, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Corrente (mA)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Fração de Var(Y−X)", fontsize=10, fontweight="bold")
        ax.set_title(f"{model} — orçamento de variância vs corrente", fontsize=11, fontweight="bold")
        if i == 0:
            ax.legend(fontsize=8.5, loc="lower center", ncol=3, framealpha=0.95)

        dists, currs2, M = _nl_grid(rows, model)
        ax2 = axes[i][1]
        im = ax2.imshow(M, aspect="auto", cmap="Oranges", origin="lower", vmin=0)
        ax2.set_xticks(range(len(currs2))); ax2.set_xticklabels(currs2, fontsize=8)
        ax2.set_yticks(range(len(dists))); ax2.set_yticklabels([f"{d:g}" for d in dists], fontsize=9)
        ax2.set_xlabel("Corrente (mA)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Distância (m)", fontsize=10, fontweight="bold")
        ax2.set_title(f"{model} — fração não-linear por regime", fontsize=11, fontweight="bold")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if np.isfinite(M[a, b]):
                    ax2.text(b, a, f"{M[a,b]*100:.0f}", ha="center", va="center", fontsize=6.5, color="#111827")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Decomposição do resíduo aparente Y−X (opção 1b) — linear vs não-linear vs ruído. "
                 "Define o alvo do reprojeto: mapa determinístico vs cauda de ruído.",
                 fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"escrito: {args.out}")


if __name__ == "__main__":
    main()
