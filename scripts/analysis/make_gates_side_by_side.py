#!/usr/bin/env python3
"""Side-by-side per-regime gate PASS/FAIL table for two models (CSV + PNG).

Mirrors comparison/fc_vs_fs_side_by_side/gates_summary_FC_vs_FS.png: rows are
regimes, columns are per-gate PASS/FAIL for the LEFT and RIGHT models, cells
green (PASS) / red (FAIL). For the cross-distance study the UNSEEN distances
(inference targets) are tinted so the extrapolation rows stand out.

Usage:
  make_gates_side_by_side.py --left <run_dir> --left-label FS \
                             --right <run_dir> --right-label FC \
                             --out <prefix> [--unseen 0.9,1.16,1.25]

<run_dir> = an exp dir (…/exp_YYYYMMDD_HHMMSS) OR its parent; the newest
exp_*/tables/summary_by_regime.csv under it is used.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GATE_COLS = [("G1", "gate_g1"), ("G2", "gate_g2"), ("G3", "gate_g3"),
             ("G4", "gate_g4"), ("G5", "gate_g5"), ("stat", "gate_g6"),
             ("status", "validation_status_full")]
GREEN, RED, UNSEEN_TINT = "#c8e6c9", "#ef9a9a", "#fff3c4"


def _find_summary(run_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(run_dir, "exp_*/tables/summary_by_regime.csv")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(run_dir, "tables/summary_by_regime.csv")))
    if not cands:
        raise FileNotFoundError(f"summary_by_regime.csv não encontrado sob {run_dir}")
    return cands[-1]


def _passfail(row: dict, col: str) -> str:
    v = str(row.get(col, "")).strip().lower()
    if col == "validation_status_full":
        return "PASS" if v == "pass" else "FAIL"
    return "PASS" if v == "true" else "FAIL"


def _load(run_dir: str) -> dict:
    """regime_key -> {label fields + per-gate PASS/FAIL}."""
    out = {}
    for r in csv.DictReader(open(_find_summary(run_dir))):
        try:
            d = float(r["dist_target_m"]); c = int(float(r["curr_target_mA"]))
        except (KeyError, ValueError):
            continue
        rec = {"distance_m": d, "current_mA": c}
        for short, col in GATE_COLS:
            rec[short] = _passfail(r, col)
        out[(d, c)] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--left-label", default="L")
    ap.add_argument("--right", required=True)
    ap.add_argument("--right-label", default="R")
    ap.add_argument("--out", required=True, help="output prefix (writes .csv and .png)")
    ap.add_argument("--unseen", default="", help="comma distances tinted as inference targets")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    unseen = {float(x) for x in args.unseen.split(",") if x.strip()}
    L, R = _load(args.left), _load(args.right)
    keys = sorted(set(L) | set(R))
    Ll, Rl = args.left_label, args.right_label

    # CSV
    fields = (["regime", "distance", "current_mA", "unseen"]
              + [f"{Ll}_{s}" for s, _ in GATE_COLS]
              + [f"{Rl}_{s}" for s, _ in GATE_COLS])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for (d, c) in keys:
            dtag = ("%g" % d).replace(".", "p")
            row = {"regime": f"dist_{dtag}m__curr_{c}mA",
                   "distance": f"{d:g} m", "current_mA": c,
                   "unseen": "yes" if d in unseen else ""}
            for src, lab in ((L, Ll), (R, Rl)):
                rec = src.get((d, c), {})
                for s, _ in GATE_COLS:
                    row[f"{lab}_{s}"] = rec.get(s, "—")
            w.writerow(row)

    # PNG (matplotlib table)
    col_labels = (["distance", "current_mA"]
                  + [f"{Ll}_{s}" for s, _ in GATE_COLS]
                  + [f"{Rl}_{s}" for s, _ in GATE_COLS])
    cells, colors = [], []
    for (d, c) in keys:
        base = UNSEEN_TINT if d in unseen else "white"
        line = [f"{d:g} m", str(c)]
        cline = [base, base]
        for src in (L, R):
            rec = src.get((d, c), {})
            for s, _ in GATE_COLS:
                v = rec.get(s, "—")
                line.append(v)
                cline.append(GREEN if v == "PASS" else RED if v == "FAIL" else base)
        cells.append(line); colors.append(cline)

    nrows, ncols = len(cells), len(col_labels)
    fig, ax = plt.subplots(figsize=(max(18, ncols * 1.15), max(4, 0.42 * (nrows + 2))))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=col_labels, cellColours=colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.35)
    for j in range(ncols):
        tbl[0, j].set_text_props(weight="bold"); tbl[0, j].set_facecolor("#e0e0e0")
    title = args.title or f"Gate pass/fail por regime — {Ll} vs {Rl}"
    sub = "linhas destacadas = distâncias NÃO VISTAS (inferência vs gabarito real)" if unseen else \
          "cada modelo avaliado contra seu próprio dado"
    plt.suptitle(title + "\n" + sub, fontsize=13, weight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.out + ".png", dpi=130, bbox_inches="tight")
    print(f"escrito: {args.out}.csv  e  {args.out}.png  ({nrows} regimes, unseen={sorted(unseen)})")


if __name__ == "__main__":
    sys.exit(main())
