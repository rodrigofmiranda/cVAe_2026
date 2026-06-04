#!/usr/bin/env python3
"""Inventory of cVAE training runs on the full_square dataset.

Scans known output roots, classifies each run as good-basin (val_recon <= -4.6)
or bad-basin, and joins the protocol leaderboard (n_pass / gates) when present.
Prints a ranked table of the good-basin runs — the "best runs" report.

NOTE: host-specific roots (shared GPU box). /home/eduardo is not listable by the
rodrigo user, so eduardo's runs must be reached via an explicit root *below* it.
Override roots via argv. Different-geometry lines (full_circle / shape) are
excluded because their val_recon scale is not comparable to full_square.
"""
import json, os, csv, subprocess, sys

DEFAULT_ROOTS = [
    "/home/eduardo/cVAe_2026/outputs",
    "/home/rodrigo/cVAe_2026_full_square/outputs",
    "/home/rodrigo/cVAe_2026_mdn_return/outputs",
    "/home/rodrigo/cvae_repro_141943/outputs",
    "/home/rodrigo/repro_c622/outputs",
    "/home/rodrigo/cvae_repro_outputs",
    "/home/rodrigo/cvae_det_out",
]
GOOD = -4.6  # full_square good-basin threshold (bad basin never goes below ~-3.98)


def find_histories(roots):
    files = []
    for r in roots:
        if os.path.isdir(r):
            files += subprocess.run(
                ["find", r, "-name", "training_history.json", "-path", "*/logs/train/*"],
                capture_output=True, text=True,
            ).stdout.split()
    return sorted(set(files))


def leaderboard(exp):
    lb = os.path.join(exp, "tables", "protocol_leaderboard.csv")
    if not os.path.exists(lb):
        return None
    try:
        r = list(csv.DictReader(open(lb)))[0]
        g = "/".join(str(r.get("gate_g%d_pass" % i)) for i in range(1, 7))
        return r.get("n_pass"), r.get("n_fail"), g
    except Exception:
        return None


def main(roots):
    rows = []
    for f in find_histories(roots):
        try:
            h = json.load(open(f))
            v = [float(x) for x in h["history"]["val_recon_loss"]]
        except Exception:
            continue
        if not v:
            continue
        fin = min(v)
        if fin > GOOD:
            continue
        exp = f.replace("/logs/train/training_history.json", "")
        lb = leaderboard(exp)
        npass = lb[0] if lb else "-"
        gates = lb[2] if lb else "-"
        clone = "/".join(f.split("/")[2:4])
        rows.append((npass, round(fin, 3), len(v), str(h.get("tag", "?"))[:26], gates,
                     clone, os.path.basename(exp)))

    def key(r):
        try:
            return (-int(r[0]), r[1])
        except (TypeError, ValueError):
            return (1, r[1])

    rows.sort(key=key)
    print(f"# Best-runs inventory (full_square) — {len(rows)} good-basin runs (val_recon <= {GOOD})\n")
    print(f"{'n_pass':>6} {'min':>8} {'eps':>5}  {'gates G1..G6':>14}  {'tag':<26} clone / exp")
    print("-" * 120)
    for npass, fin, ne, tag, gates, clone, exp in rows:
        print(f"{str(npass):>6} {fin:>8} {ne:>5}  {gates:>14}  {tag:<26} {clone} / {exp}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_ROOTS)
