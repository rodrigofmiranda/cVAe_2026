#!/usr/bin/env python3
"""Compare per-epoch val_recon_loss across the 4 demo runs to test bit-identity."""
import json, glob, sys

def load(sub):
    pats = sorted(glob.glob(f"/home/rodrigo/cvae_det_out/{sub}/exp_*/logs/train/training_history.json"))
    if not pats:
        return None
    h = json.load(open(pats[-1]))
    return [float(x) for x in h["history"]["val_recon_loss"]]

runs = {t: load(f"demo_{t}") for t in ("ndet_a","ndet_b","det_a","det_b")}
for t, v in runs.items():
    print(f"{t:8s}: {v}")

def identical(a, b):
    if a is None or b is None: return None
    if len(a) != len(b): return False
    return all(repr(x) == repr(y) for x, y in zip(a, b))   # full-precision equality

print()
print(f"NON-deterministic twins identical?  ndet_a == ndet_b : {identical(runs['ndet_a'], runs['ndet_b'])}")
print(f"DETERMINISTIC twins identical?      det_a  == det_b  : {identical(runs['det_a'],  runs['det_b'])}")
