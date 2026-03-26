#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick real test: 2 grid variants × 9 experiments (dist=1.0m, all currents).

Usage:
    python scripts/test_dist1m_2grid.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loading import discover_experiments, parse_dist_curr_from_path

# ── 1. Find dist=1.0m experiments ──────────────────────────
dataset_root = Path("data/dataset_fullsquare_organized").resolve()
all_exps = discover_experiments(dataset_root, verbose=False)
sel_exps = [
    str(e) for e in all_exps
    if parse_dist_curr_from_path(e)[0] == 1.0
]
print(f"Selected {len(sel_exps)} experiments at dist=1.0m")
for p in sel_exps:
    print(f"  {p}")

# ── 2. Overrides: 2 grid variants (G0_ref), dist=1.0m only ─
overrides = {
    # Grid: only 2 reference models
    "grid_group": "G0_ref",
    "max_grids": 2,
    # Experiment filter: only dist=1.0m
    "_selected_experiments": sel_exps,
    # Keras verbosity
    "keras_verbose": 2,
}

# ── 3. Launch training ────────────────────────────────────
from src.training.engine import train_engine

summary = train_engine(
    dataset_root=str(dataset_root),
    output_base="outputs",
    run_id="test_dist1m_2grid",
    overrides=overrides,
)

print(f"\n{'='*60}")
print(f"Status: {summary['status']}")
print(f"Run dir: {summary.get('run_dir', 'N/A')}")
print(f"{'='*60}")
