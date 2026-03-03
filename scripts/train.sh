#!/usr/bin/env bash
set -euo pipefail
# Commit 3G: uses CLI entrypoint (wraps monolith main()).
# Legacy direct call (still works):
#   python -u -m src.training.cvae_TRAIN_documented

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
export PYTHONPATH="$REPO_ROOT"

python -u -m src.training.train \
    --dataset_root "$DATASET_ROOT" \
    --output_base  "$OUTPUT_BASE"

# --- Commit 3H: smoke-test examples (uncomment to use) ---
# python -u -m src.training.train \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --max_epochs 2 --max_experiments 1 --max_samples_per_exp 200000
# python -u -m src.training.train \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" --dry_run
# --- Commit 3I: grid selection controls ---
# python -u -m src.training.train \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --max_epochs 2 --max_experiments 1 --max_grids 1
# python -u -m src.training.train \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --max_epochs 2 --grid_group "G1_core"
# python -u -m src.training.train \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --max_epochs 2 --grid_tag "lat4.*b0p001"
