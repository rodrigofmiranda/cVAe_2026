#!/usr/bin/env bash
set -euo pipefail
# Canonical protocol wrapper for shared-global training/evaluation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
export PYTHONPATH="$REPO_ROOT"

python -u -m src.protocol.run \
    --dataset_root "$DATASET_ROOT" \
    --output_base  "$OUTPUT_BASE" \
    --train_once_eval_all \
    "$@"

# --- Smoke-test examples (uncomment to use) ---
# python -u -m src.protocol.run \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --train_once_eval_all \
#     --max_epochs 2 --max_regimes 1 --max_experiments 1 --max_samples_per_exp 200000
# python -u -m src.protocol.run \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" --dry_run
# --- Grid selection controls ---
# python -u -m src.protocol.run \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --train_once_eval_all \
#     --max_epochs 2 --max_regimes 1 --max_grids 1
# python -u -m src.protocol.run \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --train_once_eval_all \
#     --max_epochs 2 --grid_group "G1_core"
# python -u -m src.protocol.run \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --train_once_eval_all \
#     --max_epochs 2 --grid_tag "lat4.*b0p001"
# --- Keras verbosity control ---
# python -u -m src.protocol.run \
#     --dataset_root "$DATASET_ROOT" --output_base "$OUTPUT_BASE" \
#     --train_once_eval_all \
#     --max_epochs 1 --max_grids 1 --keras_verbose 2
