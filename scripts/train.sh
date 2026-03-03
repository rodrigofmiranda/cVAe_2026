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
