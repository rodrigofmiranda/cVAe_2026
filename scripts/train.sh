#!/usr/bin/env bash
set -euo pipefail
# Future modular CLI (Commit 1 keeps monolith as default)
# python -m src.training.train --config configs/train.yaml

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
export PYTHONPATH="$REPO_ROOT"

python -u -m src.training.cvae_TRAIN_documented
