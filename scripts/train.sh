#!/usr/bin/env bash
set -euo pipefail
# Future modular CLI (Commit 1 keeps monolith as default)
# python -m src.training.train --config configs/train.yaml

export DATASET_ROOT=${DATASET_ROOT:-/workspace/2026/data/dataset_fullsquare_organized}
export OUTPUT_BASE=${OUTPUT_BASE:-/workspace/2026/outputs}

python -u src/training/cvae_TRAIN_documented.py
