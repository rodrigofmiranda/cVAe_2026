#!/usr/bin/env bash
set -euo pipefail

export DATASET_ROOT=${DATASET_ROOT:-/workspace/2026/data/dataset_fullsquare_organized}
export OUTPUT_BASE=${OUTPUT_BASE:-/workspace/2026/outputs}

python -u src/training/cvae_TRAIN_documented.py
