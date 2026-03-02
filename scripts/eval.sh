#!/usr/bin/env bash
set -euo pipefail
# Future modular CLI (Commit 1 keeps monolith as default)
# python -m src.evaluation.evaluate --run_dir outputs/run_...

export OUTPUT_BASE=${OUTPUT_BASE:-/workspace/2026/outputs}

python -u src/evaluation/analise_cvae_reviewed.py
