#!/usr/bin/env bash
set -euo pipefail
# Future modular CLI (Commit 1 keeps monolith as default)
# python -m src.evaluation.evaluate --run_dir outputs/run_...

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
export PYTHONPATH="$REPO_ROOT"

python -u -m src.evaluation.analise_cvae_reviewed
