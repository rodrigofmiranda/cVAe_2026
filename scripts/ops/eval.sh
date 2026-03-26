#!/usr/bin/env bash
set -euo pipefail
# Canonical evaluation CLI wrapper.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
export PYTHONPATH="$REPO_ROOT"

# If RUN_DIR is set, use it; otherwise pick latest run
if [[ -n "${RUN_DIR:-}" ]]; then
    python -u -m src.evaluation.evaluate \
        --dataset_root "$DATASET_ROOT" \
        --run_dir      "$RUN_DIR"
else
    # Fallback: pick latest run directory in OUTPUT_BASE
    LATEST_RUN=$(ls -1d "$OUTPUT_BASE"/run_* 2>/dev/null | sort | tail -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "ERROR: No run_* directories found in $OUTPUT_BASE" >&2
        exit 1
    fi
    python -u -m src.evaluation.evaluate \
        --dataset_root "$DATASET_ROOT" \
        --run_dir      "$LATEST_RUN"
fi

# --- Commit 3H: smoke-test examples (uncomment to use) ---
# python -u -m src.evaluation.evaluate \
#     --dataset_root "$DATASET_ROOT" --run_dir /path/to/outputs/run_x \
#     --max_experiments 1 --max_samples_per_exp 200000
# python -u -m src.evaluation.evaluate \
#     --dataset_root "$DATASET_ROOT" --run_dir /path/to/outputs/run_x --dry_run
