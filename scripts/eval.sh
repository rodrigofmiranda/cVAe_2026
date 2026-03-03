#!/usr/bin/env bash
set -euo pipefail
# Commit 3G: uses CLI entrypoint (wraps monolith main()).
# Legacy direct call (still works):
#   python -u -m src.evaluation.analise_cvae_reviewed

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
    # Fallback: pick latest run via the monolith's own pick_latest_run logic
    LATEST_RUN=$(ls -1d "$OUTPUT_BASE"/run_* 2>/dev/null | sort | tail -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "ERROR: No run_* directories found in $OUTPUT_BASE" >&2
        exit 1
    fi
    python -u -m src.evaluation.evaluate \
        --dataset_root "$DATASET_ROOT" \
        --run_dir      "$LATEST_RUN"
fi
