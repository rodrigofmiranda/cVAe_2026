#!/usr/bin/env bash
set -euo pipefail
# Canonical evaluation CLI wrapper.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
BOOTSTRAP_SCRIPT="$REPO_ROOT/scripts/ops/container_bootstrap_python.sh"
AUTO_BOOTSTRAP_PYTHON="${CVAE_AUTO_BOOTSTRAP_PYTHON:-1}"

if [[ "$AUTO_BOOTSTRAP_PYTHON" == "1" && -f "$BOOTSTRAP_SCRIPT" ]]; then
  export CVAE_TF25_WORKDIR="${CVAE_TF25_WORKDIR:-$REPO_ROOT}"
  # shellcheck disable=SC1090
  source "$BOOTSTRAP_SCRIPT"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT"
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi

# If RUN_DIR is set, use it; otherwise pick latest run
if [[ -n "${RUN_DIR:-}" ]]; then
    "$PYTHON_BIN" -u -m src.evaluation.evaluate \
        --dataset_root "$DATASET_ROOT" \
        --run_dir      "$RUN_DIR"
else
    # Fallback: pick latest run directory in OUTPUT_BASE
    LATEST_RUN=$(ls -1d "$OUTPUT_BASE"/run_* 2>/dev/null | sort | tail -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "ERROR: No run_* directories found in $OUTPUT_BASE" >&2
        exit 1
    fi
    "$PYTHON_BIN" -u -m src.evaluation.evaluate \
        --dataset_root "$DATASET_ROOT" \
        --run_dir      "$LATEST_RUN"
fi

# --- Commit 3H: smoke-test examples (uncomment to use) ---
# python -u -m src.evaluation.evaluate \
#     --dataset_root "$DATASET_ROOT" --run_dir /path/to/outputs/run_x \
#     --max_experiments 1 --max_samples_per_exp 200000
# python -u -m src.evaluation.evaluate \
#     --dataset_root "$DATASET_ROOT" --run_dir /path/to/outputs/run_x --dry_run
