#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/16qam}"
OUT_BASE="${OUT_BASE:-$REPO_ROOT/outputs/architectures/_adhoc}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/_launch_logs}"
STAT_TESTS="${STAT_TESTS:-0}"
STAT_MODE="${STAT_MODE:-quick}"
STAT_N_PERM="${STAT_N_PERM:-}"
STAT_MAX_N="${STAT_MAX_N:-}"
STAT_SEED="${STAT_SEED:-42}"

if (( $# > 0 )); then
  CURRENTS=("$@")
else
  CURRENTS=(100 300 500 700)
fi

mkdir -p "$OUT_BASE" "$LOG_DIR"

EXTRA_ARGS=()
if [[ "$STAT_TESTS" == "1" ]]; then
  EXTRA_ARGS+=(--stat_tests --stat_mode "$STAT_MODE" --stat_seed "$STAT_SEED")
  if [[ -n "$STAT_N_PERM" ]]; then
    EXTRA_ARGS+=(--stat_n_perm "$STAT_N_PERM")
  fi
  if [[ -n "$STAT_MAX_N" ]]; then
    EXTRA_ARGS+=(--stat_max_n "$STAT_MAX_N")
  fi
fi

run_one() {
  local label="$1"
  local model_run_dir="$2"
  local log_path="$LOG_DIR/run_eval_16qam_${label}_$(date -u +%Y%m%d_%H%M%S).log"
  local out_root="$OUT_BASE/$label/16qam/sel4curr"

  echo "[triplet] label=$label"
  echo "[triplet] model_run_dir=$model_run_dir"
  echo "[triplet] out_root=$out_root"
  echo "[triplet] log=$log_path"

  "$PYTHON_BIN" -u "$REPO_ROOT/scripts/analysis/run_eval_16qam_all_regimes.py" \
    --repo_root "$REPO_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --model_run_dir "$model_run_dir" \
    --out_root "$out_root" \
    --currents_mA "${CURRENTS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$log_path"

  echo "[triplet] done label=$label"
  echo
}

run_one \
  "e0_anchor_s27" \
  "$REPO_ROOT/outputs/support_ablation/e0/exp_20260406_164913/train"

run_one \
  "e2_edge_s27" \
  "$REPO_ROOT/outputs/support_ablation/e2/exp_20260406_203946/train"

run_one \
  "e3c_covsoft_s27" \
  "$REPO_ROOT/outputs/support_ablation/e3c/exp_20260407_113203/train"
