#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
SHAPE_REPO_ROOT="${SHAPE_REPO_ROOT:-$PARENT_ROOT/cVAe_2026_shape}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/16qam}"
OUT_BASE="${OUT_BASE:-$REPO_ROOT/outputs/analysis/eval_16qam_crossline_20260422_plus_soft_radial}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/_launch_logs}"

STAT_TESTS="${STAT_TESTS:-1}"
STAT_MODE="${STAT_MODE:-quick}"
STAT_N_PERM="${STAT_N_PERM:-}"
STAT_MAX_N="${STAT_MAX_N:-}"
STAT_SEED="${STAT_SEED:-42}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

SOFT_LABEL="${SOFT_LABEL:-full_circle_soft_rinf_local}"
SOFT_MODEL_RUN_DIR="${SOFT_MODEL_RUN_DIR:-$REPO_ROOT/outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256/train}"
SOFT_OUT_ROOT="${SOFT_OUT_ROOT:-$OUT_BASE/$SOFT_LABEL}"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-$OUT_BASE/crossline_summary}"

FULL_SQUARE_EVAL_ROOT="${FULL_SQUARE_EVAL_ROOT:-$SHAPE_REPO_ROOT/outputs/analysis/eval_16qam_crossline_20260420_clean/full_square_s27cov_sciv1_lr0p00015}"
FULL_CIRCLE_CLEAN_EVAL_ROOT="${FULL_CIRCLE_CLEAN_EVAL_ROOT:-$REPO_ROOT/outputs/analysis/eval_16qam_crossline_20260420_clean/full_circle_clean_lat10}"
FULL_CIRCLE_DISK_EVAL_ROOT="${FULL_CIRCLE_DISK_EVAL_ROOT:-$REPO_ROOT/outputs/analysis/eval_16qam_crossline_20260420_clean/full_circle_disk_geom3}"

if (( $# > 0 )); then
  CURRENTS=("$@")
else
  CURRENTS=(100 300 500 700)
fi

mkdir -p "$OUT_BASE" "$LOG_DIR" "$SUMMARY_OUT_DIR"

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

if [[ "$SKIP_EVAL" != "1" ]]; then
  LOG_PATH="$LOG_DIR/run_eval_16qam_${SOFT_LABEL}_$(date -u +%Y%m%d_%H%M%S).log"
  echo "[crossline+soft] label=$SOFT_LABEL"
  echo "[crossline+soft] model_run_dir=$SOFT_MODEL_RUN_DIR"
  echo "[crossline+soft] out_root=$SOFT_OUT_ROOT"
  echo "[crossline+soft] log=$LOG_PATH"

  "$PYTHON_BIN" -u "$REPO_ROOT/scripts/analysis/run_eval_16qam_all_regimes.py" \
    --repo_root "$REPO_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --model_run_dir "$SOFT_MODEL_RUN_DIR" \
    --out_root "$SOFT_OUT_ROOT" \
    --currents_mA "${CURRENTS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_PATH"
fi

if [[ "$SKIP_COMPARE" != "1" ]]; then
  "$PYTHON_BIN" -u "$REPO_ROOT/scripts/analysis/compare_eval_16qam_crossline.py" \
    --title "16QAM Crossline Comparison Extended With Soft-Radial" \
    --out_dir "$SUMMARY_OUT_DIR" \
    --candidate "full_square=$FULL_SQUARE_EVAL_ROOT" \
    --candidate "full_circle_clean=$FULL_CIRCLE_CLEAN_EVAL_ROOT" \
    --candidate "full_circle_geometry_biased=$FULL_CIRCLE_DISK_EVAL_ROOT" \
    --candidate "full_circle_geometry_light=$SOFT_OUT_ROOT"
fi
