#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/train_support_ablation.sh <stage> [extra protocol args...]

Stages:
  e0   support_e0_baselines
  e1   support_e1_geom3
  e2   support_e2_edge_weight
  e3   support_e3_geom3_edge_weight
  e3b  support_e3b_geom3_edge_retune
  e3c  support_e3c_geom3_edge_decision
  e4   support_e4_disk
  e5   confirmation run; requires SUPPORT_E5_GRID_TAG or explicit --grid_tag

Notes:
  - E0-E4 are mixed-family exploratory sweeps by default.
  - E3b is a short S27-only local retune around the E3 winner.
  - E3c is a 2-candidate decision run after E3b.
  - For controlled paired comparisons, pin a single candidate with:
      --grid_tag <tag> --max_grids 1
    Example:
      scripts/ops/train_support_ablation.sh e1 \
        --grid_tag S27cov_lc0p25_tail95_t0p03_geom3 --max_grids 1 --seed 42
  - E5 is intentionally manual because it depends on the E0-E4 findings.

Environment overrides:
  DATASET_ROOT            Dataset root (default: repo-local full_square dataset)
  OUTPUT_BASE             Output base (default: outputs/support_ablation/<stage>)
  MAX_SAMPLES_PER_EXP     Default 100000
  STAT_MODE               Default quick
  SUPPORT_DIAG_BINS       Default 4
  SUPPORT_E5_PRESET       Default support_e3_geom3_edge_weight
  SUPPORT_E5_GRID_TAG     Required for e5 unless passed via --grid_tag

Examples:
  scripts/ops/train_support_ablation.sh e0
  scripts/ops/train_support_ablation.sh e4 --max_regimes 4
  SUPPORT_E5_GRID_TAG="S27cov_lc0p25_tail95_t0p03_geom3_edge" \
    scripts/ops/train_support_ablation.sh e5 --seed 42
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

STAGE="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
shift || true

case "$STAGE" in
  e0)
    GRID_PRESET="support_e0_baselines"
    STAGE_DIR="e0"
    EXTRA_ARGS=()
    ;;
  e1)
    GRID_PRESET="support_e1_geom3"
    STAGE_DIR="e1"
    EXTRA_ARGS=()
    ;;
  e2)
    GRID_PRESET="support_e2_edge_weight"
    STAGE_DIR="e2"
    EXTRA_ARGS=()
    ;;
  e3)
    GRID_PRESET="support_e3_geom3_edge_weight"
    STAGE_DIR="e3"
    EXTRA_ARGS=()
    ;;
  e3b)
    GRID_PRESET="support_e3b_geom3_edge_retune"
    STAGE_DIR="e3b"
    EXTRA_ARGS=()
    ;;
  e3c)
    GRID_PRESET="support_e3c_geom3_edge_decision"
    STAGE_DIR="e3c"
    EXTRA_ARGS=()
    ;;
  e4)
    GRID_PRESET="support_e4_disk"
    STAGE_DIR="e4"
    EXTRA_ARGS=()
    ;;
  e5)
    GRID_PRESET="${SUPPORT_E5_PRESET:-support_e3_geom3_edge_weight}"
    STAGE_DIR="e5"
    EXTRA_ARGS=()
    if [[ -n "${SUPPORT_E5_GRID_TAG:-}" ]]; then
      EXTRA_ARGS+=(--grid_tag "${SUPPORT_E5_GRID_TAG}" --max_grids 1)
    fi
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    usage >&2
    exit 1
    ;;
esac

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs/support_ablation/${STAGE_DIR}}"

MAX_SAMPLES_PER_EXP="${MAX_SAMPLES_PER_EXP:-100000}"
STAT_MODE="${STAT_MODE:-quick}"
SUPPORT_DIAG_BINS="${SUPPORT_DIAG_BINS:-4}"

if [[ "$STAGE" == "e5" ]]; then
  HAS_GRID_TAG=0
  for arg in "$@"; do
    if [[ "$arg" == "--grid_tag" ]]; then
      HAS_GRID_TAG=1
      break
    fi
  done
  if [[ -z "${SUPPORT_E5_GRID_TAG:-}" && "$HAS_GRID_TAG" -eq 0 ]]; then
    echo "e5 requires SUPPORT_E5_GRID_TAG or explicit --grid_tag." >&2
    exit 1
  fi
fi

CMD=(
  "$REPO_ROOT/scripts/ops/train.sh"
  --protocol configs/protocol_default.json
  --grid_preset "$GRID_PRESET"
  --no_data_reduction
  --max_samples_per_exp "$MAX_SAMPLES_PER_EXP"
  --stat_tests
  --stat_mode "$STAT_MODE"
  --support_filter_eval_mode matched_support_and_full
  --support_diag_bins "$SUPPORT_DIAG_BINS"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running support ablation stage %s\n' "$STAGE"
printf '  dataset_root=%s\n' "$DATASET_ROOT"
printf '  output_base=%s\n' "$OUTPUT_BASE"
printf '  grid_preset=%s\n' "$GRID_PRESET"
printf '  extra_args=%s\n' "${EXTRA_ARGS[*]:-<none>}"

exec "${CMD[@]}"
