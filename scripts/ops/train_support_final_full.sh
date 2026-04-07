#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/train_support_final_full.sh <candidate> [extra protocol args...]

Candidates:
  e2   S27cov_lc0p25_tail95_t0p03_edge
  e3c  S27cov_geom3_edge_rt_covsoft_a1p5_tau0p82_tc0p45_wmax2p5_lc0p20

Defaults:
  - full data (no train/val caps)
  - protocol_default.json (12 regimes)
  - stat_tests enabled
  - stat_mode=quick

Environment overrides:
  DATASET_ROOT   Dataset root (default: repo-local full_square dataset)
  OUTPUT_BASE    Output base (default: outputs/support_ablation/final_full/<candidate>)
  STAT_MODE      quick|full (default: quick)

Examples:
  scripts/ops/train_support_final_full.sh e2
  scripts/ops/train_support_final_full.sh e3c --seed 42
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

CANDIDATE="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
shift || true

case "$CANDIDATE" in
  e2)
    GRID_PRESET="support_e2_edge_weight"
    GRID_TAG="S27cov_lc0p25_tail95_t0p03_edge"
    STAGE_DIR="e2_edge_s27"
    ;;
  e3c)
    GRID_PRESET="support_e3c_geom3_edge_decision"
    GRID_TAG="S27cov_geom3_edge_rt_covsoft_a1p5_tau0p82_tc0p45_wmax2p5_lc0p20"
    STAGE_DIR="e3c_covsoft_s27"
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown candidate: ${CANDIDATE}" >&2
    usage >&2
    exit 1
    ;;
esac

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs/support_ablation/final_full/${STAGE_DIR}}"
STAT_MODE="${STAT_MODE:-quick}"

CMD=(
  "$REPO_ROOT/scripts/ops/train.sh"
  --protocol configs/protocol_default.json
  --grid_preset "$GRID_PRESET"
  --grid_tag "$GRID_TAG"
  --max_grids 1
  --no_data_reduction
  --stat_tests
  --stat_mode "$STAT_MODE"
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running support finalist full-data candidate %s\n' "$CANDIDATE"
printf '  dataset_root=%s\n' "$DATASET_ROOT"
printf '  output_base=%s\n' "$OUTPUT_BASE"
printf '  grid_preset=%s\n' "$GRID_PRESET"
printf '  grid_tag=%s\n' "$GRID_TAG"
printf '  stat_mode=%s\n' "$STAT_MODE"

exec "${CMD[@]}"
