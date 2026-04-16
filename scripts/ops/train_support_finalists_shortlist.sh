#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/ops/train_support_finalists_shortlist.sh [extra protocol args...]

Purpose:
  Run the final short-list comparison after the scientific screening.

Candidates:
  - E2 control
  - tail98
  - lr0p00015

Defaults:
  - protocol_default.json (12 regimes)
  - full data (no train/val caps)
  - preset: support_e2_finalists_shortlist_v1
  - stat_tests enabled
  - stat_mode=quick
  - patience=50
  - reduce_lr_patience=25

Environment overrides:
  DATASET_ROOT         Dataset root (default: repo-local full_square dataset)
  OUTPUT_BASE          Output base (default: outputs/support_ablation/final_grid/e2_finalists_shortlist)
  STAT_MODE            quick|full (default: quick)
  PATIENCE             Early stopping patience (default: 50)
  REDUCE_LR_PATIENCE   LR plateau patience (default: 25)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs/support_ablation/final_grid/e2_finalists_shortlist}"
STAT_MODE="${STAT_MODE:-quick}"
PATIENCE="${PATIENCE:-50}"
REDUCE_LR_PATIENCE="${REDUCE_LR_PATIENCE:-25}"

CMD=(
  "$REPO_ROOT/scripts/ops/train.sh"
  --protocol configs/protocol_default.json
  --grid_preset support_e2_finalists_shortlist_v1
  --no_data_reduction
  --stat_tests
  --stat_mode "$STAT_MODE"
  --patience "$PATIENCE"
  --reduce_lr_patience "$REDUCE_LR_PATIENCE"
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running support finalists shortlist\n'
printf '  dataset_root=%s\n' "$DATASET_ROOT"
printf '  output_base=%s\n' "$OUTPUT_BASE"
printf '  grid_preset=%s\n' "support_e2_finalists_shortlist_v1"
printf '  stat_mode=%s\n' "$STAT_MODE"
printf '  patience=%s\n' "$PATIENCE"
printf '  reduce_lr_patience=%s\n' "$REDUCE_LR_PATIENCE"

exec "${CMD[@]}"
