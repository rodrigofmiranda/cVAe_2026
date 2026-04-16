#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/ops/train_full_circle_disk_bs8192_lat10.sh [extra protocol args...]

Purpose:
  Run the incremental Full Circle follow-up around the current disk_geom3 anchor.

Candidates:
  - disk_geom3 anchor
  - disk_geom3_bs8192
  - disk_geom3_lat10

Defaults:
  - protocol_full_circle_sel4curr.json (12 regimes)
  - 100k train / exp
  - 20k val / exp
  - quick stats with reduced sample caps
  - preset: support_full_circle_disk_bs8192_lat10_v1

Environment overrides:
  DATASET_ROOT         Dataset root (default: repo-local FULL_CIRCLE dataset)
  RUN_STAMP            Timestamp prefix for output naming (default: current time)
  OUTPUT_BASE          Output base (default: outputs/full_circle/<RUN_STAMP>_disk_bs8192_lat10_100k)
  STAT_MODE            quick|full (default: quick)
  PATIENCE             Early stopping patience (default: 50)
  REDUCE_LR_PATIENCE   LR plateau patience (default: 25)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/FULL_CIRCLE_2026}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs/full_circle/${RUN_STAMP}_disk_bs8192_lat10_100k}"
STAT_MODE="${STAT_MODE:-quick}"
PATIENCE="${PATIENCE:-50}"
REDUCE_LR_PATIENCE="${REDUCE_LR_PATIENCE:-25}"

CMD=(
  "$REPO_ROOT/scripts/ops/train.sh"
  --protocol configs/protocol_full_circle_sel4curr.json
  --grid_preset support_full_circle_disk_bs8192_lat10_v1
  --no_data_reduction
  --max_samples_per_exp 100000
  --max_val_samples_per_exp 20000
  --max_dist_samples 20000
  --stat_tests
  --stat_mode "$STAT_MODE"
  --stat_max_n 2000
  --patience "$PATIENCE"
  --reduce_lr_patience "$REDUCE_LR_PATIENCE"
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running Full Circle disk bs8192/lat10 follow-up\n'
printf '  dataset_root=%s\n' "$DATASET_ROOT"
printf '  output_base=%s\n' "$OUTPUT_BASE"
printf '  grid_preset=%s\n' "support_full_circle_disk_bs8192_lat10_v1"
printf '  stat_mode=%s\n' "$STAT_MODE"
printf '  patience=%s\n' "$PATIENCE"
printf '  reduce_lr_patience=%s\n' "$REDUCE_LR_PATIENCE"

exec "${CMD[@]}"