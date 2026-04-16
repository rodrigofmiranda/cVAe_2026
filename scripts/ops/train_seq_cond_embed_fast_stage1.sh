#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/ops/train_seq_cond_embed_fast_stage1.sh [extra protocol args...]

Purpose:
  Launch the first MDN-return quick screen on the active cond-embed fast path.

Preset:
  - seq_cond_embed_fast_stage1 (S35)

Defaults:
  - all_regimes_sel4curr.json (12 regimes)
  - 100k train / exp
  - 20k val / exp
  - quick stats with reduced sample caps
  - widened decoder clamp for the active cond-embed continuation

Environment overrides:
  DATASET_ROOT                Dataset root (default: repo-local Full Square dataset)
  RUN_STAMP                   Timestamp prefix for output naming (default: current time)
  OUTPUT_BASE                 Output base (default: outputs/<RUN_STAMP>_seq_cond_embed_fast_stage1_100k)
  STAT_MODE                   quick|full (default: quick)
  PATIENCE                    Early stopping patience (default: 80)
  REDUCE_LR_PATIENCE          LR plateau patience (default: 40)
  CVAE_DECODER_LOGVAR_CLAMP_LO  Lower decoder logvar clamp (default: -6.82)
  CVAE_DECODER_LOGVAR_CLAMP_HI  Upper decoder logvar clamp (default: 0.31)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

export DATASET_ROOT="${DATASET_ROOT:-${CVAE_DATASET_CONTAINER_ROOT:-$REPO_ROOT/data/dataset_fullsquare_organized}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs/${RUN_STAMP}_seq_cond_embed_fast_stage1_100k}"
STAT_MODE="${STAT_MODE:-quick}"
PATIENCE="${PATIENCE:-80}"
REDUCE_LR_PATIENCE="${REDUCE_LR_PATIENCE:-40}"
export CVAE_DECODER_LOGVAR_CLAMP_LO="${CVAE_DECODER_LOGVAR_CLAMP_LO:--6.82}"
export CVAE_DECODER_LOGVAR_CLAMP_HI="${CVAE_DECODER_LOGVAR_CLAMP_HI:-0.31}"

CMD=(
  "$REPO_ROOT/scripts/ops/train.sh"
  --protocol configs/all_regimes_sel4curr.json
  --grid_preset seq_cond_embed_fast_stage1
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

printf 'Running MDN return cond-embed fast stage 1\n'
printf '  dataset_root=%s\n' "$DATASET_ROOT"
printf '  output_base=%s\n' "$OUTPUT_BASE"
printf '  grid_preset=%s\n' "seq_cond_embed_fast_stage1"
printf '  stat_mode=%s\n' "$STAT_MODE"
printf '  patience=%s\n' "$PATIENCE"
printf '  reduce_lr_patience=%s\n' "$REDUCE_LR_PATIENCE"
printf '  decoder_logvar_clamp=[%s, %s]\n' "$CVAE_DECODER_LOGVAR_CLAMP_LO" "$CVAE_DECODER_LOGVAR_CLAMP_HI"

exec "${CMD[@]}"