#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/ops/launch_seq_cond_embed_fast_stage1_gpu.sh [extra protocol args...]

Purpose:
  Launch the MDN return stage-1 quick screen in tmux + Docker without relying
  on repo-local Git LFS materialization.

Default dataset strategy:
  - mount a local, already-materialized Full Square dataset read-only into the container
  - avoid using the tracked dataset files inside this clone as runtime input

Environment overrides:
  HOST_DATASET_ROOT            Host path to a materialized dataset_fullsquare_organized
  CONTAINER_DATASET_ROOT       Container mount path (default: /workspace/shared_data/dataset_fullsquare_organized)
  RUN_STAMP                    Timestamp prefix for tmux/container/output naming
  SEED                         Training seed (default: 42)
  STAT_MODE                    quick|full (default: quick)
  PATIENCE                     Early stopping patience (default: 80)
  REDUCE_LR_PATIENCE           LR plateau patience (default: 40)
  CVAE_DECODER_LOGVAR_CLAMP_LO Lower decoder logvar clamp (default: -6.82)
  CVAE_DECODER_LOGVAR_CLAMP_HI Upper decoder logvar clamp (default: 0.31)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

detect_host_dataset_root() {
  local candidates=(
    "/home/rodrigo/cVAe_2026/data/dataset_fullsquare_organized"
    "/home/rodrigo/cVAe_2026_shape/data/dataset_fullsquare_organized"
    "/home/rodrigo/cVAe_2026_mdn_anchor/data/dataset_fullsquare_organized"
    "/home/rodrigo/cVAe_2026_mdn_explore/data/dataset_fullsquare_organized"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

HOST_DATASET_ROOT="${HOST_DATASET_ROOT:-$(detect_host_dataset_root || true)}"
CONTAINER_DATASET_ROOT="${CONTAINER_DATASET_ROOT:-/workspace/shared_data/dataset_fullsquare_organized}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
STAT_MODE="${STAT_MODE:-quick}"
PATIENCE="${PATIENCE:-80}"
REDUCE_LR_PATIENCE="${REDUCE_LR_PATIENCE:-40}"
CVAE_DECODER_LOGVAR_CLAMP_LO="${CVAE_DECODER_LOGVAR_CLAMP_LO:--6.82}"
CVAE_DECODER_LOGVAR_CLAMP_HI="${CVAE_DECODER_LOGVAR_CLAMP_HI:-0.31}"

if [[ -z "$HOST_DATASET_ROOT" ]]; then
  echo "Could not find a materialized Full Square dataset on the host. Set HOST_DATASET_ROOT explicitly." >&2
  exit 1
fi

SESSION="mdn_return_s35_${RUN_STAMP}"
CONTAINER="cvae_tf25_mdn_return_s35_${RUN_STAMP}"
OUTPUT_BASE="/workspace/2026/feat_seq_bigru_residual_cvae/outputs/${RUN_STAMP}_seq_cond_embed_fast_stage1_100k"
HOST_OUTPUT_DIR="$REPO_ROOT/outputs/${RUN_STAMP}_seq_cond_embed_fast_stage1_100k"
LOG_FILE="${OUTPUT_BASE}/logs/seed${SEED}.log"

mkdir -p "$HOST_OUTPUT_DIR/logs"

CVAE_TF25_TMUX_SESSION="$SESSION" \
CVAE_TF25_CONTAINER_NAME="$CONTAINER" \
CVAE_DATASET_HOST_ROOT="$HOST_DATASET_ROOT" \
CVAE_DATASET_CONTAINER_ROOT="$CONTAINER_DATASET_ROOT" \
scripts/ops/run_tf25_gpu.sh

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

EXTRA_CMD=""
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  EXTRA_CMD=" ${EXTRA_ARGS[*]}"
fi

tmux send-keys -t "$SESSION" \
  "cd /workspace/2026/feat_seq_bigru_residual_cvae && mkdir -p \"${OUTPUT_BASE}/logs\" && CVAE_DATASET_CONTAINER_ROOT=\"${CONTAINER_DATASET_ROOT}\" CVAE_DECODER_LOGVAR_CLAMP_LO=\"${CVAE_DECODER_LOGVAR_CLAMP_LO}\" CVAE_DECODER_LOGVAR_CLAMP_HI=\"${CVAE_DECODER_LOGVAR_CLAMP_HI}\" OUTPUT_BASE=\"${OUTPUT_BASE}\" STAT_MODE=\"${STAT_MODE}\" PATIENCE=\"${PATIENCE}\" REDUCE_LR_PATIENCE=\"${REDUCE_LR_PATIENCE}\" scripts/ops/train_seq_cond_embed_fast_stage1.sh --seed ${SEED}${EXTRA_CMD} | tee \"${LOG_FILE}\"" C-m

printf 'Launched MDN return cond-embed fast stage 1\n'
printf '  session=%s\n' "$SESSION"
printf '  container=%s\n' "$CONTAINER"
printf '  host_dataset_root=%s\n' "$HOST_DATASET_ROOT"
printf '  container_dataset_root=%s\n' "$CONTAINER_DATASET_ROOT"
printf '  output=%s\n' "$HOST_OUTPUT_DIR"