#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/ops/train_full_circle_clean_bs8192_lat10_split.sh [extra protocol args...]

Purpose:
  Launch the clean Full Circle restart split across two tmux + Docker GPU stacks.

Split plan:
  - split_a: clean baseline + clean_bs8192
  - split_b: clean_lat10

Defaults:
  - common RUN_STAMP for both splits
  - protocol_full_circle_sel4curr.json (12 regimes)
  - 100k train / exp
  - 20k val / exp
  - quick stats with reduced sample caps

Environment overrides:
  DATASET_ROOT         Dataset root (default: repo-local FULL_CIRCLE dataset)
  RUN_STAMP            Timestamp prefix shared by both splits (default: current time)
  SEED                 Seed passed to both launches (default: 42)
  STAT_MODE            quick|full (default: quick)
  PATIENCE             Early stopping patience (default: 50)
  REDUCE_LR_PATIENCE   LR plateau patience (default: 25)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/FULL_CIRCLE_2026}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
STAT_MODE="${STAT_MODE:-quick}"
PATIENCE="${PATIENCE:-50}"
REDUCE_LR_PATIENCE="${REDUCE_LR_PATIENCE:-25}"

SESSION_A="fc_clean_a_${RUN_STAMP}"
SESSION_B="fc_clean_b_${RUN_STAMP}"
CONTAINER_A="cvae_tf25_fc_clean_a_${RUN_STAMP}"
CONTAINER_B="cvae_tf25_fc_clean_b_${RUN_STAMP}"

OUTPUT_BASE_A="/workspace/2026/feat_seq_bigru_residual_cvae/outputs/full_circle/${RUN_STAMP}_clean_bs8192_lat10_100k_split_a"
OUTPUT_BASE_B="/workspace/2026/feat_seq_bigru_residual_cvae/outputs/full_circle/${RUN_STAMP}_clean_bs8192_lat10_100k_split_b"
LOG_A="${OUTPUT_BASE_A}/logs/seed${SEED}.log"
LOG_B="${OUTPUT_BASE_B}/logs/seed${SEED}.log"

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

EXTRA_CMD=""
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  EXTRA_CMD=" ${EXTRA_ARGS[*]}"
fi

mkdir -p "$REPO_ROOT/outputs/full_circle/${RUN_STAMP}_clean_bs8192_lat10_100k_split_a/logs"
mkdir -p "$REPO_ROOT/outputs/full_circle/${RUN_STAMP}_clean_bs8192_lat10_100k_split_b/logs"

CVAE_TF25_TMUX_SESSION="$SESSION_A" \
CVAE_TF25_CONTAINER_NAME="$CONTAINER_A" \
scripts/ops/run_tf25_gpu.sh

CVAE_TF25_TMUX_SESSION="$SESSION_B" \
CVAE_TF25_CONTAINER_NAME="$CONTAINER_B" \
scripts/ops/run_tf25_gpu.sh

tmux send-keys -t "$SESSION_A" \
  "cd /workspace/2026/feat_seq_bigru_residual_cvae && mkdir -p \"${OUTPUT_BASE_A}/logs\" && DATASET_ROOT=\"/workspace/2026/feat_seq_bigru_residual_cvae/data/FULL_CIRCLE_2026\" OUTPUT_BASE=\"${OUTPUT_BASE_A}\" STAT_MODE=\"${STAT_MODE}\" PATIENCE=\"${PATIENCE}\" REDUCE_LR_PATIENCE=\"${REDUCE_LR_PATIENCE}\" scripts/ops/train_full_circle_clean_bs8192_lat10.sh --grid_tag \"^S27cov_fc_clean_lc0p25_t0p03$|^S27cov_fc_clean_lc0p25_t0p03_bs8192$\" --seed ${SEED}${EXTRA_CMD} | tee \"${LOG_A}\"" C-m

tmux send-keys -t "$SESSION_B" \
  "cd /workspace/2026/feat_seq_bigru_residual_cvae && mkdir -p \"${OUTPUT_BASE_B}/logs\" && DATASET_ROOT=\"/workspace/2026/feat_seq_bigru_residual_cvae/data/FULL_CIRCLE_2026\" OUTPUT_BASE=\"${OUTPUT_BASE_B}\" STAT_MODE=\"${STAT_MODE}\" PATIENCE=\"${PATIENCE}\" REDUCE_LR_PATIENCE=\"${REDUCE_LR_PATIENCE}\" scripts/ops/train_full_circle_clean_bs8192_lat10.sh --grid_tag \"^S27cov_fc_clean_lc0p25_t0p03_lat10$\" --seed ${SEED}${EXTRA_CMD} | tee \"${LOG_B}\"" C-m

printf 'Launched clean Full Circle split run\n'
printf '  run_stamp=%s\n' "$RUN_STAMP"
printf '  split_a_session=%s\n' "$SESSION_A"
printf '  split_a_container=%s\n' "$CONTAINER_A"
printf '  split_a_output=%s\n' "$REPO_ROOT/outputs/full_circle/${RUN_STAMP}_clean_bs8192_lat10_100k_split_a"
printf '  split_b_session=%s\n' "$SESSION_B"
printf '  split_b_container=%s\n' "$CONTAINER_B"
printf '  split_b_output=%s\n' "$REPO_ROOT/outputs/full_circle/${RUN_STAMP}_clean_bs8192_lat10_100k_split_b"