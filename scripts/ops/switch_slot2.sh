#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_HOME="/home/$(id -un)"
WORKTREE_HOME="${CVAE_WORKTREE_HOME:-${DEFAULT_HOME}}"
WORKTREE_PREFIX="${CVAE_WORKTREE_PREFIX:-cVAe_2026}"

declare -A WORKTREE_PATHS=(
  [anchor]="${WORKTREE_HOME}/${WORKTREE_PREFIX}_mdn_anchor"
  [explore]="${WORKTREE_HOME}/${WORKTREE_PREFIX}_mdn_explore"
  [shape]="${WORKTREE_HOME}/${WORKTREE_PREFIX}_shape"
  [legacy2025]="${WORKTREE_HOME}/${WORKTREE_PREFIX}_legacy2025"
)

declare -A SESSION_NAMES=(
  [anchor]="anchor"
  [explore]="explore"
  [shape]="shape"
  [legacy2025]="legacy2025"
)

declare -A CONTAINER_NAMES=(
  [anchor]="cvae_anchor"
  [explore]="cvae_explore"
  [shape]="cvae_shape"
  [legacy2025]="cvae_legacy2025"
)

SLOT2_LINES=(explore shape legacy2025)

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/switch_slot2.sh status
  scripts/ops/switch_slot2.sh explore
  scripts/ops/switch_slot2.sh shape
  scripts/ops/switch_slot2.sh legacy2025

This script keeps `anchor` untouched and rotates only the second hot slot.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

stop_line() {
  local line="$1"
  local session_name="${SESSION_NAMES[$line]}"
  local container_name="${CONTAINER_NAMES[$line]}"

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    echo "stopped_tmux=${session_name}"
  fi

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    echo "stopped_container=${container_name}"
  fi
}

show_status() {
  local line path session_name container_name

  echo "worktree_home=${WORKTREE_HOME}"
  echo "worktree_prefix=${WORKTREE_PREFIX}"

  for line in anchor "${SLOT2_LINES[@]}"; do
    path="${WORKTREE_PATHS[$line]}"
    session_name="${SESSION_NAMES[$line]}"
    container_name="${CONTAINER_NAMES[$line]}"

    if [ -d "${path}" ]; then
      echo "worktree_${line}=${path}"
    else
      echo "worktree_${line}=missing:${path}"
    fi

    if tmux has-session -t "${session_name}" 2>/dev/null; then
      echo "tmux_${line}=up"
    else
      echo "tmux_${line}=down"
    fi

    if docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
      echo "container_${line}=up"
    else
      echo "container_${line}=down"
    fi
  done
}

start_line() {
  local line="$1"
  local path="${WORKTREE_PATHS[$line]}"
  local run_script="${path}/scripts/ops/run_tf25_gpu.sh"

  if [ ! -d "${path}" ]; then
    echo "Missing worktree for '${line}': ${path}" >&2
    exit 1
  fi

  if [ ! -x "${run_script}" ]; then
    echo "Missing run script for '${line}': ${run_script}" >&2
    exit 1
  fi

  CVAE_TF25_TMUX_SESSION="${SESSION_NAMES[$line]}" \
  CVAE_TF25_CONTAINER_NAME="${CONTAINER_NAMES[$line]}" \
  bash "${run_script}"
}

main() {
  local target="${1:-status}"
  local line

  require_cmd tmux
  require_cmd docker

  case "${target}" in
    status)
      show_status
      ;;
    explore|shape|legacy2025)
      for line in "${SLOT2_LINES[@]}"; do
        if [ "${line}" != "${target}" ]; then
          stop_line "${line}"
        fi
      done
      stop_line "${target}"
      start_line "${target}"
      echo "slot2_active=${target}"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
