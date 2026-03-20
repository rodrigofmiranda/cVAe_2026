#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${CVAE_TF25_TMUX_SESSION:-cvae_tf25_gpu}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required on the host for persistent container sessions." >&2
  exit 1
fi

if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' is not running." >&2
  echo "Start it first with: scripts/run_tf25_gpu.sh" >&2
  exit 1
fi

if [ -n "${TMUX:-}" ]; then
  exec tmux switch-client -t "${SESSION_NAME}"
fi

exec tmux attach -t "${SESSION_NAME}"
