#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CVAE_TF25_CONTAINER_NAME:-cvae_tf25_gpu}"
SESSION_NAME="${CVAE_TF25_TMUX_SESSION:-cvae_tf25_gpu}"

FOUND=0

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}" >/dev/null 2>&1 || true
  echo "Stopped tmux session '${SESSION_NAME}'."
  FOUND=1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Stopped and removed container '${CONTAINER_NAME}'."
  FOUND=1
fi

if [ "${FOUND}" -eq 0 ]; then
  echo "No tmux session or container found for '${SESSION_NAME}'/'${CONTAINER_NAME}'."
fi
