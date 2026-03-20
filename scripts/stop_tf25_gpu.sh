#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CVAE_TF25_CONTAINER_NAME:-cvae_tf25_gpu}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Stopped and removed '${CONTAINER_NAME}'."
  exit 0
fi

if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
  if sudo docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    sudo docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    sudo docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "Stopped and removed '${CONTAINER_NAME}' with sudo docker."
    exit 0
  fi
fi

echo "Container '${CONTAINER_NAME}' was not found."
