#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CVAE_TF25_CONTAINER_NAME:-cvae_tf25_gpu}"

if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' is not running." >&2
  echo "Start it first with: scripts/run_tf25_gpu.sh" >&2
  exit 1
fi

if docker exec -it "${CONTAINER_NAME}" bash; then
  exit 0
fi

STATUS=$?
if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
  echo "docker exec failed; retrying with sudo docker..." >&2
  exec sudo docker exec -it "${CONTAINER_NAME}" bash
fi

echo "docker exec failed for '${CONTAINER_NAME}'." >&2
echo "If the host reports an AppArmor permission error, try:" >&2
echo "  sudo docker exec -it ${CONTAINER_NAME} bash" >&2
exit "${STATUS}"
