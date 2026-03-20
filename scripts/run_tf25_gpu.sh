#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_NAME="${CVAE_TF25_CONTAINER_NAME:-cvae_tf25_gpu}"
IMAGE_NAME="${CVAE_TF25_IMAGE:-vlc/tf25-gpu-ready:1}"
CONTAINER_WORKDIR="${CVAE_TF25_WORKDIR:-/workspace/cVAe_2026}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

CONTAINER_ID="$(
  docker run -d \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia \
    --security-opt apparmor=unconfined \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -u "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${CONTAINER_WORKDIR}" \
    -w "${CONTAINER_WORKDIR}" \
    --entrypoint bash \
    "${IMAGE_NAME}" \
    -lc "sleep infinity"
)"

echo "${CONTAINER_ID}"
echo "container_name=${CONTAINER_NAME}"
echo "image=${IMAGE_NAME}"
echo "mount=${REPO_ROOT}:${CONTAINER_WORKDIR}"
echo "enter: ${SCRIPT_DIR}/enter_tf25_gpu.sh"
echo "stop:  ${SCRIPT_DIR}/stop_tf25_gpu.sh"
