#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_NAME="${CVAE_TF25_CONTAINER_NAME:-cvae_tf25_gpu}"
SESSION_NAME="${CVAE_TF25_TMUX_SESSION:-cvae_tf25_gpu}"
IMAGE_NAME="${CVAE_TF25_IMAGE:-vlc/tf25-gpu-ready:1}"
DEFAULT_CONTAINER_WORKDIR="${REPO_ROOT}"
CONTAINER_WORKDIR="${CVAE_TF25_WORKDIR:-${DEFAULT_CONTAINER_WORKDIR}}"
TF_CPP_MIN_LOG_LEVEL_VALUE="${CVAE_TF_CPP_MIN_LOG_LEVEL:-2}"
TF_ENABLE_ONEDNN_OPTS_VALUE="${CVAE_TF_ENABLE_ONEDNN_OPTS:-0}"
BOOTSTRAP_PLOT_DEPS="${CVAE_BOOTSTRAP_PLOT_DEPS:-1}"
CONTAINER_BOOTSTRAP_SCRIPT="${CVAE_CONTAINER_BOOTSTRAP_SCRIPT:-scripts/ops/container_bootstrap_python.sh}"

CONTAINER_BOOTSTRAP_ABS="${CONTAINER_WORKDIR}/${CONTAINER_BOOTSTRAP_SCRIPT}"
CONTAINER_ENTRY_CMD="
if [ -f $(printf '%q' "${CONTAINER_BOOTSTRAP_ABS}") ]; then
  source $(printf '%q' "${CONTAINER_BOOTSTRAP_ABS}")
else
  echo \"[bootstrap][warn] Missing bootstrap script: ${CONTAINER_BOOTSTRAP_ABS}\"
fi
exec bash -l
"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required on the host for persistent container sessions." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' is already running."
  echo "enter: ${SCRIPT_DIR}/enter_tf25_gpu.sh"
  echo "stop:  ${SCRIPT_DIR}/stop_tf25_gpu.sh"
  exit 0
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

RUN_CMD="
cd $(printf '%q' "${REPO_ROOT}") && exec docker run --rm -it \
  --name $(printf '%q' "${CONTAINER_NAME}") \
  --runtime=nvidia \
  --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=$(printf '%q' "${TF_CPP_MIN_LOG_LEVEL_VALUE}") \
  -e TF_ENABLE_ONEDNN_OPTS=$(printf '%q' "${TF_ENABLE_ONEDNN_OPTS_VALUE}") \
  -e CVAE_BOOTSTRAP_PLOT_DEPS=$(printf '%q' "${BOOTSTRAP_PLOT_DEPS}") \
  -e CVAE_TF25_WORKDIR=$(printf '%q' "${CONTAINER_WORKDIR}") \
  -e CVAE_HOST_REPO_ROOT=$(printf '%q' "${REPO_ROOT}") \
  -e HOME=$(printf '%q' "${CONTAINER_WORKDIR}") \
  -u $(printf '%q' "$(id -u):$(id -g)") \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $(printf '%q' "${REPO_ROOT}:${CONTAINER_WORKDIR}") \
  -w $(printf '%q' "${CONTAINER_WORKDIR}") \
  --entrypoint bash \
  $(printf '%q' "${IMAGE_NAME}") \
  -lc $(printf '%q' "${CONTAINER_ENTRY_CMD}")
"

tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${RUN_CMD}")"

echo "tmux_session=${SESSION_NAME}"
echo "container_name=${CONTAINER_NAME}"
echo "image=${IMAGE_NAME}"
echo "mount=${REPO_ROOT}:${CONTAINER_WORKDIR}"
echo "bootstrap_plot_deps=${BOOTSTRAP_PLOT_DEPS}"
echo "enter: ${SCRIPT_DIR}/enter_tf25_gpu.sh"
echo "stop:  ${SCRIPT_DIR}/stop_tf25_gpu.sh"
