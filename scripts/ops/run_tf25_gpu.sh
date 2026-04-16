#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_NAME="${CVAE_TF25_CONTAINER_NAME:-cvae_tf25_gpu}"
SESSION_NAME="${CVAE_TF25_TMUX_SESSION:-cvae_tf25_gpu}"
IMAGE_NAME="${CVAE_TF25_IMAGE:-vlc/tf25-gpu-ready:1}"
CONTAINER_WORKDIR="${CVAE_TF25_WORKDIR:-/workspace/2026/feat_seq_bigru_residual_cvae}"
HOST_DATASET_ROOT="${CVAE_DATASET_HOST_ROOT:-}"
CONTAINER_DATASET_ROOT="${CVAE_DATASET_CONTAINER_ROOT:-}"
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

EXTRA_DOCKER_ARGS=""
DATASET_MOUNT_INFO=""
if [[ -n "${HOST_DATASET_ROOT}" || -n "${CONTAINER_DATASET_ROOT}" ]]; then
  if [[ -z "${HOST_DATASET_ROOT}" || -z "${CONTAINER_DATASET_ROOT}" ]]; then
    echo "Both CVAE_DATASET_HOST_ROOT and CVAE_DATASET_CONTAINER_ROOT must be set together." >&2
    exit 1
  fi
  if [[ ! -d "${HOST_DATASET_ROOT}" ]]; then
    echo "Dataset mount source does not exist: ${HOST_DATASET_ROOT}" >&2
    exit 1
  fi
  EXTRA_DOCKER_ARGS+=" -e CVAE_DATASET_CONTAINER_ROOT=$(printf '%q' "${CONTAINER_DATASET_ROOT}")"
  EXTRA_DOCKER_ARGS+=" -v $(printf '%q' "${HOST_DATASET_ROOT}:${CONTAINER_DATASET_ROOT}:ro")"
  DATASET_MOUNT_INFO="${HOST_DATASET_ROOT}:${CONTAINER_DATASET_ROOT}:ro"
fi

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
  -e HOME=$(printf '%q' "${CONTAINER_WORKDIR}") \
  -u $(printf '%q' "$(id -u):$(id -g)") \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $(printf '%q' "${REPO_ROOT}:${CONTAINER_WORKDIR}") \
  ${EXTRA_DOCKER_ARGS} \
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
if [[ -n "${DATASET_MOUNT_INFO}" ]]; then
  echo "dataset_mount=${DATASET_MOUNT_INFO}"
fi
echo "bootstrap_plot_deps=${BOOTSTRAP_PLOT_DEPS}"
echo "enter: ${SCRIPT_DIR}/enter_tf25_gpu.sh"
echo "stop:  ${SCRIPT_DIR}/stop_tf25_gpu.sh"
