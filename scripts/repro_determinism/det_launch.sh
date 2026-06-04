#!/usr/bin/env bash
# Run ONE deterministic short training. Args: <out_subdir> <seed> <max_epochs>
set -euo pipefail
SUB="$1"; SEED="$2"; EP="$3"
docker run --rm \
  --name "cvae_rodrigo_det_${SUB}" \
  --runtime=nvidia \
  --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e CVAE_DETERMINISTIC=1 \
  -e REPRO_OUT_SUBDIR="$SUB" \
  -e REPRO_SEED="$SEED" \
  -e REPRO_MAX_EPOCHS="$EP" \
  -u "$(id -u):$(id -g)" \
  -v /home/eduardo/cVAe_2026:/workspace/2026/feat_seq_bigru_residual_cvae \
  -v /home/rodrigo/cvae_det/src:/workspace/2026/feat_seq_bigru_residual_cvae/src:ro \
  -v /home/rodrigo/cvae_det_out:/workspace/repro_outputs \
  -w /workspace/2026/feat_seq_bigru_residual_cvae \
  --entrypoint bash \
  vlc/tf25-gpu-ready:1 -lc "$(cat /home/rodrigo/cvae_det/det_inner.sh)"
