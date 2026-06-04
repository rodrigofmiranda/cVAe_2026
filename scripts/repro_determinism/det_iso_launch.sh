#!/usr/bin/env bash
# Isolation test. Args: <tag> <DET> <ENV> <OPDET> <SETSEED>
set -euo pipefail
TAG="$1"; DET="$2"; ENVF="$3"; OPDET="$4"; SETSEED="$5"
docker run --rm \
  --name "cvae_rodrigo_iso_${TAG}" \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e CVAE_DETERMINISTIC="$DET" -e CVAE_DET_ENV="$ENVF" -e CVAE_DET_OPDET="$OPDET" -e CVAE_DET_SETSEED="$SETSEED" \
  -e REPRO_OUT_SUBDIR="iso_${TAG}" \
  -u "$(id -u):$(id -g)" \
  -v /home/eduardo/cVAe_2026:/workspace/2026/feat_seq_bigru_residual_cvae \
  -v /home/rodrigo/cvae_det/src:/workspace/2026/feat_seq_bigru_residual_cvae/src:ro \
  -v /home/rodrigo/cvae_det_out:/workspace/repro_outputs \
  -w /workspace/2026/feat_seq_bigru_residual_cvae \
  --entrypoint bash \
  vlc/tf25-gpu-ready:1 -lc "$(cat /home/rodrigo/cvae_det/det_iso.sh)"
