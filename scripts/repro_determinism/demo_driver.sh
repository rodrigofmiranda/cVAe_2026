#!/usr/bin/env bash
# Determinism demonstration: 2 non-deterministic + 2 deterministic short runs
# (same seed 42), then compare per-epoch val_recon for bit-identity.
set -uo pipefail
EP=3
run_one() { # tag DET ENV OPDET SETSEED
  local tag=$1 det=$2 envf=$3 opdet=$4 setseed=$5
  echo "=== DEMO_RUN $tag (DET=$det ENV=$envf OPDET=$opdet SETSEED=$setseed) $(date -u +%T) ==="
  docker run --rm --name "cvae_rodrigo_demo_$tag" \
    --runtime=nvidia --security-opt apparmor=unconfined \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CVAE_DETERMINISTIC="$det" -e CVAE_DET_ENV="$envf" -e CVAE_DET_OPDET="$opdet" -e CVAE_DET_SETSEED="$setseed" \
    -e REPRO_OUT_SUBDIR="demo_$tag" -e REPRO_MAX_EPOCHS="$EP" \
    -u "$(id -u):$(id -g)" \
    -v /home/eduardo/cVAe_2026:/workspace/2026/feat_seq_bigru_residual_cvae \
    -v /home/rodrigo/cvae_det/src:/workspace/2026/feat_seq_bigru_residual_cvae/src:ro \
    -v /home/rodrigo/cvae_det_out:/workspace/repro_outputs \
    -w /workspace/2026/feat_seq_bigru_residual_cvae \
    --entrypoint bash vlc/tf25-gpu-ready:1 -lc "$(cat /home/rodrigo/cvae_det/demo_inner.sh)" \
    > "/home/rodrigo/cvae_det_out/demo_$tag.log" 2>&1
  echo "  $tag rc=$?"
}
rm -rf /home/rodrigo/cvae_det_out/demo_ndet_a /home/rodrigo/cvae_det_out/demo_ndet_b \
       /home/rodrigo/cvae_det_out/demo_det_a /home/rodrigo/cvae_det_out/demo_det_b
run_one ndet_a 0 1 0 0
run_one ndet_b 0 1 0 0
run_one det_a  1 1 0 0
run_one det_b  1 1 0 0
echo "=== COMPARISON ==="
python3 /home/rodrigo/cvae_det/compare.py
echo "DEMO_DONE"
