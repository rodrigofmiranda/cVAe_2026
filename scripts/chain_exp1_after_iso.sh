#!/usr/bin/env bash
# Chain launcher: wait for the isolation run (cvae_v3_iso075) to free the GPU,
# then launch Exp 1 (G6-aligned loss, preset v3_g6_aligned) detached, with its
# own ntfy watcher. Run this script itself detached (setsid nohup ... & disown).
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
OUT_REL=outputs/v3g6_aligned_det_20260610
OUT="$REPO/$OUT_REL"
TOPIC="projeto_vlc_ia"

# 1) wait for the isolation container to exit (GPU free)
while docker ps --format '{{.Names}}' 2>/dev/null | grep -qx cvae_v3_iso075; do
  sleep 120
done

mkdir -p "$OUT"
docker rm -f cvae_v3_g6a >/dev/null 2>&1 || true

# 2) launch Exp 1 detached (deterministic, V3 clamp, stat tests included)
docker run -d --rm --name cvae_v3_g6a \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=2 -e CVAE_BOOTSTRAP_PLOT_DEPS=0 \
  -e CVAE_DETERMINISTIC=1 -e CVAE_DET_OPDET=0 -e CVAE_DET_SETSEED=0 \
  -u "$(id -u):$(id -g)" \
  -e HOME=/workspace/2026/feat_seq_bigru_residual_cvae \
  -e CVAE_DECODER_LOGVAR_CLAMP_LO=-6.61 -e CVAE_DECODER_LOGVAR_CLAMP_HI=-0.54 \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$REPO":/workspace/2026/feat_seq_bigru_residual_cvae \
  -v /home/rodrigo/cVAe_2026_full_square/.pydeps:/workspace/2026/feat_seq_bigru_residual_cvae/.pydeps \
  -v /home/rodrigo/cVAe_2026_full_square/.git:/home/rodrigo/cVAe_2026_full_square/.git:ro \
  -v /home/rodrigo/1-Data:/data:ro \
  -w /workspace/2026/feat_seq_bigru_residual_cvae \
  --entrypoint bash vlc/tf25-gpu-ready:1 -lc "
    source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
    export PYTHONPATH=\$PWD
    python -u -m src.protocol.run \
      --dataset_root /data/Dataset/V3/FULLSQUARE_2026_V3_ORGANIZED \
      --output_base $OUT_REL \
      --protocol configs/protocol_v3_fullsquare.json \
      --train_once_eval_all \
      --grid_preset v3_g6_aligned \
      --grid_tag V3G6A_s39b_multibw_energy_lmmd05_le05 \
      --seed 33 --no_data_reduction \
      --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 \
      > $OUT_REL/run.log 2>&1
  "

curl -fsS -H "Title: 🚀 Exp1 G6-aligned — iniciado" \
  -d "GPU liberou; Exp1 (multibw MMD + energy, lmmd=0.5, le=0.5, seed 33) lançado. Log: $OUT/run.log" \
  "https://ntfy.sh/$TOPIC" >/dev/null 2>&1

# 3) watcher for Exp 1 completion
exec bash /home/rodrigo/.claude/skills/vlc-cvae-research/scripts/ntfy_run_watch.sh \
  cvae_v3_g6a "$OUT" "Exp1 G6-aligned (multibw+energy)"
