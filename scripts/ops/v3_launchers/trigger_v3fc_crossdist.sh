#!/usr/bin/env bash
# Espera os 2 seeds liberarem a GPU, então lança o full-circle cross-distance
# (mesmo desenho do full-square). Detached, sobrevive a cortes do harness.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
OUT=outputs/v3fc_crossdist_20260613
TOPIC=projeto_vlc_ia
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

# 1) esperar os 2 seeds sumirem
while docker ps --format '{{.Names}}' 2>/dev/null | grep -qE 'cvae_v3_s35cg6a_s3[56]'; do sleep 120; done
send "▶️ full-circle cross-dist — slot liberou, lançando" "seeds 35/36 terminaram; subindo full-circle base."

mkdir -p "$REPO/$OUT"
docker run -d --rm --name cvae_v3fc_crossdist \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=2 -e CVAE_BOOTSTRAP_PLOT_DEPS=0 \
  -e CVAE_DETERMINISTIC=1 -e CVAE_DET_OPDET=0 -e CVAE_DET_SETSEED=0 \
  -u "$(id -u):$(id -g)" -e HOME="$WORKDIR" \
  -e CVAE_DECODER_LOGVAR_CLAMP_LO=-6.61 -e CVAE_DECODER_LOGVAR_CLAMP_HI=-0.54 \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$REPO":"$WORKDIR" \
  -v /home/rodrigo/cVAe_2026_full_square/.pydeps:"$WORKDIR"/.pydeps \
  -v /home/rodrigo/cVAe_2026_full_square/.git:/home/rodrigo/cVAe_2026_full_square/.git:ro \
  -v /home/rodrigo/1-Data:/data:ro \
  -w "$WORKDIR" \
  --entrypoint bash vlc/tf25-gpu-ready:1 -lc "
    source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
    export PYTHONPATH=\$PWD
    python -u -m src.protocol.run \
      --dataset_root /data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED \
      --output_base $OUT --protocol configs/protocol_v3fc_cross_train.json \
      --train_once_eval_all --grid_preset v3_g6_aligned_s35c --grid_tag S35CG6A_multibw_energy_lmmd05_le05 \
      --seed 33 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_train.log 2>&1
    curl -fsS -H 'Title: ✅ full-circle cross-dist — treino ok' -d 'treino 36 ok; iniciando inferência 63' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
    unset CVAE_DETERMINISTIC CVAE_DET_OPDET CVAE_DET_SETSEED
    MODEL_DIR=\$(ls -d $OUT/exp_*/train 2>/dev/null | tail -1)
    python -u -m src.protocol.run \
      --dataset_root /data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED \
      --output_base $OUT --protocol configs/protocol_v3fc_cross_evalall.json \
      --train_once_eval_all --reuse_model_run_dir \"\$MODEL_DIR\" \
      --grid_preset v3_g6_aligned_s35c --grid_tag S35CG6A_multibw_energy_lmmd05_le05 \
      --seed 33 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_eval63.log 2>&1
    curl -fsS -H '🏁 full-circle cross-dist — inferência 63 ok' -d 'mapa cross-distance full-circle pronto' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
  "
send "▶️ full-circle cross-dist — container no ar" "cvae_v3fc_crossdist iniciado (clamp herdado do full-square: 1o suspeito se basin ruim)."
