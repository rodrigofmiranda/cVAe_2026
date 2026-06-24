#!/usr/bin/env bash
# Réplica seed do E1 light (controle de reprodutibilidade da bacia): MESMO config
# FC light-gauss do E1, mudando SÓ a seed (33 -> 7). Roda em paralelo, na folga da
# GPU, com memory-growth pra não estourar. SÓ o braço light, SÓ treino (sem eval63,
# pra ser leve). Não toca no E1 nem no eduardo. É a run DESCARTÁVEL: se o braço
# heavy do E1 precisar da GPU, esta aqui é a que para.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
OUT=outputs/v3fc_e1gauss_seed7_20260617
TAG=GAUSS_e1_light   # casa SÓ o candidato light
PRESET=v3_g6_aligned_s35c_gauss
TOPIC=projeto_vlc_ia
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

send "▶️ Réplica seed7 do E1 light — lançando em paralelo" "controle de reprodutibilidade da bacia (mesmo config FC light-gauss, seed 33->7); memory-growth; só treino."

mkdir -p "$REPO/$OUT"
docker run -d --rm --name cvae_v3fc_e1gauss_s7 \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=2 -e CVAE_BOOTSTRAP_PLOT_DEPS=0 \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -e CVAE_DETERMINISTIC=1 -e CVAE_DET_OPDET=0 -e CVAE_DET_SETSEED=0 \
  -u "$(id -u):$(id -g)" -e HOME="$WORKDIR" \
  -e CVAE_DECODER_LOGVAR_CLAMP_LO=-6.61 -e CVAE_DECODER_LOGVAR_CLAMP_HI=-0.54 \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$REPO":"$WORKDIR" \
  -v /home/rodrigo/cVAe_2026_full_square/.pydeps:"$WORKDIR"/.pydeps \
  -v /home/rodrigo/1-Data:/data:ro \
  -w "$WORKDIR" \
  --entrypoint bash vlc/tf25-gpu-ready:1 -lc "
    source scripts/ops/container_bootstrap_python.sh 2>/dev/null || true
    export PYTHONPATH=\$PWD
    python -u -m src.protocol.run \
      --dataset_root /data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED \
      --output_base $OUT --protocol configs/protocol_v3fc_cross_train.json \
      --train_once_eval_all --grid_preset $PRESET --grid_tag $TAG \
      --seed 7 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_train.log 2>&1
    curl -fsS -H '🔁 Réplica seed7 — treino ok' -d 'braço light seed7 treinado; compare val_recon/gates vs E1 light seed33' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
  "
send "▶️ Réplica seed7 — container no ar" "cvae_v3fc_e1gauss_s7 iniciado (light seed7, só treino). Avisa quando terminar."
echo "lançado: cvae_v3fc_e1gauss_s7 -> $REPO/$OUT"
