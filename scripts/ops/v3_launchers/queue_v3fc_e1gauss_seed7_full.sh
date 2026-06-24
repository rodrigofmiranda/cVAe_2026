#!/usr/bin/env bash
# Enfileira o seed7 COMPLETO (light: train + eval63) p/ rodar SOZINHO depois do E1.
# Motivo: a co-run concorrente morreu por OOM de RAM do HOST (swap 5.5/8GB; 2
# processos TF + datasets não cabem nos 64GB) — o gargalo foi RAM, não GPU. Solo
# tem toda a RAM/GPU → teste de reprodutibilidade limpo. NÃO toca no eduardo
# (shell ocioso, 0 recurso). Detached (setsid nohup) p/ sobreviver a cortes.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
OUT=outputs/v3fc_e1gauss_seed7_full_20260617
TAG=GAUSS_e1_light   # só o braço light (controle de reprodutibilidade da bacia)
PRESET=v3_g6_aligned_s35c_gauss
TOPIC=projeto_vlc_ia
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

# 1) esperar o E1 terminar: o container roda train(2 braços)+eval63 e sai via --rm
while docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^cvae_v3fc_e1gauss$'; do sleep 120; done
send "🏁 E1 terminou — preparando seed7 solo" "container cvae_v3fc_e1gauss saiu; verificando GPU livre antes de lançar."

# 2) proxy CERTO (lição do bug anterior): confirmar GPU REALMENTE livre, não só ausência de container
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); MEM=${MEM:-99999}
while [ "${MEM:-99999}" -gt 4000 ]; do sleep 120; MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); MEM=${MEM:-99999}; done
send "▶️ seed7 COMPLETO — GPU livre, lançando solo" "reprodutibilidade definitiva da bacia (light seed7, train + eval63)."

mkdir -p "$REPO/$OUT"
docker run -d --rm --name cvae_v3fc_e1gauss_s7full \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=2 -e CVAE_BOOTSTRAP_PLOT_DEPS=0 \
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
    curl -fsS -H '✅ seed7 — treino ok' -d 'light seed7 treinado; iniciando eval 63' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
    unset CVAE_DETERMINISTIC CVAE_DET_OPDET CVAE_DET_SETSEED
    MODEL_DIR=\$(ls -d $OUT/exp_*/train 2>/dev/null | tail -1)
    python -u -m src.protocol.run \
      --dataset_root /data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED \
      --output_base $OUT --protocol configs/protocol_v3fc_cross_evalall.json \
      --train_once_eval_all --reuse_model_run_dir \"\$MODEL_DIR\" \
      --grid_preset $PRESET --grid_tag $TAG \
      --seed 7 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_eval63.log 2>&1
    curl -fsS -H '🏁 seed7 COMPLETO — pronto' -d 'mapa 63 do seed7 pronto; compare gates/val_recon vs E1 seed33 (reprodutibilidade da bacia)' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
  "
send "▶️ seed7 COMPLETO — container no ar" "cvae_v3fc_e1gauss_s7full iniciado (solo, light seed7, train + eval63)."
echo "enfileirado: cvae_v3fc_e1gauss_s7full -> $REPO/$OUT (espera o E1 sair)"
