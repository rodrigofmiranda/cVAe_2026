#!/usr/bin/env bash
# E1 (reprojeto V3): retreina o FULL-CIRCLE com decoder GAUSSIANO (controle de
# variável única vs o campeão MDN s35c). Espera o slot da GPU liberar (NÃO toca
# no run do eduardo nem em outros cvae_v3), então lança detached. Sobrevive a
# cortes do harness (rode com setsid nohup).
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
OUT=outputs/v3fc_e1gauss_20260617
TAG=GAUSS_e1   # regex: casa os dois braços (light + heavy)
PRESET=v3_g6_aligned_s35c_gauss
TOPIC=projeto_vlc_ia
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

# 1) esperar a GPU liberar: enquanto qualquer treino cvae (eduardo ou v3) estiver no ar
ED_WAS_UP=0
docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^cvae_eduardo$' && ED_WAS_UP=1
while docker ps --format '{{.Names}}' 2>/dev/null \
        | grep -v '^cvae_v3fc_e1gauss$' \
        | grep -qiE 'cvae_eduardo|cvae_v3(fc|fs|_s|_)'; do
  # avisa assim que o run do eduardo sair (mesmo que outro ainda segure o slot)
  if [ "$ED_WAS_UP" = "1" ] && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^cvae_eduardo$'; then
    send "🏁 Run do eduardo terminou" "cvae_eduardo saiu da GPU."
    ED_WAS_UP=0
  fi
  sleep 120
done
[ "$ED_WAS_UP" = "1" ] && send "🏁 Run do eduardo terminou" "cvae_eduardo saiu da GPU."
send "▶️ E1 gauss FC — slot liberou, lançando 2 braços" "GPU livre; subindo full-circle Gaussiano: light (s35c) + heavy (W9/h128/2L)."

mkdir -p "$REPO/$OUT"
docker run -d --rm --name cvae_v3fc_e1gauss \
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
      --seed 33 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_train.log 2>&1
    curl -fsS -H '✅ E1 gauss FC — treino ok' -d 'treino 36 ok; iniciando inferência 63' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
    unset CVAE_DETERMINISTIC CVAE_DET_OPDET CVAE_DET_SETSEED
    MODEL_DIR=\$(ls -d $OUT/exp_*/train 2>/dev/null | tail -1)
    python -u -m src.protocol.run \
      --dataset_root /data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED \
      --output_base $OUT --protocol configs/protocol_v3fc_cross_evalall.json \
      --train_once_eval_all --reuse_model_run_dir \"\$MODEL_DIR\" \
      --grid_preset $PRESET --grid_tag $TAG \
      --seed 33 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_eval63.log 2>&1
    curl -fsS -H '🏁 E1 gauss FC — TUDO pronto (light+heavy)' -d 'mapa cross-distance 63 dos 2 braços Gaussianos pronto; compare gates vs campeao MDN s35c' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
  "
send "▶️ E1 gauss FC — container no ar" "cvae_v3fc_e1gauss iniciado (2 braços: light s35c + heavy W9/h128/2L). Avisa quando terminar."
