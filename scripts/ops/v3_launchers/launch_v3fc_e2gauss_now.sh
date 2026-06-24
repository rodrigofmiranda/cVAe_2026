#!/usr/bin/env bash
# One-shot: lança o E2 gauss FC (GPU livre).
# Treino com grid densificado (B1): 54 regimes.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
OUT=outputs/v3fc_e2gauss_20260620
TAG=GAUSS_e1   # Reusa a definição de grid de E1 (decoder gaussiano)
PRESET=v3_g6_aligned_s35c_gauss
TOPIC=projeto_vlc_ia
send() { curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

send "▶️ E2 gauss FC — lançado (B1 densificado)" "Iniciando treino cVAE Full Circle com grid densificado (54 regimes: 0.75, 0.9, 1.0, 1.16, 1.35, 1.5 m)."

mkdir -p "$REPO/$OUT"
docker run -d --rm --name cvae_v3fc_e2gauss \
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
      --output_base $OUT --protocol configs/protocol_v3fc_cross_train_e2.json \
      --train_once_eval_all --grid_preset $PRESET --grid_tag $TAG \
      --seed 33 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_train.log 2>&1
    curl -fsS -H '✅ E2 gauss FC — treino ok' -d 'treino 54 ok; iniciando inferência 63' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
    unset CVAE_DETERMINISTIC CVAE_DET_OPDET CVAE_DET_SETSEED
    MODEL_DIR=\$(ls -d $OUT/exp_*/train 2>/dev/null | tail -1)
    python -u -m src.protocol.run \
      --dataset_root /data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED \
      --output_base $OUT --protocol configs/protocol_v3fc_cross_evalall.json \
      --train_once_eval_all --reuse_model_run_dir \"\$MODEL_DIR\" \
      --grid_preset $PRESET --grid_tag $TAG \
      --seed 33 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_eval63.log 2>&1
    curl -fsS -H '🏁 E2 gauss FC — TUDO pronto' -d 'E2 finalizado (treino 54 + eval63)' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
  "
send "▶️ E2 gauss FC — container no ar" "cvae_v3fc_e2gauss iniciado (treino em 54 regimes). Acompanhe em run_train.log."
echo "lançado: cvae_v3fc_e2gauss -> $REPO/$OUT"
