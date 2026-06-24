#!/usr/bin/env bash
# One-shot: eval63 PRÓPRIO do E1-light (seed33), fechando o light-vs-light de 63
# regimes vs seed7 (30/63). Só inferência (reusa o modelo treinado), GPU livre.
# Aponta EXPLICITAMENTE p/ o braço light (exp_20260617_145453), não tail -1 (que
# pegou o heavy no E1 original). NÃO treina. Detached, ntfy.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
OUT=outputs/v3fc_e1light_eval63_20260624
MODEL_DIR=outputs/v3fc_e1gauss_20260617/exp_20260617_145453/train  # braço LIGHT seed33
TAG=GAUSS_e1_light
PRESET=v3_g6_aligned_s35c_gauss
TOPIC=projeto_vlc_ia
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

send "▶️ eval63 E1-light (seed33) — lançado" "fechando o light-vs-light de 63 regimes vs seed7 (30/63); só inferência, reusa o modelo treinado."

mkdir -p "$REPO/$OUT"
docker run -d --rm --name cvae_v3fc_e1light_eval63 \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=2 -e CVAE_BOOTSTRAP_PLOT_DEPS=0 \
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
      --output_base $OUT --protocol configs/protocol_v3fc_cross_evalall.json \
      --train_once_eval_all --reuse_model_run_dir \"$MODEL_DIR\" \
      --grid_preset $PRESET --grid_tag $TAG \
      --seed 33 --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
      --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_eval63.log 2>&1
    curl -fsS -H '🏁 eval63 E1-light pronto' -d 'mapa 63 do E1-light seed33 pronto; compare vs seed7 30/63 (reprodutibilidade no eval completo)' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
  "
send "▶️ eval63 E1-light — container no ar" "cvae_v3fc_e1light_eval63 iniciado (só inferência 63). Avisa quando terminar."
echo "lançado: cvae_v3fc_e1light_eval63 -> $REPO/$OUT"
