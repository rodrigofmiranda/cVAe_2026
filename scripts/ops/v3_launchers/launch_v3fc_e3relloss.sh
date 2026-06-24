#!/usr/bin/env bash
# E3 / Fix A1: FC Gaussiano + loss de erro relativo (lambda_rel=10), 2 SEEDS
# (33 e 7) p/ controlar o não-determinismo da bacia. SEQUENCIAL: cada seed roda
# em FOREGROUND (docker run --rm sem -d) dentro deste script detached, então o
# host NUNCA tem 2 treinos juntos (teto de RAM). Cada seed: train (light) + eval63.
# Rode detached: setsid nohup bash launch_v3fc_e3relloss.sh ...
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
PRESET=v3_g6_aligned_s35c_gauss_relloss
TAG=relA1
TOPIC=projeto_vlc_ia
send(){ curl -fsS -H "Title: $1" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; }

# proxy CERTO: GPU realmente livre (lição do trigger quebrado)
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); MEM=${MEM:-99999}
while [ "${MEM:-99999}" -gt 4000 ]; do sleep 120; MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); MEM=${MEM:-99999}; done

send "▶️ E3 relloss (A1) — iniciando 2 seeds" "FC Gaussiano + loss relativo (lambda_rel=10); seeds 33 e 7 sequenciais; alvo: 0.75m (problema A)."

for SEED in 33 7; do
  OUT=outputs/v3fc_e3relloss_recal_s${SEED}_20260624
  NAME=cvae_v3fc_e3relloss_s${SEED}
  mkdir -p "$REPO/$OUT"
  send "▶️ E3 relloss seed${SEED} — treino" "lançando seed ${SEED} (train light + eval63)."
  # FOREGROUND (sem -d): bloqueia até terminar → seed seguinte só começa depois
  docker run --rm --name "$NAME" \
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
        --seed $SEED --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
        --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_train.log 2>&1
      curl -fsS -H '✅ E3 relloss seed${SEED} — treino ok' -d 'iniciando eval63' https://ntfy.sh/$TOPIC >/dev/null 2>&1 || true
      unset CVAE_DETERMINISTIC CVAE_DET_OPDET CVAE_DET_SETSEED
      MODEL_DIR=\$(ls -d $OUT/exp_*/train 2>/dev/null | tail -1)
      python -u -m src.protocol.run \
        --dataset_root /data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED \
        --output_base $OUT --protocol configs/protocol_v3fc_cross_evalall.json \
        --train_once_eval_all --reuse_model_run_dir \"\$MODEL_DIR\" \
        --grid_preset $PRESET --grid_tag $TAG \
        --seed $SEED --no_data_reduction --stat_tests --stat_mode quick --stat_seed 42 \
        --train_regime_diagnostics_focus_only_0p8m 0 > $OUT/run_eval63.log 2>&1
    "
  send "🏁 E3 relloss seed${SEED} — pronto" "seed ${SEED} train+eval63 ok. Compare 0.75m e n_pass vs E1-light (27/36, 30/63)."
done

send "🏁🏁 E3 relloss (A1) — AMBAS as seeds prontas" "seeds 33 e 7 concluídas; verifique se 0.75m saiu de 0/9 e a faixa entre seeds."
echo "fim: E3 relloss s33 + s7"
