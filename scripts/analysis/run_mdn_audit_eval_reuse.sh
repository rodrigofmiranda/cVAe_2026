#!/usr/bin/env bash
# MDN decomposition audit (eval-only reuse, no retrain) of the S35C campaign
# model over the 12-regime V3 protocol. Emits logs/eval/<regime>/mdn_audit.json
# per regime via the instrumented engine (mdn_decomposition_audit).
# NOTE: no determinism env — eval-only reuse + CVAE_DETERMINISTIC=1 crashes
# (reuse path skips pipeline set_seed; "Random ops require a seed").
# No --stat_tests: G6 not needed for the audit, saves the permutation cost.
set -u
REPO=/home/rodrigo/cVAe_2026_full_square_v3det
WORKDIR=/workspace/2026/feat_seq_bigru_residual_cvae
OUT=outputs/mdn_audit_s35c_20260611
MODEL_DIR=outputs/v3_campaign_reduced_20260611/s35c/exp_20260611_181112/train
CNAME=cvae_v3_mdn_audit

mkdir -p "$REPO/$OUT"
docker rm -f "$CNAME" >/dev/null 2>&1 || true

docker run -d --rm --name "$CNAME" \
  --runtime=nvidia --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TF_CPP_MIN_LOG_LEVEL=2 -e CVAE_BOOTSTRAP_PLOT_DEPS=0 \
  -u "$(id -u):$(id -g)" \
  -e HOME="$WORKDIR" \
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
      --dataset_root /data/Dataset/V3/FULLSQUARE_2026_V3_ORGANIZED \
      --output_base $OUT \
      --protocol configs/protocol_v3_fullsquare.json \
      --train_once_eval_all \
      --reuse_model_run_dir $MODEL_DIR \
      --grid_preset seq_cond_embed_fast_stage1 --grid_tag S35C_fast_e64_base \
      --seed 33 --no_data_reduction --max_samples_per_exp 200000 \
      --train_regime_diagnostics_focus_only_0p8m 0 \
      > $OUT/run.log 2>&1
    curl -fsS -H 'Title: 🔬 MDN audit S35C concluído' \
      -d 'eval-only reuse 12 regimes; mdn_audit.json por regime em $OUT' \
      https://ntfy.sh/projeto_vlc_ia >/dev/null 2>&1 || true
  "
