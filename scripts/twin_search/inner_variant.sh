set -e
cd /workspace/2026/feat_seq_bigru_residual_cvae
source scripts/ops/container_bootstrap_python.sh
# Portable knobs (env, with V2 defaults). REPRO_MAX_SAMPLES=0 -> full data.
DS="${REPRO_DATASET_ROOT:-/workspace/2026/feat_seq_bigru_residual_cvae/data/dataset_fullsquare_organized}"
PROTO="${REPRO_PROTOCOL:-configs/protocol_default.json}"
MS="${REPRO_MAX_SAMPLES:-300000}"
CAP=""; [ "$MS" != "0" ] && CAP="--max_samples_per_exp $MS"
echo "[var] tag=$REPRO_TAG seed=$REPRO_SEED sub=$REPRO_SUB cap=${MS} dataset=$DS DET=$CVAE_DETERMINISTIC"
python3 -u -m src.protocol.run \
  --dataset_root "$DS" \
  --output_base  /workspace/2026/feat_seq_bigru_residual_cvae/outputs/twin_search/${REPRO_SUB} \
  --protocol     "$PROTO" \
  --train_once_eval_all \
  --grid_preset  "${REPRO_PRESET:-seq_edgegap_targeted_short}" \
  --grid_tag     "${REPRO_TAG}" \
  --no_data_reduction \
  $CAP \
  --seed "${REPRO_SEED}" --stat_tests --stat_mode quick --stat_seed "${REPRO_SEED}" \
  --keras_verbose 2
echo "[var] DONE rc=$?"
