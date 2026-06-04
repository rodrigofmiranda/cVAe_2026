set -e
cd /workspace/2026/feat_seq_bigru_residual_cvae
source scripts/ops/container_bootstrap_python.sh
python3 -c "import os,numpy,tensorflow as tf;print('[twin] numpy',numpy.__version__,'tf',tf.__version__,'HOME',os.environ.get('HOME'))"
echo "[twin] DET=$CVAE_DETERMINISTIC ENV=$CVAE_DET_ENV OPDET=$CVAE_DET_OPDET SETSEED=$CVAE_DET_SETSEED seed=$REPRO_SEED sub=$REPRO_SUB"
python3 -u -m src.protocol.run \
  --dataset_root /workspace/2026/feat_seq_bigru_residual_cvae/data/dataset_fullsquare_organized \
  --output_base  /workspace/2026/feat_seq_bigru_residual_cvae/outputs/twin_search/${REPRO_SUB} \
  --protocol     configs/protocol_default.json \
  --train_once_eval_all \
  --grid_preset  seq_edgegap_targeted_short \
  --grid_tag     S39B_edgegap_lowlr_all08_w18_p120 \
  --no_data_reduction \
  --seed "${REPRO_SEED}" --stat_tests --stat_mode quick --stat_seed "${REPRO_SEED}" \
  --keras_verbose 2
echo "[twin] DONE rc=$?"
