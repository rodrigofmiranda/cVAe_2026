set -e
export PYTHONPATH=/workspace/2026/feat_seq_bigru_residual_cvae/.pydeps
export CVAE_PYDEPS_DIR=/workspace/repro_outputs/.pydeps
export CVAE_MPLCONFIGDIR=/workspace/repro_outputs/.mplconfig
export CVAE_PIP_CACHE_DIR=/workspace/repro_outputs/.cache/pip
export KERAS_HOME=/workspace/repro_outputs/.keras
source scripts/ops/container_bootstrap_python.sh
echo "[iso] DET=$CVAE_DETERMINISTIC ENV=$CVAE_DET_ENV OPDET=$CVAE_DET_OPDET SETSEED=$CVAE_DET_SETSEED sub=$REPRO_OUT_SUBDIR"
python3 -u -m src.protocol.run \
  --dataset_root /workspace/2026/feat_seq_bigru_residual_cvae/data/dataset_fullsquare_organized \
  --output_base  /workspace/repro_outputs/${REPRO_OUT_SUBDIR} \
  --protocol     configs/protocol_default.json \
  --train_once_eval_all \
  --grid_preset  seq_edgegap_targeted_short \
  --grid_tag     S39B_edgegap_lowlr_all08_w18_p120 \
  --no_data_reduction \
  --seed 42 --stat_tests --stat_mode quick --stat_seed 42 \
  --keras_verbose 2 --max_epochs 2 --skip_eval 2>&1 | grep -vE "NUMA|ptx85|cuda_executor|StreamExecutor|cuFFT|cuDNN factory|cuBLAS" || true
echo "[iso] DONE"
