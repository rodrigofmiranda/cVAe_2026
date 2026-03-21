# Legacy 2025 Zero-Y Runbook

This runbook captures the next execution steps for the experimental
`arch_variant="legacy_2025_zero_y"` port in the 2026 pipeline.

## Goal

Validate the old 2025 point-wise heteroscedastic architecture under the
strict 2026 protocol, without changing the online default architecture.

## 1. Smoke train

Use this first to confirm that the legacy variant still trains end-to-end in
the canonical training entrypoint.

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -m src.training.train \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --run_id legacy2025_zero_y_smoke \
  --grid_preset legacy2025_smoke \
  --max_grids 1 \
  --max_experiments 1 \
  --max_epochs 3 \
  --keras_verbose 2
```

Expected outcome:
- single-grid smoke run
- `arch_variant=legacy_2025_zero_y`
- saved full model contains `encoder`, `prior_net`, `decoder`

## 2. Pivot-regime benchmark

This is the first scientific comparison step for the legacy port.

Protocol:
- global training subset: 4 currents `(100, 300, 500, 700 mA)` across 3 distances
- evaluation regime: `1.0 m / 300 mA`
- protocol file: `configs/one_regime_1p0m_300mA_sel4curr.json`

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA_sel4curr.json \
  --train_once_eval_all \
  --grid_preset legacy2025_ref \
  --max_grids 1 \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests \
  --stat_mode quick
```

Expected outcome:
- one shared legacy-2025 model trained once on the selected 12 training groups
- evaluation focused on the pivot regime `dist_1m__curr_300mA`
- canonical protocol artifacts, including:
  - `manifest.json`
  - `tables/summary_by_regime.csv`
  - `logs/latent_summary.json`
  - `logs/metricas_globais_reanalysis.json`

## 3. Larger grid search on the reduced 4-current subset

Use this when the goal is to screen whether the legacy architecture is a real
candidate before attempting any all-currents run.

Protocol:
- same reduced training subset as the pivot benchmark
- 4 currents `(100, 300, 500, 700 mA)` across 3 distances
- one pivot evaluation regime: `1.0 m / 300 mA`
- larger search preset: `legacy2025_large`

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
PYTHONPATH=/workspace/2026/feat_seq_bigru_residual_cvae python3.8 -u -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA_sel4curr.json \
  --train_once_eval_all \
  --grid_preset legacy2025_large \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --keras_verbose 0 \
  --stat_tests \
  --stat_mode quick
```

Preset design:
- `12` grid points
- layer sizes: `[32,64,128,256]`, `[64,128,256,512]`
- latent dims: `8`, `16`, `24`
- beta: `0.03`, `0.1`
- fixed operational batch size: `8192`
- fixed `lr=1e-4`, `anneal=50`, `free_bits=0.0`

Reasoning:
- keep the data subset small enough to afford a wider search
- keep `batch_size=8192`, because the batch-size sweep showed it was the
  largest value that stayed stable on the reduced protocol
- vary only capacity and KL pressure, which are the main unknowns now

## Notes

- `legacy2025_smoke` is a cheap validation preset; do not interpret it
  scientifically.
- `legacy2025_ref` is the benchmark preset closest to the old 2025 setup:
  `layer_sizes=[32,64,128,256]`, `latent_dim=16`, `beta=0.1`,
  `lr=1e-4`, `batch_size=4096`, `dropout=0.0`, `kl_anneal_epochs=50`,
  `free_bits=0.0`.
- For batch-size scaling around that same reference config, use the dedicated
  protocol in [LEGACY_2025_BATCHSIZE_PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/LEGACY_2025_BATCHSIZE_PROTOCOL.md).
- For this variant, latent diagnostics intentionally report `KL(q||p)` as
  `n/a`, because the port uses `KL(q||N(0,I))` rather than a learned
  conditional prior.
