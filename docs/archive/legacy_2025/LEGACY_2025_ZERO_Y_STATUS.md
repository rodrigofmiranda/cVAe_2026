# Legacy 2025 Zero-Y Status

Historical note:

- this document is archived on purpose
- it references historical local runs and a retired single-regime protocol file
- use current branch docs for active execution guidance

Current status of the experimental `arch_variant="legacy_2025_zero_y"` in the
2026 pipeline.

## Implemented

- dedicated legacy model integration completed
- dedicated standard-normal loss path completed
- evaluation/reporting adapted for standard-normal latent semantics
- save/load path validated for full model, `prior_net`, and `decoder`
- grid presets currently available:
  - `legacy2025_smoke`
  - `legacy2025_ref`
  - `legacy2025_batch_sweep`
  - `legacy2025_large`

## Important historical note

This run is invalid as scientific reference:

- `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_190337`

Reason:

- before commit `f8f752c`, train reduction could reduce `X/Y` while truncating
  `D/C` by length instead of applying the same sampled indices
- this corrupted condition-label alignment and collapsed normalization in the run

Observed bad normalization in the invalid run:

- `D_min = D_max = 0.8`
- `C_min = C_max = 100`

The bug was fixed in:

- `/workspace/2026/feat_seq_bigru_residual_cvae/src/data/loading.py`
- `/workspace/2026/feat_seq_bigru_residual_cvae/src/training/pipeline.py`
- `/workspace/2026/feat_seq_bigru_residual_cvae/tests/test_data_reduction_alignment.py`

Post-fix spot-check on the intended 12-group reduced subset:

- `TRAIN_UNIQUE_D = [0.8, 1.0, 1.5]`
- `TRAIN_UNIQUE_C = [100, 300, 500, 700]`
- `NORM_PARAMS = {D_min: 0.8, D_max: 1.5, C_min: 100, C_max: 700}`

## Valid reference runs

### Reduced 4-current pivot benchmark

- valid reference run:
  - `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_193036`
- protocol:
  - `configs/one_regime_1p0m_300mA_sel4curr.json`
- architecture:
  - `legacy2025_ref`
- reference result:
  - `ΔEVM = -1.8946 pp`
  - `ΔSNR = +0.5557 dB`
  - `Δmean_l2 = 0.02266`
  - `cVAE PSD_L2 = 0.25219`
  - `MMD² = 0.004884`

### Batch-size sweep

The batch-size sweep results are recorded in:

- `/workspace/2026/feat_seq_bigru_residual_cvae/docs/LEGACY_2025_BATCHSIZE_RESULTS.md`

Key runs:

- accepted ceiling:
  - `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_195010`
  - `batch_size = 8192`
- rejected escalation:
  - `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_195709`
  - `batch_size = 16384`

Current operational conclusion:

- use `batch_size = 8192` for wider reduced-data legacy sweeps

## Current run in progress

Scientific screening currently running:

- run:
  - `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_204208`
- launch log:
  - `/workspace/2026/feat_delta_residual_adv/outputs/legacy2025_large_sel4curr.launch.log`
- mode:
  - `train_once_eval_all`
- protocol:
  - `configs/one_regime_1p0m_300mA_sel4curr.json`
- preset:
  - `legacy2025_large`

This sweep uses:

- reduced training subset:
  - currents `100, 300, 500, 700 mA`
  - distances `0.8, 1.0, 1.5 m`
- evaluation regime:
  - `1.0 m / 300 mA`
- grid width:
  - `12` configurations
- fixed batch size:
  - `8192`

Search factors:

- layer sizes:
  - `[32,64,128,256]`
  - `[64,128,256,512]`
- latent dims:
  - `8`, `16`, `24`
- beta:
  - `0.03`, `0.1`

## Output hygiene

On 2026-03-18, `outputs/` was cleaned to remove:

- smoke runs already consumed
- dry-runs
- exploratory leftovers
- many incomplete/interrupted `exp_*`
- all old `run_*` directories

The result is that the current references above were intentionally preserved,
while the rest of the workspace output footprint was reduced substantially.

## Next action

Wait for the current large reduced-data sweep to finish, then compare the
top-ranked candidates against the valid reference run
`/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_193036`.

Primary comparison columns:

- `best_val_loss`
- `cvae_delta_evm_%`
- `cvae_delta_snr_db`
- `cvae_delta_mean_l2`
- `cvae_psd_l2`
- `stat_mmd2`

First visual inspection target in the winning regime:

- `plots/README.txt`
- `plots/reports/summary_report.png`
- `plots/core/overlay_constellation.png`
- `plots/core/overlay_residual_delta.png`
