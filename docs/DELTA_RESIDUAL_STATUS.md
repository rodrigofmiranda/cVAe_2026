# Delta Residual Status

Current status of the experimental `arch_variant="delta_residual"` under the
strict 2026 protocol.

## Goal

Test the simplified hypothesis:

- train the model on `Δ = Y - X`
- keep inference as `Ŷ = X + Δ̂`
- stay point-wise first
- use the reduced 4-current protocol before any full-data expansion

## Implemented

- new variant integrated in:
  - `src/models/cvae.py`
  - `src/models/losses.py`
  - `src/evaluation/report.py`
  - `src/evaluation/engine.py`
  - `src/training/grid_plan.py`
- save/load compatibility preserved
- point-wise `concat` and `channel_residual` remained compatible after validation

Implementation commit:

- `c32ee63` `feat(delta): add explicit residual-target cvae variant`

## Reference Protocol

- protocol: `configs/one_regime_1p0m_300mA_sel4curr.json`
- currents: `100/300/500/700 mA`
- distances: `0.8/1.0/1.5 m`
- pivot regime: `1.0 m / 300 mA`
- training mode: `train_once_eval_all`
- train reduction: `200k`

## Runs

### 1. Operational Smoke

- run: `/workspace/2026/outputs/exp_20260318_230955`
- preset: `delta_residual_smoke`

Outcome:

- variant trained and evaluated end-to-end
- scientifically weak
- pivot summary:
  - `ΔEVM=+2.109 pp`
  - `ΔSNR=-0.579 dB`
  - `cvae_delta_mean_l2=0.0255`
  - `cvae_psd_l2=0.7528`
  - `MMD²=0.034140`
  - `Energy=0.020307`

Use only as operational proof that the variant works in the full pipeline.

### 2. First Useful Sweep

- run: `/workspace/2026/outputs/exp_20260318_231458`
- preset: `delta_residual_small`

Winning config:

- `D1delta_lat4_b0p001_fb0p0_lr0p0003_L128-256-512`

Pivot summary:

- `ΔEVM=-3.512 pp`
- `ΔSNR=+1.060 dB`
- `baseline_delta_mean_l2=0.00295`
- `cvae_delta_mean_l2=0.00841`
- `baseline_psd_l2=0.22294`
- `cvae_psd_l2=0.24760`
- `MMD²=0.001245`
- `Energy=0.000682`
- `validation_status=fail`

Interpretation:

- this is the best `delta_residual` scientific reference so far
- still not a validated digital twin
- but much stronger than the initial smoke and stronger than the recent low-beta refinement

### 3. Winner-Centric Refinement

- run: `/workspace/2026/outputs/exp_20260318_233023`
- preset: `delta_residual_refine`

Winning config by grid score:

- `D2delta_lat4_b0p0005_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`

Pivot summary:

- `ΔEVM=+1.551 pp`
- `ΔSNR=-0.430 dB`
- `baseline_delta_mean_l2=0.00276`
- `cvae_delta_mean_l2=0.00358`
- `baseline_psd_l2=0.21674`
- `cvae_psd_l2=0.74423`
- `MMD²=0.025407`
- `Energy=0.015610`
- `validation_status=fail`

Interpretation:

- this refinement did **not** beat `/workspace/2026/outputs/exp_20260318_231458`
- lowering `beta` to `0.0005` looked acceptable on grid rank metrics, but degraded the final pivot regime badly

### 4. Local Winner Refinement

- run: `/workspace/2026/outputs/exp_20260318_235319`
- preset: `delta_residual_local`

Winning config by grid score:

- `D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`

Pivot summary:

- `ΔEVM=-3.678 pp`
- `ΔSNR=+1.113 dB`
- `baseline_delta_mean_l2=0.00342`
- `cvae_delta_mean_l2=0.00903`
- `baseline_psd_l2=0.22519`
- `cvae_psd_l2=0.23994`
- `MMD²=0.001129`
- `Energy=0.000787`
- `validation_status=fail`

Interpretation:

- this run is the best current `delta_residual` scientific reference
- it improved over `/workspace/2026/outputs/exp_20260318_231458` on:
  - `ΔEVM`
  - `ΔSNR`
  - `PSD_L2`
  - `MMD²`
- it was slightly worse on:
  - `delta_mean_l2`
  - `Energy`
- it still does not pass the strict protocol gates, so it is not yet a validated digital twin

## Current Best Reference

Use this as the best current `delta_residual` reference:

- `/workspace/2026/outputs/exp_20260318_235319`

Current best config:

- `arch_variant=delta_residual`
- `layer_sizes=[128,256,512]`
- `latent_dim=5`
- `beta=0.001`
- `free_bits=0.0`
- `lr=3e-4`
- `batch_size=16384`
- `kl_anneal_epochs=80`

## What We Learned

- the simplified residual-target idea is viable
- `free_bits=0.0` helped this line
- `beta=0.001` is currently better than `0.0005` for final scientific metrics
- a very small latent space still seems best; the current best run uses `latent_dim=5`
- grid-ranking metrics alone are not enough; the pivot regime can disagree sharply

## Next Likely Step

Use the historical cross-reference to open a new frontier instead of repeating
the already tested center:

- preset: `delta_residual_frontier`
- keep:
  - `layer_sizes=[128,256,512]`
  - `lr=3e-4`
  - `batch_size=16384`
  - `kl_anneal_epochs=80`
- vary:
  - `latent_dim in {4,5,6}`
  - `beta in {0.0007, 0.00085, 0.00115}`
  - `free_bits in {0.0, 0.02, 0.05}`
- compare against `/workspace/2026/outputs/exp_20260318_235319`
