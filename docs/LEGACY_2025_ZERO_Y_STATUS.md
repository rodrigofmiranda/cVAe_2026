# Legacy 2025 Zero-Y Status

Updated after commit `f8f752c` (`fix(data): keep condition labels aligned after train reduction`).

## Implemented

- New experimental variant: `arch_variant="legacy_2025_zero_y"`
- Dedicated legacy model/loss integration completed
- Grid presets added:
  - `legacy2025_smoke`
  - `legacy2025_ref`
- Evaluation/reporting adapted for standard-normal latent semantics
- Save/load path validated for full model, `prior_net`, and `decoder`

## Verified runs

- CPU smoke:
  - `/workspace/2026/outputs/legacy2025_zero_y_smoke_cpu`
  - status: completed
  - purpose: end-to-end pipeline validation without GPU

- GPU smoke:
  - `/workspace/2026/outputs/legacy2025_zero_y_smoke_gpu`
  - status: completed
  - purpose: end-to-end training validation on real GPU path

## Important finding

The first pivot benchmark below is **not scientifically valid**:

- `/workspace/2026/outputs/exp_20260318_190337`

Reason:
- during post-split train reduction, `X/Y` were reduced but `D/C` were only
  truncated by length instead of being reduced with the same selected indices
- this misaligned condition labels and corrupted the condition normalization
  stored in `state_run.json`

Observed bad normalization in the invalid run:
- `D_min = D_max = 0.8`
- `C_min = C_max = 100`

That is incompatible with the intended 12-group training subset:
- distances: `0.8, 1.0, 1.5`
- currents: `100, 300, 500, 700`

## Fix applied

Files:
- `/workspace/2026/src/data/loading.py`
- `/workspace/2026/src/training/pipeline.py`
- `/workspace/2026/tests/test_data_reduction_alignment.py`

What changed:
- train reduction now uses the same sampled indices for all aligned arrays
  (`X`, `Y`, `D`, `C`)
- normalization params now reflect the actual reduced training subset

Post-fix spot-check on the same 12-group subset:
- `TRAIN_UNIQUE_D = [0.8, 1.0, 1.5]`
- `TRAIN_UNIQUE_C = [100, 300, 500, 700]`
- `NORM_PARAMS = {D_min: 0.8, D_max: 1.5, C_min: 100, C_max: 700}`

## Next action

Rerun the same pivot benchmark after the alignment fix, without changing:
- protocol subset
- architecture preset
- train reduction cap (`200k`)

Command basis:
- protocol: `configs/one_regime_1p0m_300mA_sel4curr.json`
- preset: `legacy2025_ref`
