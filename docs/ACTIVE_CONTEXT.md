# Active Context

Use this file to avoid reading the whole repo every time.

## Read Order

Read only these files first:

1. [CODEX.md](/workspace/2026/CODEX.md)
2. [PROJECT_STATUS.md](/workspace/2026/PROJECT_STATUS.md)
3. [docs/PROTOCOL.md](/workspace/2026/docs/PROTOCOL.md)
4. [docs/DELTA_RESIDUAL_ADV_STATUS.md](/workspace/2026/docs/DELTA_RESIDUAL_ADV_STATUS.md)
5. [docs/DELTA_RESIDUAL_STATUS.md](/workspace/2026/docs/DELTA_RESIDUAL_STATUS.md)
6. [docs/RUN_REANALYSIS_PLAYBOOK.md](/workspace/2026/docs/RUN_REANALYSIS_PLAYBOOK.md)

Everything else is secondary unless a specific task requires it.

## Branch Focus

Active branch:

- `feat/delta-residual-adv`

Current purpose:

- keep the experimental point-wise cVAE-GAN line aligned with the modern
  protocol-first workflow
- compare it fairly against the strongest non-adversarial references
- keep the seq line as the main scientific benchmark, not as the active code
  path of this branch

## Current Scientific References

Strongest current seq reference in this workspace:

- `/workspace/2026/outputs/exp_20260320_171510`
- winner:
  - `S2seq_W7_h64_lat4_b0p003_lmmd1p0_fb0p10_lr0p0003_L128-256-512`

Historical all-gates-passed seq reference:

- `/workspace/2026/outputs/exp_20260318_204149`

Best current non-adversarial point-wise reference:

- `/workspace/2026/outputs/exp_20260318_235319`

Current strong point-wise anchor carried into comparisons:

- `COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256`

## Adversarial Line Status

`delta_residual_adv` is:

- implemented
- corrected technically by `ee2681f`
- operationally aligned with the current protocol path
- still scientifically pending fresh reruns

Do not use these as final scientific references:

- `/workspace/2026/outputs/exp_20260320_012223`
- `/workspace/2026/outputs/exp_20260320_014614`
- `/workspace/2026/outputs/exp_20260320_020652`

## Canonical Artifacts

Training-side:

- `train/tables/gridsearch_results.csv`
- `train/tables/grid_training_diagnostics.csv`
- `train/plots/training/dashboard_analysis_complete.png`
- `train/plots/champion/analysis_dashboard.png`

Protocol-side:

- `tables/summary_by_regime.csv`
- `tables/protocol_leaderboard.csv`
- `plots/best_model/heatmap_gate_metrics_by_regime.png`

## Fast Reanalysis

For any future run copied into `outputs/`, use:

```bash
cd /workspace/2026
python scripts/summarize_experiment.py outputs/exp_YYYYMMDD_HHMMSS
```

If you do not pass a path, it uses the latest `outputs/exp_*`.
