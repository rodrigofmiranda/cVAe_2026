# Active Context

Use this file to avoid reading the whole repo every time.

## Read Order

Read only these files first:

1. [CODEX.md](../CODEX.md)
2. [PROJECT_STATUS.md](../PROJECT_STATUS.md)
3. [docs/PROTOCOL.md](PROTOCOL.md)
4. [docs/DELTA_RESIDUAL_STATUS.md](DELTA_RESIDUAL_STATUS.md)
5. [docs/RUN_REANALYSIS_PLAYBOOK.md](RUN_REANALYSIS_PLAYBOOK.md)
6. [docs/FUTURE_ADVERSARIAL_STRATEGY.md](FUTURE_ADVERSARIAL_STRATEGY.md)

Everything else is secondary unless a specific task requires it.

## Branch Focus

Active branch:

- `feat/seq-bigru-residual-cvae`

Current purpose:

- keep a single active research branch for the supported architectures
- choose the experiment family by `arch_variant`, `grid_tag`, or `grid_preset`
- keep `seq_bigru_residual` and `delta_residual` comparisons inside the
  same protocol-first workflow

Architectures available in this branch:

- `seq_bigru_residual`
- `delta_residual`
- legacy support variants already present in `src/models/cvae.py`

## Current Scientific References

Strongest current `seq_bigru_residual` reference in this workspace:

- `outputs/exp_20260320_171510`
- winner:
  - `S2seq_W7_h64_lat4_b0p003_lmmd1p0_fb0p10_lr0p0003_L128-256-512`

Historical all-gates-passed `seq_bigru_residual` reference:

- `outputs/exp_20260318_204149`

Best current non-adversarial point-wise reference:

- `outputs/exp_20260318_235319`

Current strong point-wise anchor carried into comparisons:

- `COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256`

## Future Adversarial Note

- the adversarial line was removed from the active worktree
- the historical implementation remains preserved in
  `/workspace/2026/feat_delta_residual_adv`
- if we need to bring that strategy back later, use
  [docs/FUTURE_ADVERSARIAL_STRATEGY.md](FUTURE_ADVERSARIAL_STRATEGY.md)
  as the single implementation note

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
cd /workspace/2026/feat_seq_bigru_residual_cvae
python scripts/summarize_experiment.py outputs/exp_YYYYMMDD_HHMMSS
```

If you do not pass a path, it uses the latest `outputs/exp_*`.
