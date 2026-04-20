# Run Reanalysis Playbook

Use this when a new `exp_...` is copied into `outputs/`.

## Goal

Answer quickly:

1. Did the run finish?
2. Which candidate won the grid?
3. Was training healthy or noisy?
4. How did the winner behave across regimes?
5. What should change in the next grid?

## Primary Tool

```bash
cd /home/rodrigo/cVAe_2026_mdn_return
python scripts/summarize_experiment.py outputs/exp_YYYYMMDD_HHMMSS
```

Or simply:

```bash
python scripts/summarize_experiment.py
```

That uses the latest `outputs/exp_*`.

## What To Read First

### If training completed

1. `train/tables/gridsearch_results.csv`
2. `train/tables/grid_training_diagnostics.csv`
3. `train/plots/training/dashboard_analysis_complete.png`

### If protocol evaluation completed

1. `tables/summary_by_regime.csv`
2. `tables/protocol_leaderboard.csv`
3. `plots/best_model/heatmap_gate_metrics_by_regime.png`

## Decision Heuristics

### Training-side

Good signs:

- winner stable across reruns
- `active_dim_ratio` healthy
- no `flag_posterior_collapse`
- no obvious `flag_overfit`
- `best_epoch_ratio` not pinned to the end

Suspicious signs:

- repeated `flag_undertrained`
- repeated `flag_lr_floor` plus meaningful late improvement
- strong `flag_unstable` across the best candidates

### Protocol-side

Good signs:

- one family wins consistently
- clear regime pattern rather than random failure
- heatmap shows the same weak region repeatedly

Suspicious signs:

- missing `summary_by_regime.csv`
- `eval_status=failed`
- gate columns full of `NaN`

## Current Interpretation Pattern

For the recent seq winner:

- `1.5 m` is strong
- `0.8 m` is the hard zone
- this points more toward context/capacity/regularization than latent collapse

If the task is about reviving the archived adversarial strategy:

- first read [docs/archive/ideas/FUTURE_ADVERSARIAL_STRATEGY.md](../archive/ideas/FUTURE_ADVERSARIAL_STRATEGY.md)
- compare only against:
  - current best point-wise non-adversarial reference
  - current best seq reference

## Manual Checks When Needed

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('outputs/exp_YYYYMMDD_HHMMSS/tables/summary_by_regime.csv')
print(df[['regime_id', 'validation_status']])
print(df.groupby('dist_target_m')['validation_status'].value_counts())
PY
```

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('outputs/exp_YYYYMMDD_HHMMSS/train/tables/grid_training_diagnostics.csv')
print(df[['rank','tag','score_v2','best_epoch_ratio','active_dim_ratio','flag_undertrained','flag_unstable']].head())
PY
```
