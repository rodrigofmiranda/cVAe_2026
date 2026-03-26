# Experiment Workflow

This document merges the old diagnosis checklist and run-reanalysis playbook.

Use it for two things:

- launching a controlled experiment
- analyzing a finished `exp_*`

## Canonical Entry Point

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -m src.protocol.run
```

## Before Running

Check:

```bash
git status -sb
git log --oneline -5
pgrep -af "src.protocol.run|src.training.train"
```

## Quick Run Pattern

Use quick caps when screening a hypothesis:

- `--max_samples_per_exp`
- `--max_val_samples_per_exp`
- `--max_dist_samples`
- `--stat_mode quick`
- `--stat_max_n`

## Full Run Pattern

Use the full protocol only after a quick run is already scientifically
promising.

## How To Review A Finished Run

Inspect in this order:

1. `tables/protocol_leaderboard.csv`
2. `tables/summary_by_regime.csv`
3. `tables/residual_signature_by_regime.csv`
4. `tables/stat_fidelity_by_regime.csv`
5. `train/tables/gridsearch_results.csv`

## Questions To Answer

Always answer:

1. Did the run complete?
2. Which candidate won the grid?
3. Which regimes passed or failed?
4. Did the winner improve the current reference?
5. What is the single best next test?

## Rule

Do not accept a line based only on train-side ranking or lower loss.

Protocol result comes first.
