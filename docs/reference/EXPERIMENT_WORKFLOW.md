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

If the run is happening on another machine or another container image, also
check environment parity before trusting the protocol result:

```bash
python - <<'PY'
import importlib
for name in ["numpy", "tensorflow", "matplotlib"]:
    importlib.import_module(name)
print("environment ok")
PY
```

If `matplotlib` is missing, the dashboard plot is skipped. The protocol result
should still remain valid; missing plots are no longer a reason to discard the
run.

## Quick Run Pattern

Use quick caps when screening a hypothesis:

- `--max_samples_per_exp`
- `--max_val_samples_per_exp`
- `--max_dist_samples`
- `--stat_mode quick`
- `--stat_max_n`

Attention for `seq_bigru_residual`:

- if `--max_samples_per_exp` and/or `--max_val_samples_per_exp` are active,
  confirm the branch contains the post-cap `df_split` fix from commit
  `a1660e2`
- otherwise quick sequential runs can preserve the center sample while still
  leaking temporal context across experiment boundaries inside each window
- this check matters for both:
  - training-side windowing
  - protocol quick evaluation / `_quick_cvae_predict`

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

Also do not accept a quick sequential result from another branch until you have
verified the branch contains the post-cap `df_split` fix if per-experiment caps
were used.

Also inspect `gridsearch_results.csv` for runtime fallback columns when a seq
candidate is run on a new GPU stack. A successful retry onto the compatibility
GRU backend is operationally acceptable, but it should be noted in the run
review.
