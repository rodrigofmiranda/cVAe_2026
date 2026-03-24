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
- use `configs/all_regimes_sel4curr.json` as the minimum active protocol
  (`0.8/1.0/1.5 m x 100/300/500/700 mA`)
- single-regime protocols were retired from the active path

Architectures available in this branch:

- `seq_bigru_residual`
- `delta_residual`
- legacy support variants already present in `src/models/cvae.py`

## Current Scientific References

Strongest current `seq_bigru_residual` reference in this workspace:

- `outputs/exp_20260322_193738`
- winner:
  - `S4seq_W7_h64_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
- result summary:
  - `6/12` passes total
  - `0/4` passes at `0.8 m`
  - `2/4` passes at `1.0 m`
  - `4/4` passes at `1.5 m`

Recent comparison run that did **not** overtake the reference:

- `outputs/exp_20260323_210309`
- winner:
  - `S4seq_W7_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
- result summary:
  - `6/12` passes total
  - lower `gate_pass_ratio` than `exp_20260322_193738`
  - conclusion:
    - higher `seq_hidden_size` alone did not improve protocol-level science
    - the strongest family remains `W7_h64` with `lambda_mmd=1.25`

Previous strong multi-regime `seq_bigru_residual` reference:

- `outputs/exp_20260320_171510`
- winner:
  - `S2seq_W7_h64_lat4_b0p003_lmmd1p0_fb0p10_lr0p0003_L128-256-512`

Historical all-gates-passed `seq_bigru_residual` reference:

- `outputs/exp_20260318_204149`

Best current non-adversarial point-wise reference:

- historical run id `exp_20260318_235319`

Current strong point-wise anchor carried into comparisons:

- `COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256`

## Current Recommended Overnight Grid

Next protocol-first overnight run:

- preset: `seq_overnight_12h`
- target duration:
  - about `10` to `12` hours on the recent A6000-class setup
- scientific goal:
  - keep the winning `W7_h64` family
  - test lower initial learning rates for stability
  - test stronger `lambda_mmd` for the hard `0.8 m` regimes
  - keep only a small low-LR `W9_h96` hedge block

Canonical command:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset seq_overnight_12h \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests --stat_mode full --stat_max_n 5000 \
  --no_data_reduction
```

## Future Adversarial Note

- the adversarial line was removed from the active worktree
- the old local adversarial folder was deleted to save disk space
- the branch name `feat/delta-residual-adv` remains available only as historical traceability
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
