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
7. [docs/NOISE_DISTRIBUTION_AUDIT.md](NOISE_DISTRIBUTION_AUDIT.md)

Everything else is secondary unless a specific task requires it.

## Branch Focus

Active branch:

- `feat/sample-aware-mmd`

Current purpose:

- keep `outputs/exp_20260324_023558` as the stable seq reference while testing
  a targeted objective/diagnostic intervention
- instrument the residual distribution at three levels:
  - regime
  - axis (`I/Q`)
  - amplitude bin
- compare `mean_residual` vs `sampled_residual` MMD under the same protocol
- use `configs/all_regimes_sel4curr.json` as the minimum active protocol
  (`0.8/1.0/1.5 m x 100/300/500/700 mA`)

Architectures available in this branch:

- `seq_bigru_residual`
- `delta_residual`
- legacy support variants already present in `src/models/cvae.py`

## Current Scientific References

Strongest current `seq_bigru_residual` reference in this workspace:

- `outputs/exp_20260324_023558`
- winner:
  - `S6seq_W7_h64_lat4_b0p003_lmmd1p75_fb0p10_lr0p0003_L128-256-512`
- result summary:
  - `10/12` passes total
  - `2/4` passes at `0.8 m`
  - `4/4` passes at `1.0 m`
  - `4/4` passes at `1.5 m`
  - only `0.8 m / 100 mA` and `0.8 m / 300 mA` remain failing

Recent comparison run that did **not** overtake the reference:

- `outputs/exp_20260323_210309`
- winner:
  - `S4seq_W7_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
- result summary:
  - `6/12` passes total
  - lower `gate_pass_ratio` than `exp_20260322_193738`
  - conclusion:
    - higher `seq_hidden_size` alone did not improve protocol-level science
    - the strongest family remains `W7_h64`

Recent replay run under the new axis-wise diagnostics:

- `outputs/exp_20260324_024442`
- winner:
  - `S4seq_W7_h64_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
- result summary:
  - `8/12` passes total
  - all `1.0 m` regimes pass
  - all `1.5 m` regimes pass
  - all remaining failures stayed concentrated in `0.8 m`
  - conclusion:
    - the new per-axis metrics did not change the leading architecture family
    - they made the remaining `0.8 m` marginal-shape mismatch easier to localize

Current audit note for the remaining noise-shape mismatch:

- [docs/NOISE_DISTRIBUTION_AUDIT.md](NOISE_DISTRIBUTION_AUDIT.md)
- key reading:
  - the near-regime residual histograms are still under-dispersed
  - the current `MMD` term matches residual means, not sampled residual clouds
  - `seq_finish_0p8m` is the last reasonable no-code grid before loss changes

Current instrumentation layer available in this branch:

- `tables/residual_signature_by_regime.csv`
- `tables/residual_signature_by_amplitude_bin.csv`
- `tables/train_regime_diagnostics_history.csv`
- `plots/best_model/residual_signature_overview.png`

New runtime controls:

- `train_regime_diagnostics_enabled`
- `train_regime_diagnostics_every`
- `train_regime_diagnostics_mc_samples`
- `train_regime_diagnostics_max_samples_per_regime`
- `train_regime_diagnostics_amplitude_bins`
- `train_regime_diagnostics_focus_only_0p8m`

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

## Wide Overnight Grid Still Available

The wider protocol-first overnight sweep remains available when we need broad
search instead of local finishing:

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

## Current Replay Grid For New Axis Metrics

When the goal is not broad discovery but rerunning only the strongest seq
candidates under the new axis-wise residual diagnostics, use:

- preset: `seq_replay_axis_diagnostics`
- total: `4` runs
- selected candidates:
  - `S4seq_W7_h64_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
  - `S4seq_W7_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
  - `S4seq_W9_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
  - `S2seq_W7_h64_lat4_b0p001_lmmd1p0_fb0p10_lr0p0003_L128-256-512`

Canonical command:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset seq_replay_axis_diagnostics \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests --stat_mode full --stat_max_n 5000 \
  --no_data_reduction
```

## Current Preferred Finishing Grid For 0.8 m

After the new overnight winner reached `10/12`, the next assertive step is to
attack only the remaining `0.8 m` failures.

- preset: `seq_finish_0p8m`
- total: `6` runs
- fixed family:
  - `W7_h64_lat4`
  - `free_bits=0.10`
  - `L128-256-512`
- scientific intent:
  - keep the new `lambda_mmd=1.75` winner as control
  - probe `lambda_mmd=2.0` directly
  - keep only low-LR / higher-beta hedges that were already strong in the
    overnight train-side diagnostics

Canonical command:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset seq_finish_0p8m \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests --stat_mode full --stat_max_n 5000 \
  --no_data_reduction
```

## Current Causal Grid On This Branch

The experimental branch also carries a small causal comparison preset to test
the objective change before any new weighting or loss term:

- preset: `seq_sampled_mmd_compare`
- intent:
  - keep the winning `W7_h64_lat4_b0.003` family
  - compare `mmd_mode=mean_residual` vs `mmd_mode=sampled_residual`
  - read the new residual-signature outputs, not only the final gates
- recommended run mode:
  - add `--no_baseline`
  - the baseline is not part of the causal decision on this branch
  - keep protocol time focused on the cVAE residual diagnostics

Operational readout for this preset:

- read first:
  - `tables/protocol_leaderboard.csv`
  - `tables/summary_by_regime.csv`
  - `tables/residual_signature_by_regime.csv`
  - `tables/residual_signature_by_amplitude_bin.csv`
- success means:
  - no degradation in `1.0 m` / `1.5 m`
  - improvement in `0.8m/100mA` and `0.8m/300mA`
  - visible reduction of the “too narrow residual histogram” pattern
- if that does not happen:
  - do not spend more grid budget on this objective without adding a new loss term

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
