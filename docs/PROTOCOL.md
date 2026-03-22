# Protocol Runner

Reproducible orchestration of **train + evaluate** across VLC regimes.

The runner supports two distinct execution modes:

- `per_regime_retrain` (default): train one cVAE per regime, then evaluate it.
- `train_once_eval_all` (`--train_once_eval_all`): train one shared global cVAE once, then evaluate that same model across all regimes.

This is the only supported public experiment entrypoint. The old
`src.training.train` path is kept only as a compatibility shim.

## Quick start

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
export PYTHONPATH="$PWD"

# Default reduced 12-regime protocol (3 distances x 4 currents)
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --train_once_eval_all

# Smoke-test on the reduced 12-regime protocol
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --protocol configs/all_regimes_sel4curr.json \
    --train_once_eval_all \
    --max_epochs 2 --max_grids 1 --max_experiments 1

# Dry-run (validate protocol, build model, no training)
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --dry_run

# Final universal-twin mode on the full 27-regime protocol
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --protocol configs/all_regimes_full_dataset.json \
    --train_once_eval_all \
    --max_epochs 120 --max_grids 2

# Reduced 12-regime protocol (0.8/1.0/1.5 m × 100/300/500/700 mA)
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --protocol configs/all_regimes_sel4curr.json \
    --train_once_eval_all \
    --grid_preset best_compare_large \
    --max_epochs 80 \
    --patience 8 \
    --reduce_lr_patience 4 \
    --stat_tests --stat_mode quick \
    --no_data_reduction \
    --no_baseline

# Reuse a previously trained shared model, skip retraining
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --protocol configs/all_regimes_sel4curr.json \
    --train_once_eval_all \
    --reuse_model_run_dir outputs/exp_YYYYMMDD_HHMMSS/train \
    --stat_tests --stat_mode quick \
    --no_baseline

# Mixed-family comparison of the strongest current candidates
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --protocol configs/all_regimes_sel4curr.json \
    --train_once_eval_all \
    --grid_preset best_compare_large \
    --max_epochs 80 \
    --patience 8 \
    --reduce_lr_patience 4 \
    --stat_tests --stat_mode quick \
    --no_data_reduction

# Custom protocol file (explicit regime subset)
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --protocol configs/protocol_default.json

# YAML protocol config
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --protocol_config configs/protocol_default.yaml
```

## Default behaviour — bundled reduced protocol

When **neither** `--protocol` nor `--protocol_config` is provided, the
runner loads the bundled reduced protocol:

```
configs/protocol_default.json
```

That default protocol is the active minimum scientific grid:

- distances: `0.8 / 1.0 / 1.5 m`
- currents: `100 / 300 / 500 / 700 mA`
- total: `12` regimes

Use `configs/all_regimes_full_dataset.json` only when you explicitly want the
full 27-regime sweep.

**Regime ID format:** `dist_{D}m__curr_{C}mA`

- Decimals use `p` instead of `.` (e.g. `0.8` → `0p8`)
- Trailing fractional zeros are stripped (`1.00` → `1`, `0.80` → `0p8`)
- Current is integer mA (`200`, not `200.0`)

Examples:

| Distance | Current | regime_id |
|---|---|---|
| 0.8 m | 200 mA | `dist_0p8m__curr_200mA` |
| 1.0 m | 400 mA | `dist_1m__curr_400mA` |
| 1.5 m | 800 mA | `dist_1p5m__curr_800mA` |

On startup the runner logs the first discovered regimes:

```
🔍 Auto-discovered 12 regime(s) from dataset:
   • dist_0p8m__curr_200mA  (0.8 m, 200.0 mA)
   • dist_0p8m__curr_400mA  (0.8 m, 400.0 mA)
   • dist_0p8m__curr_600mA  (0.8 m, 600.0 mA)
   … and 9 more
```

## Protocol config formats

### JSON (`--protocol`)

```json
{
    "protocol_version": "1.0",
    "description": "Custom 3-regime protocol",
    "global_settings": {
        "seed": 42,
        "val_split": 0.2,
        "max_epochs": 500,
        "psd_nfft": 2048
    },
    "regimes": [
        {
            "regime_id": "dist_0p8m__curr_200mA",
            "description": "0.8 m / 200 mA",
            "distance_m": 0.8,
            "current_mA": 200
        },
        {
            "regime_id": "dist_1p5m__curr_800mA",
            "description": "1.5 m / 800 mA",
            "distance_m": 1.5,
            "current_mA": 800
        }
    ]
}
```

When `distance_m` and `current_mA` are present, `regime_id` is
automatically rewritten to the canonical physical format — any custom
name is preserved as `regime_label`.

### YAML (`--protocol_config`)

```yaml
studies:
  - name: within_regime
    split_strategy: per_experiment
    selectors:
      - distance_m: 0.8
        current_mA: 200
      - distance_m: 1.5
        current_mA: 800
```

### Regime selection

Experiments are matched to regimes by `distance_m` / `current_mA`
with tolerances controlled by `--dist_tol_m` (default 0.05) and
`--curr_tol_mA` (default 25).

When a protocol contains `_selected_experiments`, the runner now matches them
portably by dataset-relative suffix as well as absolute path. This keeps the
same reduced protocol JSON usable across different clone roots, for example:

- `/workspace/2026/feat_seq_bigru_residual_cvae/...`
- `/mnt/clone_a/cVAe_2026/...`

without editing the protocol file.

## Architecture selection inside the protocol

The protocol itself does not hard-code a single model family. Each candidate
in the selected grid carries its own `cfg.arch_variant`, so a single run can
compare multiple architectures under the same scientific protocol.

The main families currently supported are:

- `concat`
- `channel_residual`
- `delta_residual`
- `seq_bigru_residual`
- `legacy_2025_zero_y`

Important constraint:

- `seq_bigru_residual` must be run with `--no_data_reduction`, because the
  windowed model requires contiguous temporal context.

## Outputs

```
outputs/exp_YYYYMMDD_HHMMSS/
├── train/                         (only with --train_once_eval_all)
│   ├── models/
│   ├── plots/
│   │   └── champion/
│   │       └── analysis_dashboard.png
│   │   └── training/
│   │       └── dashboard_analysis_complete.png
│   └── tables/
│       ├── grid_training_diagnostics.csv
│       ├── gridsearch_results.csv
│       └── gridsearch_results.xlsx
├── eval/
│   ├── dist_0p8m__curr_200mA/
│   │   ├── plots/
│   │   └── tables/
│   ├── dist_0p8m__curr_400mA/
│   └── ...
├── manifest.json
├── logs/
│   ├── protocol_input.json
│   ├── protocol_input.yaml   (when YAML config used)
│   ├── train/
│   └── eval/
├── tables/
│   ├── summary_by_regime.csv
│   ├── protocol_leaderboard.csv
│   └── stat_fidelity_by_regime.csv
└── plots/
    └── best_model/
        └── heatmap_gate_metrics_by_regime.png
```

### Output semantics by mode

- `per_regime_retrain`:
  - each regime directory contains both the trained model and its evaluation artifacts
- `train_once_eval_all`:
  - `train/` contains the single shared trained cVAE
  - `eval/` contains only the regime-specific evaluation artifacts produced with that shared model
  - `summary_by_regime.csv` records both `run_dir` (evaluation artifacts) and `model_run_dir` (shared model source)
  - `protocol_leaderboard.csv` is the canonical candidate ranking derived from the same gates/metrics used by the protocol
  - `plots/best_model/heatmap_gate_metrics_by_regime.png` is the canonical scientific visual summary
  - `train/plots/champion/analysis_dashboard.png` is the full dashboard of the winning model
  - `train/plots/training/dashboard_analysis_complete.png` is the operational convergence dashboard for the full grid
  - `train/tables/grid_training_diagnostics.csv` is the compact per-grid diagnostics table
- `train_once_eval_all + --reuse_model_run_dir`:
  - skips the shared-model training phase
  - reuses `models/best_model_full.keras` from the referenced `train/` directory
  - evaluates that same model under a different protocol without retraining
  - does not regenerate the training dashboard; it references the source training run instead

### Summary table columns

| Column | Description |
|---|---|
| `study` | Study name |
| `regime_id` | Physical regime ID (`dist_…m__curr_…mA`) |
| `regime_label` | Original human name (if any) |
| `description` | Human-readable description |
| `dist_target_m` | Target distance (m) |
| `curr_target_mA` | Target current (mA) |
| `model_scope` | `per_regime` or `shared_global` |
| `model_run_dir` | Source directory of the trained model used for this row |
| `best_grid_tag` | Best grid configuration tag (from training) |
| `evm_real_%`, `evm_pred_%`, `delta_evm_%` | EVM metrics |
| `snr_real_db`, `snr_pred_db`, `delta_snr_db` | SNR metrics |
| `delta_mean_l2`, `delta_cov_fro`, `var_real_delta`, `var_pred_delta`, `var_ratio_pred_real` | Residual distribution metrics |
| `delta_skew_l2`, `delta_kurt_l2`, `delta_psd_l2`, `delta_acf_l2`, `jb_p_min`, `jb_log10p_min` | Higher-order stats + gaussianity |
| `baseline_*`, `cvae_*` | Side-by-side baseline vs cVAE validation metrics |
| `stat_mmd2`, `stat_mmd_pval`, `stat_mmd_qval`, `stat_mmd2_normalized` | Formal two-sample MMD outputs |
| `stat_energy`, `stat_energy_pval`, `stat_energy_qval` | Formal Energy test outputs |
| `stat_psd_dist`, `stat_psd_ci_low`, `stat_psd_ci_high` | Spectral fidelity outputs |
| `better_than_baseline_*`, `gate_g1`…`gate_g6`, `validation_status` | Derived acceptance helpers |
| `n_experiments_selected` | How many experiments matched this regime |

## CLI flags

All flags override values from `global_settings` in the protocol config.
CLI flags take precedence.

| Flag | Effect |
|---|---|
| `--protocol PATH` | Explicit JSON protocol (default: bundled reduced 12-regime protocol) |
| `--protocol_config PATH` | YAML protocol config (takes precedence over `--protocol`) |
| `--reuse_model_run_dir PATH` | Reuse a previous shared-model `train/` directory and skip retraining |
| `--max_epochs N` | Limit training epochs |
| `--max_grids N` | Limit grid configurations per regime |
| `--max_regimes N` | Limit regimes executed after protocol resolution |
| `--grid_group REGEX` | Filter grids by group |
| `--grid_tag REGEX` | Filter grids by tag |
| `--max_experiments N` | Limit experiments loaded per regime |
| `--max_samples_per_exp N` | Truncate samples per experiment |
| `--val_split F` | Override validation split |
| `--seed N` | Override random seed |
| `--psd_nfft N` | Override PSD FFT size |
| `--dist_tol_m F` | Distance tolerance for regime matching (default: 0.05 m) |
| `--curr_tol_mA F` | Current tolerance for regime matching (default: 25 mA) |
| `--no_baseline` | Skip deterministic baseline |
| `--no_cvae` | Skip cVAE training/evaluation and keep only baseline outputs |
| `--baseline_only` | Alias for `--no_cvae` |
| `--no_dist_metrics` | Skip distribution-fidelity metrics |
| `--skip_eval` | Run training only, skip evaluation |
| `--dry_run` | Validate + build model, no training |
| `--train_once_eval_all` | Train one shared global model and evaluate it across all regimes |
| `--stat_tests` | Run MMD/Energy/PSD stat tests per regime |
| `--stat_mode quick|full` | `quick` uses lighter defaults (`stat_max_n=5000`), `full` keeps `50000` |
