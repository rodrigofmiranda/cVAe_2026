# Protocol Runner (Commit 3J)

Reproducible orchestration of **train + evaluate** across VLC regimes.

## Quick start

```bash
cd /workspace/2026
export PYTHONPATH="$PWD"

# Run default protocol (3 regimes: near/low, near/high, far/low)
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs

# Smoke-test: 1 grid, 2 epochs, 1 experiment per regime
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --max_epochs 2 --max_grids 1 --max_experiments 1

# Dry-run (validate protocol, build model, no training)
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --dry_run

# Custom protocol file
python -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base  outputs \
    --protocol configs/my_protocol.json
```

## Protocol JSON format

```json
{
    "protocol_version": "1.0",
    "description": "My experiment protocol",
    "global_settings": {
        "seed": 42,
        "val_split": 0.2,
        "max_epochs": 500,
        "max_grids": null,
        "grid_group": null,
        "grid_tag": null,
        "max_experiments": null,
        "max_samples_per_exp": null,
        "psd_nfft": 2048
    },
    "regimes": [
        {
            "regime_id": "near_low",
            "description": "0.8 m / 200 mA",
            "experiment_paths": ["dist_0.8m/curr_200mA"]
        },
        {
            "regime_id": "custom_regex",
            "description": "All 400 mA experiments",
            "experiment_regex": "curr_400mA"
        }
    ]
}
```

### Regime definition

Each regime supports two modes:

| Field | Type | Description |
|---|---|---|
| `experiment_paths` | `list[str]` | Explicit relative paths under `DATASET_ROOT` |
| `experiment_regex` | `str` | Regex matched against discovered experiment paths |

Use `experiment_paths` for deterministic, reproducible protocols.

## Outputs

```
outputs/protocol_YYYYMMDD_HHMMSS/
├── manifest.json              # git hash, versions, args, regime statuses
├── logs/
│   └── protocol_input.json    # copy of the protocol JSON used
└── tables/
    ├── summary_by_regime.csv  # consolidated metrics per regime
    └── summary_by_regime.xlsx
```

Each regime's training run is stored separately under `outputs/protocol_<regime_id>_<ts>/`.

### Summary table columns

| Column | Description |
|---|---|
| `regime_id` | Identifier from protocol |
| `best_grid_tag` | Best grid configuration tag (from training) |
| `evm_real_%`, `evm_pred_%`, `delta_evm_%` | EVM metrics |
| `snr_real_db`, `snr_pred_db`, `delta_snr_db` | SNR metrics |
| `delta_mean_l2`, `delta_cov_fro` | Residual distribution metrics |
| `delta_skew_l2`, `delta_kurt_l2`, `delta_psd_l2` | Higher-order stats |
| `kl_q_to_p_total`, `kl_p_to_N_total` | Latent KL diagnostics |

## CLI flags

All flags override values from `global_settings` in the protocol JSON.
CLI flags take precedence.

| Flag | Effect |
|---|---|
| `--max_epochs N` | Limit training epochs |
| `--max_grids N` | Limit grid configurations per regime |
| `--grid_group REGEX` | Filter grids by group |
| `--grid_tag REGEX` | Filter grids by tag |
| `--max_experiments N` | Limit experiments loaded |
| `--max_samples_per_exp N` | Truncate samples per experiment |
| `--val_split F` | Override validation split |
| `--seed N` | Override random seed |
| `--psd_nfft N` | Override PSD FFT size |
| `--skip_eval` | Run training only |
| `--dry_run` | Validate + build model, no training |
