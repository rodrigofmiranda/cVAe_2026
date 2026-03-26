# Legacy 2025 Batch-Size Protocol

Historical note:

- this document is archived on purpose
- it references a retired single-regime protocol file and historical local runs
- do not use it as the active execution guide for the current branch

This protocol measures how far the `legacy_2025_zero_y` model can increase
`batch_size` before training quality degrades materially under the same 2026
evaluation setup.

## Goal

Answer a narrow operational question:

- starting from the current legacy reference config (`batch_size=4096`)
- how far can batch size be increased
- while keeping training stable and final quality close to the reference

This is not a new architecture benchmark. Only `batch_size` changes.

## Fixed Conditions

Keep these fixed for every run:

- model variant: `arch_variant="legacy_2025_zero_y"`
- architecture: `layer_sizes=[32,64,128,256]`, `latent_dim=16`
- loss settings: `beta=0.1`, `free_bits=0.0`, `kl_anneal_epochs=50`
- optimizer: `lr=1e-4`
- data subset: [one_regime_1p0m_300mA_sel4curr.json](/workspace/2026/feat_seq_bigru_residual_cvae/configs/one_regime_1p0m_300mA_sel4curr.json)
- train reduction: default `balanced_blocks` with target `200k` samples/experiment
- protocol mode: `--train_once_eval_all`
- statistical suite: `--stat_tests --stat_mode quick`

The new preset for this sweep is:

- `legacy2025_batch_sweep`

It contains exactly these batch sizes:

- `4096`
- `8192`
- `16384`
- `32768`
- `65536`

## Execution Model

Do not run the whole sweep as one multi-grid protocol if the goal is scientific
comparison per batch size. In `protocol.run`, the detailed regime evaluation is
reported only for the selected winning grid.

Instead, run one protocol execution per batch size using:

- `--grid_preset legacy2025_batch_sweep`
- `--grid_tag bs<value>_`
- `--max_grids 1`

That yields one fully evaluated protocol run per batch size.

## Commands

Reference run first:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
PYTHONPATH=/workspace/2026/feat_seq_bigru_residual_cvae python3.8 -u -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA_sel4curr.json \
  --train_once_eval_all \
  --grid_preset legacy2025_batch_sweep \
  --grid_tag 'bs4096_' \
  --max_grids 1 \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests \
  --stat_mode quick \
  2>&1 | tee outputs/legacy2025_bs4096.launch.log
```

Then escalate one step at a time:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
for bs in 8192 16384 32768 65536; do
  PYTHONPATH=/workspace/2026/feat_seq_bigru_residual_cvae python3.8 -u -m src.protocol.run \
    --dataset_root data/dataset_fullsquare_organized \
    --output_base outputs \
    --protocol configs/one_regime_1p0m_300mA_sel4curr.json \
    --train_once_eval_all \
    --grid_preset legacy2025_batch_sweep \
    --grid_tag "bs${bs}_" \
    --max_grids 1 \
    --max_epochs 120 \
    --patience 12 \
    --reduce_lr_patience 6 \
    --stat_tests \
    --stat_mode quick \
    2>&1 | tee "outputs/legacy2025_bs${bs}.launch.log"
done
```

## What To Compare

For each run, collect:

- `global_model/tables/gridsearch_results.xlsx`
- `tables/summary_by_regime.csv`
- `tables/stat_fidelity_by_regime.csv`
- `plots/reports/summary_report.png`
- `plots/core/overlay_constellation.png`
- `plots/core/overlay_residual_delta.png`

Primary comparison row:

- `dist_1m__curr_300mA`

Metrics to track:

- `best_val_loss`
- `train_time_s`
- `best_epoch`
- `cvae_delta_evm_%`
- `cvae_delta_snr_db`
- `cvae_delta_mean_l2`
- `cvae_psd_l2`
- `stat_mmd2`
- `stat_energy`

## Acceptance Rule

Use `bs4096` as the reference. A larger batch is considered acceptable only if:

- training finishes without OOM, NaN, or divergence
- `best_val_loss <= 1.05 * ref_best_val_loss`
- `cvae_delta_evm_% <= ref_cvae_delta_evm_% + 1.0`
- `cvae_delta_mean_l2 <= 1.20 * ref_cvae_delta_mean_l2`
- `cvae_psd_l2 <= 1.20 * ref_cvae_psd_l2`
- `stat_mmd2 <= 1.25 * ref_stat_mmd2`

Interpretation:

- `cvae_delta_evm_%` closer to the reference is better; more positive means worse
- the distribution metrics are allowed a small degradation, but not a collapse
- if visual overlays show clear amplitude shrinkage or localized explosion, reject
  the batch even if the scalar thresholds barely pass

Choose the largest batch size that still passes the rule above.

## Stop Conditions

Stop escalating after the first clearly bad step:

- OOM or allocator failure
- `best_val_loss` jumps by more than `5%`
- `cvae_delta_evm_%` worsens by more than `1.0 pp`
- overlay shows obvious mode collapse, amplitude shrinkage, or burst explosion

If `32768` still looks clean, test `65536`. If `65536` fails, keep the last
passing value as the operational ceiling.

## Current Reference

The corrected reference-style legacy run is:

- [exp_20260318_193036](/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_193036)

Useful reference values from that run:

- `cvae_delta_evm_% = -1.8946`
- `cvae_delta_mean_l2 = 0.02266`
- `cvae_psd_l2 = 0.25219`
- `stat_mmd2 = 0.004884`

That makes the initial acceptance ceilings:

- `best_val_loss <= 1.05 * ref_best_val_loss`
- `cvae_delta_evm_% <= -0.8946`
- `cvae_delta_mean_l2 <= 0.02719`
- `cvae_psd_l2 <= 0.30263`
- `stat_mmd2 <= 0.006105`
