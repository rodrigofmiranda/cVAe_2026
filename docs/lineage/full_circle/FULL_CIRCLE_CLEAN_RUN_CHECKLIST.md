# Full Circle Clean Run Checklist

Date: 2026-04-17

Goal: execute the next clean `full_circle` validation run with the smallest
scientifically useful scope.

Status: completed on `RUN_STAMP=20260417_115140`.

## Scope

Run only the previously open clean candidate:

- `S27cov_fc_clean_lc0p25_t0p03_bs8192`

Do not reopen in this step:

- geometry-biased presets
- `disk_geom3`
- hard `disk_l2`
- `cornerness_norm`
- broad support-weight / support-filter sweeps

## Where To Run

Use the parked Full Circle worktree, because the dataset and launcher already
exist there:

- repo: `/home/rodrigo/cVAe_2026_shape_fullcircle`
- dataset: `/home/rodrigo/cVAe_2026_shape_fullcircle/data/FULL_CIRCLE_2026`
- launcher: `scripts/ops/train_full_circle_clean_bs8192_lat10.sh`

## Pre-Run

1. Start the dedicated tmux + Docker stack:

```bash
cd /home/rodrigo/cVAe_2026_shape_fullcircle && \
CVAE_TF25_TMUX_SESSION=fc-clean-next \
CVAE_TF25_CONTAINER_NAME=cvae_fc_clean_next \
scripts/ops/run_tf25_gpu.sh
```

2. Attach to the stack:

```bash
cd /home/rodrigo/cVAe_2026_shape_fullcircle && \
CVAE_TF25_TMUX_SESSION=fc-clean-next \
scripts/ops/enter_tf25_gpu.sh
```

## Launch

Inside the container, run exactly this:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="/workspace/2026/feat_seq_bigru_residual_cvae/outputs/full_circle/${RUN_STAMP}_clean_bs8192_only_100k"
DATASET_ROOT="/workspace/2026/feat_seq_bigru_residual_cvae/data/FULL_CIRCLE_2026" \
OUTPUT_BASE="$OUTPUT_BASE" \
STAT_MODE=quick \
PATIENCE=50 \
REDUCE_LR_PATIENCE=25 \
scripts/ops/train_full_circle_clean_bs8192_lat10.sh \
  --grid_tag '^S27cov_fc_clean_lc0p25_t0p03_bs8192$' \
  --seed 42 | tee "$OUTPUT_BASE/logs/seed42.log"
```

## Follow Logs

Host command to watch the run:

```bash
tail -f /home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/<RUN_STAMP>_clean_bs8192_only_100k/logs/seed42.log
```

## Outcome

Completed result:

- `S27cov_fc_clean_lc0p25_t0p03_bs8192` -> `2/12`
- paired rerun `S27cov_fc_clean_lc0p25_t0p03_lat10` -> `5/12`

Interpretation:

- the missing clean candidate is no longer open
- `bs8192` improved over the clean baseline `1/12`, but remained weaker than
  the clean `lat10` rerun and far below the geometry-biased line
- the clean `full_circle` family still reads as weak without geometry priors

## What Must Be Checked After Completion

1. Protocol result for `S27cov_fc_clean_lc0p25_t0p03_bs8192`
2. Regime failures, especially:
   - `0.8m / 100mA`
   - `0.8m / 300mA`
3. Whether `bs8192` beats the clean reference points already known:
  - clean baseline: `1/12`
  - clean lat10 rerun: `5/12`
4. Whether any gain appears without reintroducing geometry priors

## Decision

- `clean_bs8192` stayed weak; do not go back to the old `shape` workaround as
  the main claim; the next allowed step is only soft radial priors
- if `clean_bs8192` improves materially, then `full_circle` remains alive as a
  clean digital-twin hypothesis and deserves the next confirmation layer