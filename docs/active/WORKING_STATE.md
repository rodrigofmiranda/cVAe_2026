# Working State

This note is the handoff snapshot for the Full Circle worktree at the end of
the 2026-04-16 support-geometry audit.

## Current Worktree

- repo root: `/home/rodrigo/cVAe_2026_shape_fullcircle`
- active branch: `research/full-circle`
- intended next active line: move back to MDN in a new folder and a new branch,
  not in this worktree

## Canonical Full Circle Contract

Use these settings when reading or reproducing the quick screens from this
branch:

- protocol: `configs/protocol_full_circle_sel4curr.json`
- train mode: `--train_once_eval_all`
- caps: `100k` train per experiment, `20k` validation per experiment
- stats: `--stat_mode quick --stat_max_n 2000`
- dist cap: `--max_dist_samples 20000`
- output convention: timestamp-prefixed `RUN_STAMP` directories under
  `outputs/full_circle`

## What Was Actually Run

1. Imported Full Square finalists on Full Circle
   - output: `outputs/full_circle/e2_finalists_shortlist_100k/exp_20260416_014727`
   - champion: `S27cov_sciv1_lr0p00015`
   - outcome: `4/12`
   - reading: weak transfer; `tail98` should be treated as negative for this
     line

2. Full Circle-specific `G2` shortlist
   - output: `outputs/full_circle/g2_shortlist_100k/exp_20260416_120619`
   - champion: `S27cov_lc0p25_tail95_t0p03_disk_geom3`
   - outcome: `7/12`
   - reading: geometry-aligned disk handling improved the line materially

3. Geometry-biased follow-up around `disk_geom3`
   - split A output:
     `outputs/full_circle/disk_bs8192_lat10_100k_split_a/exp_20260416_165643`
   - split B output:
     `outputs/full_circle/disk_bs8192_lat10_100k_split_b/exp_20260416_165644`
   - best result: `S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192` -> `8/12`
   - negative probe: `disk_geom3_lat10` -> `3/12`
   - reading: best operational result of the branch, but still contaminated by
     `geom3` and `disk_l2`

4. Clean restart with no geometry priors
   - split A output:
     `outputs/full_circle/20260416_182317_clean_bs8192_lat10_100k_split_a/exp_20260416_182319`
   - split B output:
     `outputs/full_circle/20260416_182317_clean_bs8192_lat10_100k_split_b/exp_20260416_182319`
   - clean baseline: `S27cov_fc_clean_lc0p25_t0p03` -> `1/12`
   - clean lat10: `S27cov_fc_clean_lc0p25_t0p03_lat10` -> `4/12`
   - clean bs8192: trained more stably than the baseline, but was not sent to
     protocol because split A only evaluated the local training champion

## Main Scientific Reading

- Full Circle cannot inherit Full Square edge/corner assumptions by default
- `disk_geom3` and `disk_geom3_bs8192` are useful engineering evidence, not a
  neutral scientific baseline
- once `support_weight_mode=none`, `support_feature_mode=none`, and
  `support_filter_mode=none`, the line degrades sharply
- the clean restart therefore becomes the honest reference for this branch,
  even though its headline numbers are worse
- the single unresolved clean candidate is `clean_bs8192`; everything else from
  this iteration already has a clear reading

## Operational Notes

- the split launcher used for the clean restart is:
  - `scripts/ops/train_full_circle_clean_bs8192_lat10_split.sh`
- the single-stack launcher for the clean preset is:
  - `scripts/ops/train_full_circle_clean_bs8192_lat10.sh`
- timestamp-prefixed output naming was restored for the new launchers in this
  iteration
- the extra timestamp aliases created for the older geometry-biased split can
  be kept as convenience paths; the canonical experiment contents remain under
  the original split directories

## If This Branch Is Reopened

Do this first:

1. run `clean_bs8192` alone with its own `grid_tag`, so it gets a protocol
   answer independent of split-A champion selection
2. if the clean line still stays weak, only test soft radial priors next
3. do not go back to `cornerness_norm` or hard `disk_l2` if the goal is a clean
   Full Circle scientific baseline

## If Work Continues Elsewhere

This repository can now be treated as documented and parked. The next active
cycle should start from a new MDN-focused folder and branch.
