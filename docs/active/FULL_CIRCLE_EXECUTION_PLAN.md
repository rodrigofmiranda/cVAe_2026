# Full Circle Execution Record

Date: 2026-04-16

Purpose: document what was actually executed on `research/full-circle`, what was
learned, and what should be considered primary versus secondary evidence.

## Fixed Methodology For This Iteration

All quick screens in this branch used the same reduced protocol contract after
the initial correction:

- dataset root: `data/FULL_CIRCLE_2026`
- protocol: `configs/protocol_full_circle_sel4curr.json`
- `--train_once_eval_all`
- `--max_samples_per_exp 100000`
- `--max_val_samples_per_exp 20000`
- `--max_dist_samples 20000`
- `--stat_mode quick --stat_max_n 2000`
- output base under `outputs/full_circle`

Important note:

- an early Full Circle shortlist was mistakenly launched with full data; it was
  interrupted and replaced by the corrected `100k/20k` quick-screen run

## Run Chronology

| Step | Output | Scope | Best outcome | Reading |
|---|---|---|---|---|
| 1 | `outputs/full_circle/e2_finalists_shortlist_100k/exp_20260416_014727` | direct import of the three strongest Full Square finalists | `S27cov_sciv1_lr0p00015`, `4/12` | transfer was poor; `tail98` should be dropped for Full Circle |
| 2 | `outputs/full_circle/g2_shortlist_100k/exp_20260416_120619` | first Full Circle-specific shortlist (`control`, `lr0p00015`, `covsoft`, `disk`, `disk_geom3`) | `S27cov_lc0p25_tail95_t0p03_disk_geom3`, `7/12` | geometry-aligned disk handling recovered much of the line |
| 3A | `outputs/full_circle/disk_bs8192_lat10_100k_split_a/exp_20260416_165643` | geometry-biased follow-up on `disk_geom3` with `bs8192` | `S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192`, `8/12` | strongest operational result, but still geometry-biased |
| 3B | `outputs/full_circle/disk_bs8192_lat10_100k_split_b/exp_20260416_165644` | geometry-biased follow-up on `disk_geom3` with `lat10` | `S27cov_lc0p25_tail95_t0p03_disk_geom3_lat10`, `3/12` | negative probe |
| 4A | `outputs/full_circle/20260416_182317_clean_bs8192_lat10_100k_split_a/exp_20260416_182319` | clean restart with `support_weight_mode=none`, `support_feature_mode=none`, `support_filter_mode=none`; split A = baseline + `clean_bs8192` | `S27cov_fc_clean_lc0p25_t0p03`, `1/12` | removing geometry priors collapsed the line |
| 4B | `outputs/full_circle/20260416_182317_clean_bs8192_lat10_100k_split_b/exp_20260416_182319` | clean restart split B = `clean_lat10` | `S27cov_fc_clean_lc0p25_t0p03_lat10`, `4/12` | small recovery, still far below the geometry-biased line |

## What Is Scientifically Established

1. The best Full Square finalists do not transfer cleanly to Full Circle.
2. Informing the model about disk-shaped support clearly helps on this dataset.
3. The strongest numbers from this branch (`7/12` and `8/12`) are not neutral
   Full Circle baselines, because they still rely on geometry-specific support
   assumptions.
4. The clean restart is the honest baseline for scientific comparison, and it
   is much weaker than the geometry-biased line.

## What Remains Open

Only one candidate from this iteration still lacks a direct protocol answer:

- `S27cov_fc_clean_lc0p25_t0p03_bs8192`

Why it is still open:

- split A used training-time `score_v2` to pick one local champion for protocol
- the champion in split A was the clean baseline, not `clean_bs8192`
- therefore `clean_bs8192` trained successfully and looked more stable than the
  baseline, but it was never given its own protocol evaluation in this branch

## Scientific Reading

The main branch-level conclusion is not that `disk_geom3_bs8192` solved Full
Circle. The stronger conclusion is that the earlier gain depended heavily on
geometry priors inherited from the Full Square line.

That makes the clean restart decisive:

- if the goal is scientific honesty about Full Circle as a new acquisition
  geometry, the clean baseline must remain the reference point
- if the goal is engineering performance only, `disk_geom3_bs8192` is the best
  practical result obtained here
- these are not the same claim and should not be mixed in future writeups

## Operational Notes

New launchers created in this branch:

- `scripts/ops/train_full_circle_g2_shortlist.sh`
- `scripts/ops/train_full_circle_disk_bs8192_lat10.sh`
- `scripts/ops/train_full_circle_clean_bs8192_lat10.sh`
- `scripts/ops/train_full_circle_clean_bs8192_lat10_split.sh`

Additional operational points:

- timestamp-prefixed `RUN_STAMP` output naming was restored for the new
  launchers
- the clean split launcher runs two tmux + Docker GPU stacks with a shared
  `RUN_STAMP`
- convenience timestamp aliases were created for the older geometry-biased
  split outputs, but the original split directories remain canonical

## Recommended Handoff

This branch can now be treated as documented and parked.

If Full Circle is revisited later, the next experiment should be:

1. a single-candidate confirmation run for `clean_bs8192`

If that still underperforms, the next design principle should be:

1. test only soft radial priors
2. avoid `cornerness_norm`
3. avoid hard `disk_l2` filtering if the goal is a clean Full Circle baseline

For the next active cycle, the recommendation is to return to MDN in a fresh
folder and branch rather than continuing in this worktree.