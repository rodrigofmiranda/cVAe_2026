# Working State

This note is the handoff snapshot for the Full Circle worktree after the
2026-04-20/2026-04-21 soft-radial follow-up.

## Current Worktree

- repo root: `/home/rodrigo/cVAe_2026_shape_fullcircle`
- active branch: `research/full-circle`
- intended next active line: decide whether this worktree is parked or receives
  one last very local retune around the current soft-radial reference

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

5. Clean confirmations
  - split A output:
    `outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_a/exp_20260417_115142`
  - split B output:
    `outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_b/exp_20260417_115142`
  - clean bs8192: `S27cov_fc_clean_lc0p25_t0p03_bs8192` -> `2/12`
  - clean lat10: `S27cov_fc_clean_lc0p25_t0p03_lat10` -> `5/12`
  - reading: the clean line improved slightly, but remained clearly below the
    geometry-biased line

6. Soft-radial screen without `geom3` or `disk_l2`
  - block A output:
    `outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256`
  - block B output:
    `outputs/full_circle/20260420_233254_soft_radial_block_b_100k/exp_20260420_233257`
  - best localized candidate:
    `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0` -> `6/12`
  - block-B protocol champion:
    `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_covsoft_lc0p20_t0p035`
    -> `0/12`
  - reading: a light radial prior recovered one pass above the clean best,
    but `covsoft` failed badly at full-protocol level

7. Pending direct reruns to resolve unresolved soft-radial candidates
  - `bs8192` output:
    `outputs/full_circle/20260421_234722_soft_radial_resolve_bs8192_100k/exp_20260421_234723`
  - `bs8192` result:
    `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_bs8192` -> `6/12`
  - `tail98` output:
    `outputs/full_circle/20260421_234722_soft_radial_resolve_tail98_100k/exp_20260421_234724`
  - `tail98` result:
    `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_tail98` -> `1/12`
  - reading: `bs8192` tied the base geometry-light candidate, while `tail98`
    was clearly negative

## Main Scientific Reading

- Full Circle cannot inherit Full Square edge/corner assumptions by default
- `disk_geom3` and `disk_geom3_bs8192` are useful engineering evidence, not a
  neutral scientific baseline
- once `support_weight_mode=none`, `support_feature_mode=none`, and
  `support_filter_mode=none`, the line degrades sharply
- `soft_rinf_local` now provides a geometry-light middle ground:
  better than the clean best (`6/12` vs `5/12`), but still below the hard
  geometry-biased ceiling (`8/12`)
- `soft_rinf_local_bs8192` confirms that batch-size retuning does not improve
  beyond the base geometry-light result
- `soft_rinf_local_tail98` is negative and should not be promoted
- the block-B `covsoft` champion is a cautionary result: training-only ranking
  can promote a candidate that collapses under full-protocol validation
- the clean restart therefore becomes the honest reference for this branch,
  even though its headline numbers are worse
- the geometry-light shortlist is now effectively resolved

## Operational Notes

- the split launcher used for the clean restart is:
  - `scripts/ops/train_full_circle_clean_bs8192_lat10_split.sh`
- the single-stack launcher for the clean preset is:
  - `scripts/ops/train_full_circle_clean_bs8192_lat10.sh`
- the soft-radial screen launcher is:
  - `scripts/ops/train_full_circle_soft_radial_screen.sh`
- timestamp-prefixed output naming was restored for the new launchers in this
  iteration
- the extra timestamp aliases created for the older geometry-biased split can
  be kept as convenience paths; the canonical experiment contents remain under
  the original split directories
- when a soft-radial block contains multiple candidates, the shared-global
  protocol still evaluates only the internally selected champion across all 12
  regimes; unresolved block members must be rerun individually with
  `--grid_tag`

## If This Branch Is Reopened

Do this first:

1. treat the geometry-light line as localized at
   `alpha=1.5, tau=0.80, wmax=3.0`
2. if the branch is reopened, only allow a very local retune around that point
3. do not go back to `cornerness_norm` or hard `disk_l2` if the goal remains a
   clean or geometry-light Full Circle scientific baseline

## If Work Continues Elsewhere

This repository can now be treated as documented and parked. The next active
cycle should start from a new MDN-focused folder and branch.
