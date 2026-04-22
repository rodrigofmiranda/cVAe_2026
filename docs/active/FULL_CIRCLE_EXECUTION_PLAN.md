# Full Circle Execution Record

Date: 2026-04-21

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
| 5A | `outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_a/exp_20260417_115142` | isolated clean confirmation for `clean_bs8192` | `S27cov_fc_clean_lc0p25_t0p03_bs8192`, `2/12` | the previously open clean candidate now has a direct protocol answer and remains weak |
| 5B | `outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_b/exp_20260417_115142` | rerun of `clean_lat10` under the same clean contract | `S27cov_fc_clean_lc0p25_t0p03_lat10`, `5/12` | modest recovery over the earlier `4/12`, but still well below the geometry-biased line |
| 6A | `outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256` | soft-radial localization without `geom3` or `disk_l2`: `clean_lat10`, `broad`, `local`, `tight` | `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0`, `6/12` | best clean-compatible result so far; soft radial priors recovered one pass above the clean best without returning to hard geometry |
| 6B | `outputs/full_circle/20260420_233254_soft_radial_block_b_100k/exp_20260420_233257` | orthogonal probes around `soft_rinf_local`: `base`, `bs8192`, `covsoft`, `tail98` | `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_covsoft_lc0p20_t0p035`, `0/12` | negative full-regime answer for `covsoft`; because shared-global protocol evaluated only the block champion, `bs8192` and `tail98` remained unresolved after this block |
| 7A | `outputs/full_circle/20260421_234722_soft_radial_resolve_bs8192_100k/exp_20260421_234723` | direct protocol rerun of `soft_rinf_local_bs8192` only | `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_bs8192`, `6/12` | optimization-only probe tied the base soft-radial candidate but did not surpass it |
| 7B | `outputs/full_circle/20260421_234722_soft_radial_resolve_tail98_100k/exp_20260421_234724` | direct protocol rerun of `soft_rinf_local_tail98` only | `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_tail98`, `1/12` | tighter tails were clearly negative on direct full-protocol evaluation |

## What Is Scientifically Established

1. The best Full Square finalists do not transfer cleanly to Full Circle.
2. Informing the model about disk-shaped support clearly helps on this dataset.
3. A geometry-light intermediate path now exists: `soft_rinf_local` reached
   `6/12` without `geom3` and without `disk_l2`, improving over the clean best
   (`5/12`) while staying below the hard geometry-biased ceiling.
4. The strongest numbers from this branch (`7/12` and `8/12`) are not neutral
   Full Circle baselines, because they still rely on geometry-specific support
   assumptions.
5. The clean restart remains the honest baseline for scientific comparison, and it
   is much weaker than the geometry-biased line.
6. The previously open `clean_bs8192` candidate is now resolved at `2/12`, so
  the clean line remains weak even after the missing confirmation run.
7. The `covsoft` follow-up is a negative Full Circle result under direct
   protocol evaluation: the block-B champion failed `0/12`, despite looking
   attractive under training-only ranking.

## What Remains Open

The main open question is no longer whether the unresolved soft-radial probes
work. Those answers are now in hand:

- `soft_rinf_local_bs8192` tied the base `soft_rinf_local` at `6/12`
- `soft_rinf_local_tail98` dropped to `1/12`

The remaining open question is narrower:

- is `soft_rinf_local` the final geometry-light stopping point for this branch,
  or does it justify one last very local retune around the same family?

## Scientific Reading

The main branch-level conclusion is still not that `disk_geom3_bs8192` solved
Full Circle. The stronger conclusion remains that the earlier gain depended
heavily on geometry priors inherited from the Full Square line.

The soft-radial screen sharpens that reading:

- `soft_rinf_local` shows that a lighter radial prior can recover part of the
  lost ground without going back to `geom3` or `disk_l2`
- `soft_rinf_local_bs8192` confirms that the gain is not simply an optimization
  artifact from batch-size retuning, because it only ties the base result
- `covsoft` shows that adding extra calibration pressure on top of that local
  radial prior can fail catastrophically at full-protocol level
- `tail98` confirms that tighter tails are not the right local continuation for
  this Full Circle line
- the direct reruns also confirm a methodological lesson: block-level training
  ranking is not enough when only one internal champion receives the full
  protocol evaluation

That makes the baseline logic more precise:

- if the goal is scientific honesty about Full Circle as a new acquisition
  geometry, the clean baseline must remain the reference point
- if the goal is a geometry-light compromise, `soft_rinf_local` is now the
  current best evidence in that class, with `soft_rinf_local_bs8192` as an
  equal but not better optimization variant
- if the goal is engineering performance only, `disk_geom3_bs8192` is the best
  practical result obtained here
- these are not the same claim and should not be mixed in future writeups
- after the direct confirmation run, `clean_bs8192` does not rescue the clean
  line; the clean family currently reads `1/12`, `2/12`, and `5/12`

## Operational Notes

New launchers created in this branch:

- `scripts/ops/train_full_circle_g2_shortlist.sh`
- `scripts/ops/train_full_circle_disk_bs8192_lat10.sh`
- `scripts/ops/train_full_circle_clean_bs8192_lat10.sh`
- `scripts/ops/train_full_circle_clean_bs8192_lat10_split.sh`
- `scripts/ops/train_full_circle_soft_radial_screen.sh`

Additional operational points:

- timestamp-prefixed `RUN_STAMP` output naming was restored for the new
  launchers
- the clean split launcher runs two tmux + Docker GPU stacks with a shared
  `RUN_STAMP`
- convenience timestamp aliases were created for the older geometry-biased
  split outputs, but the original split directories remain canonical
- for multi-candidate soft-radial screens, the shared-global protocol still
  selects one block champion for the full 12-regime evaluation; unresolved
  candidates must therefore be rerun individually with `--grid_tag`

## Recommended Handoff

This branch can now be treated as documented and parked.

If Full Circle is revisited later, the next step is no longer to resolve the
soft-radial shortlist. That is now closed.

The only justified continuation would be:

1. accept `soft_rinf_local` as the current geometry-light reference
2. decide explicitly whether a final, very local retune around
   `alpha=1.5, tau=0.80, wmax=3.0` is worth the cost
3. otherwise park this branch as scientifically documented

For the next active cycle, the recommendation is to return to MDN in a fresh
folder and branch rather than continuing in this worktree.
