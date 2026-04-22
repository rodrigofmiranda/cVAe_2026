# Full Circle Soft-Radial Master Table (2026-04-22)

This note consolidates the completed `soft-radial` screening on
`research/full-circle`.

Its role is to convert the execution log into a stable scientific table that
can be cited by the thesis-layer documents without forcing them to parse the
entire operational chronology again.

## Sources

- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)
- [block A training diagnostics](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256/train/tables/grid_training_diagnostics.csv)
- [block B training diagnostics](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_b_100k/exp_20260420_233257/train/tables/grid_training_diagnostics.csv)
- [block A protocol leaderboard](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256/tables/protocol_leaderboard.csv)
- [block B protocol leaderboard](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_b_100k/exp_20260420_233257/tables/protocol_leaderboard.csv)
- [direct rerun `soft_rinf_local_bs8192`](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260421_234722_soft_radial_resolve_bs8192_100k/exp_20260421_234723/tables/protocol_leaderboard.csv)
- [direct rerun `soft_rinf_local_tail98`](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260421_234722_soft_radial_resolve_tail98_100k/exp_20260421_234724/tables/protocol_leaderboard.csv)

## Reading Rule

- `Promote`: keep as the main representative of the family
- `Retain`: keep as a secondary or tied representative, but not as a new main
  line
- `Discard for now`: negative evidence is already strong enough that the
  variant should not be promoted without genuinely new evidence
- `Baseline`: use as the honest scientific reference for the branch
- `Ceiling`: use as the operational upper bound, not as a neutral baseline

## Master Table

| Layer | Scientific question | Candidates | Status | Champion / current best | Primary result | Provisional decision |
| --- | --- | --- | --- | --- | --- | --- |
| `clean baseline` | What is the honest Full Circle baseline without explicit geometry priors? | `clean`, `clean_bs8192`, `clean_lat10` | resolved clean | `S27cov_fc_clean_lc0p25_t0p03_lat10` | `5/12` pass | `Baseline`: keep as the scientific reference for neutral Full Circle |
| `geometry-biased ceiling` | How far can explicit disk-aware support handling push the branch operationally? | `disk`, `disk_geom3`, `disk_geom3_bs8192`, `disk_geom3_lat10` | resolved | `S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192` | `8/12` pass | `Ceiling`: keep as the best operational result, but not as the scientific baseline |
| `soft-radial A` | Can a lighter radial prior recover part of the gap without returning to `geom3` or `disk_l2`? | `clean_lat10`, `broad`, `local`, `tight` | completed clean | `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0` | `6/12` pass | `Promote` as the geometry-light reference |
| `soft-radial B` | Do orthogonal retunes around `soft_rinf_local` improve the result? | `base`, `bs8192`, `covsoft`, `tail98` | fully resolved after direct reruns | `soft_rinf_local` and `soft_rinf_local_bs8192` | `6/12` pass tie; `covsoft=0/12`; `tail98=1/12` | `Retain` `bs8192` only as an equal optimization variant; `Discard for now` `covsoft` and `tail98` |

## Block A Summary

Artifacts:

- training diagnostics:
  [grid_training_diagnostics.csv](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256/train/tables/grid_training_diagnostics.csv)
- full protocol:
  [protocol_leaderboard.csv](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256/tables/protocol_leaderboard.csv)

Training-side ranking:

1. `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0`
2. `S27cov_fc_soft_rinf_broad_lat10_a1p20_tau0p75_wmax2p5`
3. `S27cov_fc_soft_rinf_tight_lat10_a1p80_tau0p85_wmax3p0`
4. `S27cov_fc_clean_lc0p25_t0p03_lat10`

Winner:

- `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0`

Primary result:

- `6/12` pass
- `G1=7/12`
- `G2=6/12`
- `G3=8/12`
- `G4=11/12`
- `G5=8/12`
- `G6=8/12`

Interpretation:

- The block answered the key question positively: a light radial prior can
  recover one additional accepted regime relative to the clean best
  (`5/12 -> 6/12`) without going back to the stronger `geom3`/`disk_l2`
  assumptions.
- This result therefore deserves promotion as a geometry-light reference, not
  merely as an exploratory curiosity.

## Block B Summary

Artifacts:

- training diagnostics:
  [grid_training_diagnostics.csv](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_b_100k/exp_20260420_233257/train/tables/grid_training_diagnostics.csv)
- initial full protocol:
  [protocol_leaderboard.csv](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_b_100k/exp_20260420_233257/tables/protocol_leaderboard.csv)

Training-side ranking:

1. `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_covsoft_lc0p20_t0p035`
2. `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_tail98`
3. `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0`
4. `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_bs8192`

Initial block-level outcome:

- The shared-global protocol promoted only the training champion inside the
  block.
- That champion was `covsoft`, and its direct full-protocol answer was `0/12`.

Interpretation:

- The block delivered a methodological warning: block-level training ranking is
  not reliable enough to settle the family when only one internal champion is
  fully evaluated.
- Because of that, `bs8192` and `tail98` remained unresolved until direct
  reruns were launched with explicit `--grid_tag`.

## Direct Resolution Of The Open Probes

Artifacts:

- [direct `bs8192` rerun leaderboard](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260421_234722_soft_radial_resolve_bs8192_100k/exp_20260421_234723/tables/protocol_leaderboard.csv)
- [direct `tail98` rerun leaderboard](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260421_234722_soft_radial_resolve_tail98_100k/exp_20260421_234724/tables/protocol_leaderboard.csv)

Resolved answers:

- `soft_rinf_local_bs8192` tied the base `soft_rinf_local` at `6/12`
- `soft_rinf_local_tail98` collapsed to `1/12`

Interpretation:

- `bs8192` is not a new family winner. It is an optimization-only variant that
  confirms the same geometry-light plateau reached by the base local radial
  setting.
- `tail98` is now a clearly negative continuation for this branch.
- Combined with the `covsoft=0/12` result, the family is no longer open in a
  meaningful sense: the geometry-light shortlist has been scientifically
  resolved.

## Cross-Block Conclusion

The current evidence supports the following ordering for Full Circle:

1. Operational ceiling:
   `S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192` at `8/12`
2. Geometry-light reference:
   `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0` at `6/12`
3. Tied optimization variant:
   `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_bs8192` at `6/12`
4. Honest clean baseline:
   `S27cov_fc_clean_lc0p25_t0p03_lat10` at `5/12`
5. Negative local continuations:
   `tail98=1/12`, `covsoft=0/12`

This reading sharpens the branch-level claim:

- the best operational Full Circle number still depends on explicit geometry
  priors
- the clean line remains weaker than that ceiling
- but a lighter radial prior can recover part of the lost ground without
  returning to the hardest geometry assumptions

## Thesis Framing

For thesis writing, this completed screen now supports five defensible
statements:

1. The clean Full Circle line is scientifically honest, but still weak.
2. Hard geometry bias improves performance, but should not be mislabeled as the
   neutral Full Circle baseline.
3. A softer radial prior is a real intermediate scientific result, not just an
   operational accident.
4. The best geometry-light representative is `soft_rinf_local`; `bs8192` only
   ties it and therefore does not justify replacing it.
5. `covsoft` and `tail98` can be treated as negative evidence for the current
   branch and moved out of the active shortlist.
