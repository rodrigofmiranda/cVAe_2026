# Full Circle Next Test Direction

Date: 2026-04-17

Purpose: document the next scientifically honest `full_circle` test direction
for this project, based on the original shaping hypothesis from
`/home/rodrigo/cVAe_2026_shape` and the execution evidence from
`/home/rodrigo/cVAe_2026_shape_fullcircle`.

## Why This Note Exists

There is now a methodological turn that must be kept explicit in future runs:

- the old `shape` line used support weights, support features, and support
  filters as a provisional way to emphasize different regions of the
  `full_square` dataset
- that was useful when the project did **not** yet have a true `full_circle`
  acquisition to test the support-geometry hypothesis directly
- now that a `FULL_CIRCLE_2026` dataset exists and was already screened in the
  separate `research/full-circle` worktree, those `shape` interventions must be
  treated as a proxy / workaround, not as the clean validation target

The next `full_circle` tests must therefore validate the digital twin using
real disk-shaped acquisition data, not by reintroducing geometry priors as a
substitute for missing data.

## Source Of The Hypothesis

The original support-geometry hypothesis came from:

- papers:
  - `Askari and Lampe (2025)` on shaping under real channel nonlinearity,
    memory, moments, and sequence effects
  - `Shu and Zhang (2020)` on practical shaping under hardware constraints
- working branch:
  - `/home/rodrigo/cVAe_2026_shape`
  - branch `feat/probabilistic-shaping-nonlinearity`

The practical interpretation for this repo was:

- `full_square` is a broad channel-probing support, but it excites hard corner
  regions
- `full_circle` is the clean test of whether a smoother support geometry better
  matches the physical VLC channel
- a gain from `full_circle` only matters if it survives fair bench,
  digital-twin, and downstream checks

## What The `shape` Line Was Actually Doing

The `shape` line explored support-aware interventions such as:

- `support_weight_mode`
- `support_feature_mode`
- `support_filter_mode`
- geometry-specific settings such as `disk_geom3`
  and hard region selection like `disk_l2`

These experiments were scientifically useful, but they must be read correctly:

- they were attempts to stress, weight, or bias different areas of the
  existing `full_square` support
- they were **not** the same thing as validating a real `full_circle`
  acquisition
- once real disk-shaped data exists, the project should stop treating those
  weighted / filtered support tricks as the main validation path

Short version:

- `shape` = workaround / proxy investigation
- `full_circle` = proper test of the support-geometry hypothesis

## What The Separate Full Circle Worktree Already Established

The real `full_circle` screening happened in:

- `/home/rodrigo/cVAe_2026_shape_fullcircle`
- branch `research/full-circle`

The branch-level reading was:

- imported Full Square finalists transferred poorly to Full Circle
- geometry-informed runs like `disk_geom3` and `disk_geom3_bs8192` improved the
  headline score, but remained geometry-biased evidence
- the clean restart with:
  - `support_weight_mode=none`
  - `support_feature_mode=none`
  - `support_filter_mode=none`
  became the honest baseline for scientific comparison
- that clean line was much weaker than the geometry-biased line

This means the next honest question is no longer:

- "which weighting/filtering trick helps the model on `full_square`?"

It is now:

- "does the real `full_circle` dataset support a stronger digital twin without
  relying on geometry priors that effectively reintroduce a workaround?"

## Hypotheses That Must Be Tested Now

Primary hypothesis:

- `H1`: for fixed average transmit power, a real `full_circle` acquisition
  improves both physical transmission behavior and digital-twin fidelity
  because it reduces corner stress, peak-related distortion, and angle-specific
  boundary effects present in `full_square`

Null hypothesis:

- `H0`: changing the acquisition support from square to circle does not
  materially improve the physical channel or the learned twin; the dominant
  bottleneck is still model family, conditioning, or loss design

Mechanistic hypotheses:

- `H2`: `full_circle` reduces asymmetry, compression, and harmonic distortion
  on the physical bench
- `H3`: `full_circle` reduces center-vs-edge mismatch and makes the conditional
  channel more regular as a function of radius
- `H4`: if `full_circle` improves fit but weakens downstream robustness, then
  it is hiding relevant operating regions instead of solving the channel
  modeling problem

## Decision Rule

Do **not** promote `full_circle` just because a support-aware configuration
looks easier for the model.

Promote `full_circle` only if it improves all three layers together:

1. bench-level physical diagnostics
2. digital-twin fidelity
3. downstream usefulness under fair constraints

If the gain disappears once support priors are removed, then the result should
be interpreted as a geometry-prior effect, not as validation of the new
acquisition itself.

## Next Test Sequence

The next pass should be read as a validation ladder, not as another broad grid.

### Step 1: Interpret The Completed Clean Confirmation

The missing clean confirmation run is now closed:

- `S27cov_fc_clean_lc0p25_t0p03_bs8192` -> `2/12`
- `S27cov_fc_clean_lc0p25_t0p03_lat10` rerun -> `5/12`

Reading:

- the clean line no longer has an unresolved flagship candidate
- the clean family currently reads `1/12`, `2/12`, and `5/12`
- this is still far below the geometry-biased `7/12` and `8/12` results

### Step 2: Keep The Clean Baseline As Reference

Compare any future result against the clean Full Circle reference, not against
the geometry-biased champions.

Reference points from the parked worktree:

- `S27cov_fc_clean_lc0p25_t0p03` -> `1/12`
- `S27cov_fc_clean_lc0p25_t0p03_bs8192` -> `2/12`
- `S27cov_fc_clean_lc0p25_t0p03_lat10` -> `5/12`

### Step 3: Only If Needed, Reintroduce Soft Radial Priors

If the clean candidate remains weak, the next scientific move is **not** to go
back to the old `shape` workaround wholesale.

The next allowed class of intervention should be:

- soft radial priors only

Avoid reopening as baseline:

- `cornerness_norm`
- hard `disk_l2` filtering
- geometry-weight tricks as the main scientific claim

## Metrics That Must Be Reported

Bench metrics:

- asymmetry ratio
- crest factor / effective peak stress
- baseband reconstruction correlation
- baseband reconstruction NMSE
- occupied bandwidth
- visible harmonic content near `2f`

Digital-twin metrics:

- residual mean and covariance
- residual skewness and kurtosis
- Jarque-Bera or equivalent shape-distance metric
- center-vs-edge mismatch
- amplitude-conditioned and radius-conditioned diagnostics
- regime-specific outcomes, especially `0.8m / 100mA` and `0.8m / 300mA`

Decision metrics:

- full protocol result
- stability across seeds when feasible
- whether downstream use gets stronger or only the fit gets easier

## Operational Grounding

The parked `research/full-circle` worktree already contains the relevant
launchers:

- `scripts/ops/train_full_circle_clean_bs8192_lat10.sh`
- `scripts/ops/train_full_circle_clean_bs8192_lat10_split.sh`
- `scripts/ops/train_full_circle_g2_shortlist.sh`
- `scripts/ops/train_full_circle_disk_bs8192_lat10.sh`

If this line is resumed, start from the clean candidate, not from the
geometry-biased shortlist.

To follow logs for the next clean run, use the exact pattern below after the
new `RUN_STAMP` is known:

```bash
tail -f /home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/<RUN_STAMP>_clean_bs8192_lat10_100k_split_a/logs/*.log
```

For the paired split-B logs:

```bash
tail -f /home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/<RUN_STAMP>_clean_bs8192_lat10_100k_split_b/logs/*.log
```

## Practical Reading For The Current Worktree

For this current worktree, the point of documenting `full_circle` is not to
claim that the project already switched away from MDN.

The point is to make the next structural test explicit:

- the old `shape` interventions were a bridge while real disk-shaped data did
  not exist
- now the real test exists
- therefore the next honest validation step is to complete the clean
  `full_circle` line and judge it by the three-layer rule above

Operational checklist for the next run:

- `docs/active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md`