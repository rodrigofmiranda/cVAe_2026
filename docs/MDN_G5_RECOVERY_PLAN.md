# MDN G5 Recovery Plan

This document is the active plan for the branch:

- `feat/mdn-g5-recovery`

## Why We Returned To MDN

The current `conditional-flow-decoder` line was tested enough to make a
scientific decision on the **current implementation family**:

- `outputs/exp_20260326_034522`
  - plain flow micro proof
  - `0/12`
- `outputs/exp_20260326_035723`
  - stabilized flow quick grid
  - best candidate still `0/12`

Conclusion:

- the branch proved the flow plumbing works
- the current `sinh-arcsinh` flow per axis is **not** scientifically
  competitive
- we do **not** need more GPU budget on that exact flow family

This does **not** invalidate the general idea of a richer decoder family.
It only discards the current implemented flow line.

## Strongest Current MDN Reference

Best MDN result so far:

- `outputs/exp_20260325_230938`
- winner:
  - `S14seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512`
- result:
  - `9/12`
  - `gate_g6_pass = 12`
  - all `1.0 m` pass
  - all `1.5 m` pass
  - `0.8m / 700mA` passes

Remaining failures:

- `dist_0p8m__curr_100mA`
- `dist_0p8m__curr_300mA`
- `dist_0p8m__curr_500mA`

Key reading:

- the bottleneck is no longer broad `G6`
- the dominant remaining gap is `G5`
- this points more to marginal-shape recovery than to global distributional
  failure

## Active Hypothesis

The next MDN step should **not** be another broad sweep over:

- more mixture counts
- lower `beta`
- stronger generic `MMD`

Those families were already explored and either plateaued or regressed.

The active hypothesis is:

- keep the best stable MDN anchor fixed
- improve the hard `0.8 m` regimes through **regime-aware emphasis**
- do this without reopening the broader `1.0 m / 1.5 m` regressions

## Immediate Technical Direction

Anchor configuration:

- `seq_bigru_residual`
- `mdn3`
- `W7_h64_lat4`
- `beta=0.002`
- `free_bits=0.10`
- `lr=2e-4`
- `lambda_mmd=0.25`
- `lambda_axis=0.01`
- `lambda_psd=0.0`

Next intervention to implement:

1. add regime-aware sample weighting in training
2. overweight:
   - `0.8m / 100mA`
   - `0.8m / 300mA`
   - optionally `0.8m / 500mA`
3. leave the rest at weight `1.0`
4. compare:
   - unweighted anchor
   - weighted anchor
   - one stronger weighted hedge

Implementation status:

- done
- current preset:
  - `seq_mdn_regime_weight_quick`
- implementation detail:
  - weighting is applied as fixed-size weighted resampling of the training
    windows, so the epoch length stays constant while the hard `0.8 m`
    regimes are emphasized

## Recommended Execution Order

### Phase 1

- implement regime weighting
- keep the model family unchanged
- add a compact quick preset around the S14 anchor

### Phase 2

- run a quick 12-regime compare with sample caps
- acceptance gate:
  - do not regress `1.0 m` / `1.5 m`
  - improve at least one of the remaining `0.8 m` `G5` failures

### Phase 3

- only if the weighted quick run is promising:
  - run a full single-candidate compare

## What Not To Do Next

Do not spend the next cycle on:

- more `sinh-arcsinh` flow runs
- full compare of the failed flow line
- broad MDN sweeps that reopen `beta × lambda_mmd × mdn_components`
- PSD loss

## Immediate Next Step

Implement regime-aware weighting on top of the S14 MDN anchor.
