# Support Ablation E0-E3 Comparison

Date: 2026-04-07

## Scope

This memo compares the first four support-aware shaping stages:

- `E0`: anchors / baseline screen
- `E1`: geometry features only (`geom3`)
- `E2`: edge-aware sample weighting only (`edge_weight`)
- `E3`: `geom3 + edge_weight`

The goal is to separate what each hypothesis changed in:

- border / edge behavior
- tail coverage
- overall digital-twin fidelity

## Runs Used

- `E0`: `outputs/support_ablation/e0_recovery_memfix/exp_20260406_193640`
- `E1`: `outputs/support_ablation/e1_recovery_lambdafix/exp_20260406_195854`
- `E2`: `outputs/support_ablation/e2/exp_20260406_203946`
- `E3`: `outputs/support_ablation/e3/exp_20260406_223246`

Tables used:

- `tables/protocol_leaderboard.csv`
- `tables/summary_by_regime.csv`
- `tables/residual_signature_by_regime.csv`

## Important Reading Note

`E1` is not a perfectly clean S27-only ablation, because the stage winner moved to
the point-wise `D3 ... geom3` candidate. So `E1` should be read mainly as a test
of the hypothesis "geometry alone helps", not as a strict continuation of the
same `S27` line.

## Scoreboard

| Stage | Winning candidate | Hypothesis | Passes | Pass pattern by distance |
| --- | --- | --- | ---: | --- |
| `E0` | `S27cov_lc0p25_tail95_t0p03` | anchor / no support-aware change | `3/12` | `0.8m=0`, `1.0m=0`, `1.5m=3` |
| `E1` | `D3...geom3` | add geometry features only | `3/12` | `0.8m=0`, `1.0m=0`, `1.5m=3` |
| `E2` | `S27..._edge` | add edge-aware weighting only | `5/12` | `0.8m=0`, `1.0m=2`, `1.5m=3` |
| `E3` | `S27..._geom3_edge` | combine geometry + edge weighting | `5/12` | `0.8m=0`, `1.0m=1`, `1.5m=4` |

## Border And Coverage

Coverage deltas are negative when the model under-populates the outer support.
Less negative is better.

| Stage | `avg_delta_coverage_80` | `avg_delta_coverage_95` | Border read |
| --- | ---: | ---: | --- |
| `E0` | `-0.1019` | `-0.1487` | under-populated edge / tail |
| `E1` | `-0.1383` | `-0.1817` | geometry alone made coverage worse |
| `E2` | `-0.1053` | `-0.1521` | almost same tail deficit as `E0`, but cleaner edge matching overall |
| `E3` | `-0.0926` | `-0.1408` | best tail coverage among `E0-E3` |

Interpretation:

- `E0` established the main failure mode: too little mass near the outer support.
- `E1` improved some radial descriptors, but it did **not** solve the edge-gap; it
  actually worsened mean tail coverage.
- `E2` did not create the best raw tail-coverage averages, but it was the first
  stage to convert that support-aware pressure into more actual regime passes.
- `E3` was the most aggressive push toward the border: best average coverage, more
  `1.5m` passes, and better final statistical gate count (`G6`), but with clear
  fidelity cost.

## Fidelity

Smaller is better for the error-like metrics below.

| Stage | `rel_evm` | `rel_snr` | `mean_rel_sigma` | `cov_rel_var` | `psd_l2` | Fidelity read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `E0` | `0.0462` | `0.0311` | `0.0842` | `0.1522` | `0.0676` | decent anchor, but poor sigma calibration |
| `E1` | `0.0440` | `0.0360` | `0.0455` | `0.2001` | `0.0645` | slightly better radial/global shape, worse covariance fidelity |
| `E2` | `0.0405` | `0.0305` | `0.0124` | `0.1357` | `0.0633` | best overall twin fidelity |
| `E3` | `0.0544` | `0.0423` | `0.0078` | `0.1721` | `0.0701` | better statistical acceptance, worse physical/global fidelity |

Additional statistical read:

| Stage | `gate_g3_pass` | `gate_g5_pass` | `gate_g6_pass` | `mean_stat_mmd_qval` | `mean_stat_energy_qval` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `E0` | `8` | `5` | `5` | `0.1761` | `0.1574` |
| `E1` | `8` | `4` | `4` | `0.1124` | `0.1012` |
| `E2` | `10` | `7` | `5` | `0.1731` | `0.1472` |
| `E3` | `9` | `6` | `7` | `0.2104` | `0.2098` |

Interpretation:

- `E2` gave the best overall digital twin: best balance of EVM/SNR, covariance,
  PSD, and gate consistency.
- `E3` improved final statistical acceptance (`G6`) and pushed more mass toward
  the border, but it paid for that with worse EVM/SNR/covariance/PSD.
- `E1` is the clearest negative result: geometry by itself did not carry the edge
  behavior we needed.

## What Each Hypothesis Did

### `E0`: anchor only

- Established the real failure mode.
- Good enough to model the core distribution at `1.5m`.
- Still failed to reproduce the outer support at `0.8m` and `1.0m`.

### `E1`: geometry features only

- Helped some radial summaries.
- Did **not** translate into better edge/tail coverage.
- Did **not** improve the pass count.
- Therefore: geometry alone was not the missing ingredient.

### `E2`: edge weighting only

- First hypothesis that materially changed the outcome.
- Increased passes from `3/12` to `5/12`.
- Added two wins at `1.0m` (`500mA`, `700mA`).
- Preserved the best global twin fidelity among all four stages.
- Therefore: explicit training pressure on edge/corner samples is a real lever.

### `E3`: geometry + edge weighting

- Combined support geometry with edge pressure.
- Kept the total pass count at `5/12`, but shifted the profile:
  - lost one `1.0m` pass
  - gained one `1.5m` pass
- Produced the best average tail coverage and the best `G6` count.
- But degraded the global physical twin.
- Therefore: this combination is promising for border-seeking behavior, but too
  aggressive in its first form.

## Operational Conclusion

- Best **reference twin** after `E0-E3`: `E2`
- Best **border-seeking direction** after `E0-E3`: `E3`
- Best **rejected hypothesis**: `E1` as "geometry alone"

In plain terms:

- if the priority is fidelity to the current channel twin, keep `E2`
- if the priority is pushing support / edge behavior, `E3` is the right branch to
  refine, not `E1`

That is exactly why the next steps were local refinements around `E3`, not a
return to geometry-only variants.
