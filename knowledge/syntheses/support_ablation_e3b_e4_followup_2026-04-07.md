# Support Ablation E3b-E4 Follow-up

Date: 2026-04-07

## Scope

This note continues the `E0-E3` memo and records what happened in the local
retune and the next decision stages:

- `E3b`: short local retune around the `E3` winner
- `E3c`: two-candidate decision run after `E3b`
- `E4`: adapted `disk_l2` test on top of the `E3c` winner

The focus stays the same:

- edge / border behavior
- tail coverage
- digital-twin fidelity

## Runs Used

- `E3b`: `outputs/support_ablation/e3b/exp_20260407_012715`
- `E3c`: `outputs/support_ablation/e3c/exp_20260407_113203`
- `E4`: `outputs/support_ablation/e4_covsoft_disk/exp_20260407_151024`

Tables used:

- `tables/protocol_leaderboard.csv`
- `tables/summary_by_regime.csv`
- `train/tables/gridsearch_results.csv`

## Scoreboard

| Stage | Winning candidate | Hypothesis | Passes | Pass pattern by distance |
| --- | --- | --- | ---: | --- |
| `E3b` | `S27...localcap...` | retune `geom3 + edge` locally | `5/12` | `0.8m=0`, `1.0m=1`, `1.5m=4` |
| `E3c` | `S27...covsoft...` | choose between `bridge` and `covsoft` | `6/12` | `0.8m=0`, `1.0m=2`, `1.5m=4` |
| `E4` | `S27...covsoft... + disk_l2` | restrict support with `disk_l2` | `5/12` | `0.8m=0`, `1.0m=3`, `1.5m=2` |

## Fidelity Snapshot

Smaller is better for the error-like metrics below.

| Stage | `rel_evm` | `rel_snr` | `mean_rel_sigma` | `cov_rel_var` | `psd_l2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `E3b` | `0.0403` | `0.0282` | `0.0354` | `0.1286` | `0.0641` |
| `E3c` | `0.0375` | `0.0282` | `0.0065` | `0.1188` | `0.0607` |
| `E4` | `0.0340` | `0.0364` | `0.0289` | `0.1431` | `0.0602` |

Additional statistical read:

| Stage | `gate_g5_pass` | `gate_g6_pass` | `mean_stat_mmd_qval` | `mean_stat_energy_qval` |
| --- | ---: | ---: | ---: | ---: |
| `E3b` | `5` | `8` | `0.1741` | `0.1482` |
| `E3c` | `6` | `9` | `0.2332` | `0.2340` |
| `E4` | `8` | `6` | `0.1303` | `0.1281` |

## Border And Coverage

Coverage deltas are negative when the model under-populates the outer support.
Less negative is better.

| Stage | `avg_delta_coverage_80` | `avg_delta_coverage_95` | Border read |
| --- | ---: | ---: | --- |
| `E3b` | `-0.0985` | `-0.1457` | better than `E2`, softer than `E3` |
| `E3c` | `-0.1061` | `-0.1529` | not the best average coverage, but the best overall balance |
| `E4` | `-0.1016` | `-0.1474` | `disk_l2` shifts behavior, but does not improve the real bottleneck |

Important nuance:

- `E3c` did **not** win by pushing the most mass to the edge on average
- it won by recovering fidelity while keeping enough edge-aware pressure to
  unlock more protocol passes

## What Each Step Did

### `E3b`: local retune around `E3`

- kept the edge-aware direction alive
- improved fidelity over `E3`
- still stayed at `5/12`
- main value: it narrowed the search to the most promising local settings

This stage showed that the `E3` idea was correct, but still too aggressive in
its first form.

### `E3c`: `bridge` vs `covsoft`

- first stage to reach `6/12`
- recovered the `1m / 500mA` pass while keeping `1.5m / 500mA`
- improved `G6` from `8` to `9`
- achieved the best balance so far across:
  - EVM / SNR
  - sigma calibration
  - covariance fidelity
  - statistical acceptance

This is the current best candidate of the `shape` line:

- `S27cov_geom3_edge_rt_covsoft_a1p5_tau0p82_tc0p45_wmax2p5_lc0p20`

### `E4`: adapted `disk_l2` test on top of `E3c`

- improved some central-support behavior at `1.0m`
- lost robustness at `1.5m`
- raised `G5`, but dropped `G6`
- degraded sigma calibration and statistical acceptance

The result is a useful negative finding:

- `disk_l2` is not the right next main direction for this line

## Current Bottleneck

After `E3c`, the remaining failure pattern is much clearer:

- `0.8m` remains `0/4`
- `0.8m / 700mA` is close and fails mainly on `G5`
- `1.0m / 100mA` and `1.0m / 300mA` are also mostly `G5`-limited

So the line is no longer blocked by a broad global mismatch.
It is now much more specifically blocked by:

- tail-support fidelity
- low-distance edge behavior

## Operational Conclusion

- Best current `shape` candidate: `E3c covsoft`
- Best rejected follow-up: adapted `E4 disk_l2`

In practical terms:

- keep `E3c covsoft` as the current champion
- do not promote `disk_l2`
- the next follow-up should attack `G5` at `0.8m` and `1.0m` without shrinking
  the support the way `disk_l2` did
