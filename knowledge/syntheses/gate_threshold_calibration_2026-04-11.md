# Gate Threshold Calibration

Date: 2026-04-11

## Purpose

This note records the first empirical calibration pass for the digital-twin
gate thresholds.

It is meant to answer:

- which current thresholds are clearly too loose relative to historically strong twins
- which thresholds are already close to the empirical envelope
- which quantities should remain conventional statistical decision levels
  rather than engineering thresholds

## Calibration Artifacts

Generated artifacts:

- table: [gate_threshold_calibration_table.csv](/home/rodrigo/cVAe_2026_shape/outputs/analysis/gate_threshold_calibration_20260411/gate_threshold_calibration_table.csv)
- report: [gate_threshold_calibration_report.md](/home/rodrigo/cVAe_2026_shape/outputs/analysis/gate_threshold_calibration_20260411/gate_threshold_calibration_report.md)
- positive pool: [positive_pool_rows.csv](/home/rodrigo/cVAe_2026_shape/outputs/analysis/gate_threshold_calibration_20260411/positive_pool_rows.csv)
- negative pool: [negative_pool_rows.csv](/home/rodrigo/cVAe_2026_shape/outputs/analysis/gate_threshold_calibration_20260411/negative_pool_rows.csv)
- script: [calibrate_twin_gate_thresholds.py](/home/rodrigo/cVAe_2026_shape/scripts/analysis/calibrate_twin_gate_thresholds.py)

## Calibration Method

This is an exploratory retrospective calibration.

Positive reference pool:

- pass-regime rows from historically strong runs
- includes the strongest full-data MDN references recorded in the project

Negative contrast pool:

- non-pass rows from weaker or regressed full-data runs

Important caveat:

- this is not yet an independent prospective validation study
- it is an empirical calibration aid for project thresholds

## Main Result

Several current thresholds are much looser than what historically strong twins
actually achieve.

That means the present gates are still useful operationally, but they are not
yet tight enough to represent the empirical envelope of the best models.

The two exceptions are:

- `delta_jb_stat_rel`
- `stat_qval`

These are already close to the right conceptual role:

- `JBrel` as a project-specific shape gate
- `q=0.05` as a statistical decision level, not an engineering threshold

## Recommended Thresholds

### Proposed twin-validation thresholds (`G1..G5`)

These are the recommended operational thresholds after the first empirical
calibration pass:

| Gate | Metric | Current | Proposed | Reading |
|---|---|---:|---:|---|
| `G1` | `cvae_rel_evm_error` | `0.10` | `0.04` | current threshold is clearly loose relative to strong historical twins |
| `G2` | `cvae_rel_snr_error` | `0.10` | `0.03` | should be tightened substantially |
| `G3` | `cvae_mean_rel_sigma` | `0.10` | `0.05` | current threshold is too permissive |
| `G3` | `cvae_cov_rel_var` | `0.20` | `0.15` | moderate tightening justified |
| `G4` | `cvae_psd_l2` | `0.25` | `0.18` | current threshold is looser than the positive envelope |
| `G5` | `cvae_delta_skew_l2` | `0.30` | `0.12` | large tightening justified by strong-reference pool |
| `G5` | `cvae_delta_kurt_l2` | `1.25` | `0.17` | current threshold is dramatically too loose |
| `G5` | `delta_jb_stat_rel` | `0.20` | `0.20` | keep for now; already close to the positive envelope |

### `G6`

Recommendation:

- keep `stat_mmd_qval > 0.05`
- keep `stat_energy_qval > 0.05`

But:

- do **not** interpret `0.05` as an empirically calibrated engineering
  threshold
- interpret it as a conventional statistical decision alpha

So `G6` should be standardized by:

- `stat_mode`
- `stat_max_n`
- `stat_n_perm`

and by wording, not by moving alpha away from `0.05`.

## Why These Numbers

The script used the positive-pool empirical envelope plus a small margin for
the lower-better engineering metrics.

This is intentionally conservative:

- it preserves all positive reference rows in the calibration pool
- it avoids over-tightening to the exact best run
- it still removes thresholds that are clearly too loose

## Interpretation

### What tightened the most

The strongest changes are:

- `G1` / `G2`
- `G4`
- `G5 skew`
- `G5 kurt`

This means the current protocol has been especially permissive on:

- direct relative signal error
- PSD mismatch
- higher-order moment mismatch

### What stayed almost unchanged

- `JBrel = 0.20`
- `G6 q = 0.05`

This is a useful result.

It suggests:

- the current `JBrel` threshold was already close to the empirical boundary of
  historically strong twins
- the issue with `G6` is mostly interpretation and test budget, not alpha

## Recommended Protocol Change

After this calibration, the cleanest validation design is:

1. `validation_status_twin`
   - uses calibrated `G1..G5`

2. `stat_screen_pass`
   - uses `G6`

3. `validation_status_full`
   - optional conservative status including `G6`

This preserves the formal statistical screen, but stops it from dominating the
engineering definition of a useful digital twin.

## Next Step

The next methodological step should be:

1. implement the split between twin validation and statistical screen
2. if we accept these new thresholds, update the canonical gate table in
   [validation_summary.py](/home/rodrigo/cVAe_2026_shape/src/evaluation/validation_summary.py)
3. rerun gate recomputation on key finalists to see how the historical picture changes
