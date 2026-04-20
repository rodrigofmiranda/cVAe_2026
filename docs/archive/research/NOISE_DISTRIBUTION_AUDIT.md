# Noise Distribution Audit

> Updated on 2026-03-24.

## Scope

This note audits the current mismatch between the real residual-noise
distribution and the `seq_bigru_residual` cVAE prediction, with emphasis on the
best current protocol run:

- `outputs/exp_20260324_023558`
- winner:
  - `S6seq_W7_h64_lat4_b0p003_lmmd1p75_fb0p10_lr0p0003_L128-256-512`

The visual trigger for this audit is the repeated pattern seen in
`analysis_dashboard.png`, especially:

- `eval/dist_0p8m__curr_100mA/plots/champion/analysis_dashboard.png`
- `eval/dist_0p8m__curr_300mA/plots/champion/analysis_dashboard.png`

The real residual histograms are broader and flatter, while the cVAE residual
histograms are narrower and peak more strongly around zero.

## Main Finding

The current best seq model learns the coarse residual structure, but still
underestimates the width and tail structure of the residual-noise distribution
at short distance (`0.8 m`), especially in:

- `0.8 m / 100 mA`
- `0.8 m / 300 mA`

At longer distances (`1.0 m`, `1.5 m`), the same issue remains only as a mild
under-dispersion and already falls below the current gates.

In short:

- far regimes:
  - the model is close enough
- near regimes:
  - the model is still too concentrated and too regular

## Evidence From Artifacts

### Protocol outcome

Run `exp_20260324_023558` reached:

- `10/12` passes
- `2/4` passes at `0.8 m`
- `4/4` passes at `1.0 m`
- `4/4` passes at `1.5 m`

The only failing regimes are:

- `dist_0p8m__curr_100mA`
- `dist_0p8m__curr_300mA`

### Quantitative contrast: near vs far

From `tables/summary_by_regime.csv` in `exp_20260324_023558`:

Near failing regimes:

- `dist_0p8m__curr_100mA`
  - `delta_psd_l2 = 0.087889`
  - `delta_wasserstein_I = 0.007541`
  - `delta_wasserstein_Q = 0.003922`
  - `delta_jb_stat_rel_I = 6.011476`
  - `delta_jb_stat_rel_Q = 0.176534`
  - `stat_mmd_qval = 0.002999`
  - `stat_energy_qval = 0.002999`
- `dist_0p8m__curr_300mA`
  - `delta_psd_l2 = 0.084867`
  - `delta_wasserstein_I = 0.007112`
  - `delta_wasserstein_Q = 0.004477`
  - `delta_jb_stat_rel_I = 0.876943`
  - `delta_jb_stat_rel_Q = 0.477773`
  - `stat_mmd_qval = 0.002999`
  - `stat_energy_qval = 0.002999`

Far passing regimes:

- `dist_1p5m__curr_100mA`
  - `delta_psd_l2 = 0.023895`
  - `delta_wasserstein_I = 0.003336`
  - `delta_wasserstein_Q = 0.005580`
  - `delta_jb_stat_rel_I = 0.070695`
  - `delta_jb_stat_rel_Q = 0.005519`
  - `stat_mmd_qval = 0.628486`
  - `stat_energy_qval = 0.628486`
- `dist_1p5m__curr_500mA`
  - `delta_psd_l2 = 0.023692`
  - `delta_wasserstein_I = 0.004515`
  - `delta_wasserstein_Q = 0.004677`
  - `delta_jb_stat_rel_I = 0.043946`
  - `delta_jb_stat_rel_Q = 0.013131`
  - `stat_mmd_qval = 0.705647`
  - `stat_energy_qval = 0.665122`

Interpretation:

- the near/far visual difference is real
- near regimes fail mainly because the predicted residual distribution remains
  too concentrated and too shape-regular
- the strongest axis-specific pathology is in `I` for `0.8 m / 100 mA`

### Important diagnostic nuance

In `dist_0p8m__curr_100mA`, aggregate `G5` still passes, yet:

- `delta_jb_stat_rel_I = 6.011476`

This means the current aggregate gate can hide a severe one-axis mismatch even
when the overall gate says the higher moments are acceptable.

## Audit of the Current Objective

### What the model is asked to optimize

The active seq model is a heteroscedastic conditional Gaussian decoder:

- decoder output:
  - `(mean_I, mean_Q, logvar_I, logvar_Q)`
- sequence builder:
  - [cvae_sequence.py](../../../src/models/cvae_sequence.py#L423)
- reconstruction loss:
  - heteroscedastic Gaussian NLL in [losses.py](../../../src/models/losses.py#L34)

This setup is good for learning:

- central tendency
- per-axis uncertainty scale
- coarse residual geometry

But it does not, by itself, force the sampled residual cloud to match the full
target distribution shape.

### Why the current MMD term is not enough

The extra `MMD` penalty is currently applied to the residual mean, not to a
sample drawn from the predicted decoder distribution.

For the seq family:

- `r_real = y_true - x_center`
- `r_gen  = y_mean - x_center`
- [losses.py](../../../src/models/losses.py#L265)

For the point-wise explicit residual family:

- `MMD(delta_true, delta_mean)`
- [losses.py](../../../src/models/losses.py#L353)

This is the core audit finding.

It means the current distribution-matching term mostly regularizes the center of
the predicted residual cloud, not the actual stochastic spread produced by
`(mean, logvar)`.

That is exactly consistent with the plots:

- cVAE histogram too narrow
- cVAE histogram too tall near zero
- coarse alignment improves with larger `lambda_mmd`
- fine width/tail mismatch remains, especially near `0.8 m`

### Decoder variance clamp audit

The decoder variance clamp is:

- `DECODER_LOGVAR_CLAMP_LO = -5.82`
- `DECODER_LOGVAR_CLAMP_HI = -0.69`
- [defaults.py](../../../src/config/defaults.py#L169)

This implies approximately:

- variance range:
  - `[0.00297, 0.50158]`
- standard-deviation range:
  - `[0.05448, 0.70822]`

Conclusion:

- the clamp may still matter operationally
- but it does not look like the primary reason for the narrow histograms
- the primary issue is more likely objective mismatch than hard variance ceiling

### Structural limitation of the decoder family

The active decoder family is still:

- diagonal Gaussian per timestep
- unimodal
- independent across `I` and `Q` in the decoder likelihood

So even with better loss weighting, there is a limit to how well it can match:

- sharp heavy tails
- skewed or axis-asymmetric marginals
- richer multi-shape residual clouds

This is a model-family limitation, not necessarily a pipeline bug.

## Scientific Interpretation

Current best reading:

- the seq family is scientifically valid and already strong
- it learns the residual envelope and much of the conditional structure
- it does **not yet** learn the full residual-noise distribution at short
  distance
- the remaining problem is no longer “general quality”
- it is specifically “distribution width and tail-shape mismatch near `0.8 m`”

## Recommended Action Ladder

### Last hyperparameter-only step

Run the focused finishing sweep:

- preset:
  - `seq_finish_0p8m`

Reason:

- it is still the last reasonable no-code attempt
- it directly targets the two remaining failing `0.8 m` regimes
- it tests stronger `lambda_mmd` and lower-LR hedges without reopening broad search

### If `seq_finish_0p8m` still fails

Do **not** continue with wider grid search first.

The next engineering step should be:

1. Sample-aware distribution matching
   - compute `MMD` on sampled residuals, not only on `y_mean`
2. Axis-aware marginal matching
   - add a light penalty for `I/Q` marginal mismatch
   - likely start with `Wasserstein_I/Q` or sampled `MMD_I/Q`
3. Optional regime weighting
   - only if needed after the first two fixes
   - give modest extra emphasis to `0.8 m` regimes during global training

## Proposed Implementation Direction

Minimal corrective change:

1. Draw a residual sample with the decoder variance:
   - `eps ~ N(0, I)`
   - `y_sample = y_mean + exp(0.5 * y_log_var) * eps`
   - `r_gen_sample = y_sample - x_center`
2. Replace current `MMD(r_real, y_mean - x_center)` with
   `MMD(r_real, r_gen_sample)`
3. Keep the existing heteroscedastic NLL
4. Add an optional small axis-wise penalty:
   - `lambda_axis * (W1_I + W1_Q)`
   or
   - `lambda_axis * (MMD_I + MMD_Q)`

This is the most direct fix for the exact symptom seen in the plots.

## Status

This audit does **not** claim a pipeline bug.

It claims:

- the remaining mismatch is real
- the current diagnostics are sufficient to localize it
- the current training objective explains the visual symptom well
- the next correction should be in the loss, not only in the search grid
