# Delta Residual Adv Status

Current status of the experimental `arch_variant="delta_residual_adv"` line.

## Goal

Test whether a point-wise **conditional residual cVAE-GAN** can improve the
hard residual-distribution metrics that remain weak in the non-adversarial
point-wise line:

- higher-order moments
- Jarque–Bera mismatch
- MMD / Energy
- difficult regimes such as `0.8 m`

This branch keeps the `delta_residual` backbone and adds a conditional
discriminator over `(x, d, c, Δ)`.

## Implemented

- new architecture integrated in the main pipeline:
  - `src/models/cvae.py`
  - `src/models/adversarial.py`
  - `src/models/discriminator.py`
  - `src/models/losses.py`
  - `src/training/grid_plan.py`
- the adversarial line now supports:
  - `src.protocol.run`
  - `--train_once_eval_all`
  - `--reuse_model_run_dir`
  - operational training artifacts

## Important Fixes Applied On 2026-03-20

These fixes are the baseline for any valid scientific rerun:

- commit: `ee2681f`
  - adversarial critic now consumes **sampled residuals**, not only residual means
  - MC grid ranking no longer uses the optimistic pre-fix mean shortcut
  - saved adversarial checkpoints preserve the full trainable wrapper path
  - operational training diagnostics are generated for the branch as well

## Historical Runs

### 1. Original adversarial sweep

- run: `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260320_012223`
- note:
  - obsolete for scientific comparison
  - used the pre-fix adversarial path

### 2. Corrected setup attempt before the core fix

- run: `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260320_014614`
- preset: `delta_residual_adv_med`
- note:
  - still obsolete as scientific reference
  - produced before `ee2681f`

### 3. Non-adversarial comparison run

- run: `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260320_020652`
- preset: `delta_residual_fast`
- note:
  - useful only as historical comparison
  - not a final reference against the corrected adversarial branch

## Current Scientific Status

- `delta_residual_adv` is now technically aligned with the main protocol path
- old March 20 adversarial runs should **not** be used as final evidence
- fresh reruns are still required before any scientific conclusion

At the moment, the branch should be treated as:

- operationally integrated
- architecturally corrected
- scientifically pending

## Current Comparison Baselines

Use these as the current references when evaluating new adversarial reruns:

- strongest current temporal reference:
  - `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260320_171510`
- historical all-gates-passed seq reference:
  - `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_204149`
- best current non-adversarial point-wise scientific reference:
  - `/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_235319`

## Recommended Next Step

Run fresh protocol-first reruns with the corrected branch, preferably:

- a non-adversarial point-wise anchor
- an adversarial reduced sweep
- reduced multi-regime protocol:
  - `configs/all_regimes_sel4curr.json`

Do not advance this branch toward release or thesis claims until those reruns
exist and are compared under the current gates and stat-fidelity workflow.
