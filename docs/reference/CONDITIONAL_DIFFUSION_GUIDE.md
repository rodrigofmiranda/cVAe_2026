# Conditional Diffusion Guide

This note is the short branch-local guide for the second diffusion
formulation.

## Why This Route Exists

- The open scientific problem is **global**:
  - even passing regimes still show residual constellations that are more
    uniform than the real data
  - the current families learn the bulk of the distribution but not the full
    residual shape
- The best current anchor is still:
  - `outputs/exp_20260328_153611`
  - `S27cov_lc0p25_tail95_t0p03`
  - `10/12`
- The negative decoder-family lines already closed are:
  - old `sinh-arcsinh`
  - `coupling_2d`
  - `spline_2d`
- The first diffusion formulation is also now closed:
  - smoke `outputs/exp_20260329_210444`
  - guided quick `outputs/exp_20260329_211418`
  - both finished `0/12`

## Scope Of Diffusion V2

- Architecture family:
  - `seq_bigru_residual` only
- Domain:
  - residual-domain generation first
  - sample `residual`, then reconstruct `y = x + residual`
- Conditioning:
  - keep the existing observable conditioning path:
    - signal window
    - distance
    - current
  - do not depend on the old latent `z` path as the main stochastic carrier
- Sampling:
  - stochastic path for protocol / distribution metrics
  - deterministic path via a reproducible sampler, not an ad-hoc mean hack
- Training target:
  - start with `v-pred`
  - keep `x0-pred` as the first fallback ablation

## Main Formulation Change

- `diffusion v1` embedded diffusion inside the cVAE scaffold and kept:
  - encoder
  - conditional prior
  - KL path
- `diffusion v2` should move toward a **direct conditional residual diffusion**
  formulation:
  - diffusion is the primary generator
  - conditioning comes from `(x_window, d, c)`
  - the latent-KL path should be removed or sharply weakened

## Minimal Integration Points

These are the files the first `diffusion v2` implementation should touch.

- `src/models/cvae.py`
  - decide whether to keep `decoder_distribution="diffusion"` under the current
    builder or expose a new direct route cleanly
- `src/models/cvae_sequence.py`
  - add the direct conditional denoiser path
  - keep save/load compatibility explicit
  - keep deterministic and stochastic inference explicit
- `src/models/losses.py`
  - add `v-pred` training objective
  - optionally add `x0-pred` ablation support
- `src/evaluation/report.py`
  - keep `decoder_sensitivity` finite for the new decoder family
  - route deterministic evaluation through the diffusion-compatible path
- `src/training/grid_plan.py`
  - add one structural smoke preset for v2
  - add one small guided quick preset for v2
- `tests/test_seq_cvae_build.py`
  - model builds and output shape expectations
- `tests/test_hybrid_loss_and_mdn.py`
  - save/load and inference coverage for the new family
- `tests/test_grid_plan.py`
  - preset selection and candidate-shape coverage

## First Milestones

1. Structural milestone
   - build
   - train for a few epochs
   - save/load
   - deterministic inference
   - stochastic inference
   - smoke run via `src.protocol.run`

2. Scientific milestone
   - guided quick with a very small grid
   - compare against the `S27` anchor
   - compare against the `diffusion v1` verdict
   - read protocol first, not only train-side ranking

## Current Status

- diffusion v1 verdict:
  - `outputs/exp_20260329_210444`: `0/12`
  - `outputs/exp_20260329_211418`: `0/12`
  - interpretation:
    - the family is structurally integrable
    - but the current `cVAE + diffusion + KL` formulation is negative
- diffusion v2 status:
  - no code yet
  - no smoke yet
  - this branch exists to make the formulation change explicit before coding

## Guardrails

- Do not reopen the negative flow families in this branch.
- Do not start with a large sweep.
- Do not reopen another local hyperparameter sweep of `diffusion v1`.
- Do not generalize diffusion to every architecture family before the seq route
  proves structural viability.
- Preserve the current protocol, artifact layout, and model-reuse workflow.
