# Conditional Diffusion Guide

This note is the short branch-local guide for the next global generative route.

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

## Scope Of The First Diffusion Iteration

- Architecture family:
  - `seq_bigru_residual` only
- Domain:
  - residual-domain generation first
  - sample `residual`, then reconstruct `y = x + residual`
- Conditioning:
  - keep the existing conditioning path:
    - signal window
    - distance
    - current
    - latent `z`
- Sampling:
  - stochastic path for protocol / distribution metrics
  - deterministic path via a reproducible sampler, not an ad-hoc mean hack

## Minimal Integration Points

These are the files the first implementation should touch.

- `src/models/cvae.py`
  - allow `decoder_distribution="diffusion"` for `arch_variant="seq_bigru_residual"`
  - keep the guard in place for unsupported architecture families
- `src/models/cvae_sequence.py`
  - extend `build_seq_decoder(...)`
  - extend `build_seq_cvae(...)`
  - extend `create_seq_inference_model(...)`
  - add any diffusion-specific decoder heads / timestep conditioning here
- `src/models/losses.py`
  - extend decoder distribution resolution beyond `gaussian` / `mdn`
  - add diffusion training objective
  - add deterministic point helper for report / EVM / SNR
  - add stochastic sampler for protocol evaluation
- `src/evaluation/report.py`
  - keep `decoder_sensitivity` finite for the new decoder family
  - route deterministic evaluation through the diffusion-compatible path
- `src/training/grid_plan.py`
  - add one structural smoke preset
  - add one small guided quick preset anchored near `S27`
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
   - read protocol first, not only train-side ranking

## Guardrails

- Do not reopen the negative flow families in this branch.
- Do not start with a large sweep.
- Do not generalize diffusion to every architecture family before the seq route
  proves structural viability.
- Preserve the current protocol, artifact layout, and model-reuse workflow.
