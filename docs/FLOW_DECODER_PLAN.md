# Flow Decoder Plan

This document is the active plan for the branch:

- `feat/conditional-flow-decoder`

## Why This Branch Exists

The current evidence across the recent runs is:

- the Gaussian seq reference is still the strongest protocol result:
  - `outputs/exp_20260324_023558`
  - `10/12` passes
- the conservative `mdn3` line became competitive:
  - `outputs/exp_20260325_230938`
  - `9/12` passes
- broader MDN sweeps did not close the remaining gap
- the remaining failures are concentrated in near-range regimes, with
  distribution-shape mismatch rather than only mean/covariance mismatch

The literature review that motivated this branch points to the decoder
likelihood as the next structural bottleneck:

- Tomczak: latent-variable models can be improved through the posterior, prior,
  and decoder
- Kingma / IAF: posterior flows improve variational inference, i.e. the
  inference side
- Kingma / Variational Lossy Autoencoder: stronger decoders improve generative
  modeling, but overly strong autoregressive decoders can also bypass the latent

Our current empirical reading is:

- the main mismatch is in the observation model `p(y | x, d, c, z)`
- not primarily in `q(z | x, d, c, y)` or in the prior family

## Main Hypothesis

Replace the current Gaussian / MDN decoder head with a conditional flow over
the residual:

- `r_t = y_t - x_t`

and model:

- `p(r_t | h_t, z_t, x_t, d, c)`

using an exact-likelihood conditional normalizing flow.

Because the output is only 2-D (`I`, `Q`), a flow decoder is relatively cheap
and should be more stable than a large mixture sweep.

## Proposed Architecture

Keep:

- `arch_variant="seq_bigru_residual"`
- encoder / prior / latent path
- context features:
  - sequence state
  - latent sample
  - center sample `x_t`
  - regime scalars `d`, `c`

Change:

- the decoder output head

Target structure:

1. a conditioner network emits flow parameters from the context
2. a 2-D base distribution is defined in residual space
3. a stack of conditional coupling or spline transforms maps base residuals to
   the target residual law
4. training uses the exact log-likelihood of the transformed residual

Recommended starting point:

- implement a smaller exact proof first:
  - conditional `sinh-arcsinh` flow per output axis
  - exact likelihood
  - exact stochastic sampling
  - deterministic median-like representative via base sample `eps=0`
- if this proof is promising, then move to a richer 2-D coupling / spline stack

## Training Objective

Stage 1 should be intentionally simple:

- `L_total = L_flow_nll + beta * KL_fb`

Initial settings:

- keep `beta` in the recent stable range:
  - `0.002` as the first anchor
- keep `free_bits=0.10`
- start with:
  - `lambda_mmd = 0.0`
  - `lambda_axis = 0.0`
  - `lambda_psd = 0.0`

Only after the plain flow likelihood is stable:

- optionally add a small `lambda_mmd`
- optionally add a small axis-shape penalty

Do **not** start with PSD loss in this branch.

## Execution Plan

### Phase 1 — Plumbing

Goal:

- add the conditional flow decoder head
- keep the existing seq cVAE training pipeline working
- ensure serialization / reload / inference work

Deliverables:

- config keys for flow decoder selection
- smoke preset for `seq_flow_smoke`
- unit tests for shape, sampling, and reload

Current status:

- implemented
- smoke preset added:
  - `seq_flow_smoke`
- structural smoke completed:
  - `outputs/exp_20260326_033237`
- result:
  - the flow branch is wired end-to-end
  - it is not yet scientifically competitive
 - note:
   - the structural smoke kept a small `lambda_mmd=0.25` anchor only to
     exercise the auxiliary-loss path during the first end-to-end integration
   - Phase 2 returns to the intended plain-flow objective

### Phase 2 — Single-Candidate Quick Proof

Goal:

- prove the branch can train and evaluate end-to-end
- avoid a grid before we know the new likelihood is numerically sane

Candidate shape:

- `seq_bigru_residual`
- `latent_dim=4`
- `W=7`, `h=64`
- `beta=0.002`
- `free_bits=0.10`
- flow head only

Quick mode:

- keep all `12` regimes
- cap train / validation / stat samples

Preset:

- `seq_flow_proof_quick`

Recommended command:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset seq_flow_proof_quick \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --max_samples_per_exp 100000 \
  --max_val_samples_per_exp 20000 \
  --max_dist_samples 20000 \
  --stat_tests --stat_mode quick --stat_max_n 2000 \
  --no_data_reduction \
  --no_baseline \
  --train_regime_diagnostics_enabled 0
```

### Phase 3 — Small Flow Grid

Only if Phase 2 is sane.

Explore a small grid around:

- number of flow transforms
- coupling type
- `beta`
- a small `lambda_mmd` reintroduced only if needed

### Phase 4 — Full Protocol Compare

Only after a quick run is competitive.

Compare against:

- Gaussian reference:
  - `outputs/exp_20260324_023558`
- best MDN quick:
  - `outputs/exp_20260325_230938`

## Acceptance Criteria

Quick-proof success means:

- no numerical blow-up
- protocol completes
- no severe variance inflation
- `1.0 m` and `1.5 m` remain strong
- at least one of the hard `0.8 m` regimes improves in distribution-shape
  metrics without broad regression

Branch-level success means:

- match or exceed the MDN quick line
- then challenge the Gaussian reference

## What Not To Do First

Do not prioritize these before the flow decoder proof exists:

- more MDN sweeps
- PSD loss
- posterior-flow-only work
- prior-only changes
- large autoregressive decoder changes

## Immediate Next Step

Run a short proof beyond the structural smoke, then decide whether to:

- tune the current `sinh-arcsinh` flow line
- or move directly to a richer conditional coupling / spline decoder
