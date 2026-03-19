# Conditional Residual cVAE-GAN Plan

## Short Answer

Yes: if we keep the current conditional VAE backbone and add an adversarial
term with a discriminator, the result is best described as a
**conditional residual cVAE-GAN**.

More precisely, in this repository it should be treated as:

- a **conditional** model
- trained on the **residual target** `Δ = Y - X`
- with the current **cVAE** latent structure preserved
- plus an **adversarial loss** that pushes generated residuals toward the
  real residual distribution

This is **not** a pure GAN. It is a residual-target cVAE with an additional
adversarial objective.

## Why This Variant Makes Sense Here

Current lessons from the branch:

- `delta_residual` is the best point-wise family so far
- `seq_bigru_residual` is the strongest temporal family and the only line with
  a historical all-gates-passed reference
- the hardest remaining gaps are not basic signal mapping alone; they are
  distributional:
  - higher-order moments
  - tails / Gaussianity mismatch
  - spectral shape
  - formal two-sample similarity (`MMD`, `Energy`)

An adversarial term may help the generator capture residual structure that
the current ELBO-style objectives still smooth out.

## Recommended Naming

Use a new `arch_variant`, not a separate codebase fork.

Recommended progression:

- first implementation:
  - `arch_variant="delta_residual_adv"`
- later, only if justified:
  - `arch_variant="seq_bigru_residual_adv"`

The first target should be the point-wise residual line, because it is simpler,
faster to iterate on, and already operationally stable.

## Core Formulation

### Generator

Keep the current `delta_residual` semantics:

- input: `(x, d, c)`
- latent path: cVAE with encoder/prior/decoder
- target: `Δ = Y - X`
- inference: `Ŷ = X + Δ̂`

Generator training objective:

- residual reconstruction term
- KL term
- optional existing auxiliary terms (`MMD`, later if still useful)
- adversarial term

Conceptually:

`L_G = L_recon + β L_KL + λ_adv L_adv [+ λ_mmd L_mmd]`

### Discriminator / Critic

The discriminator should be **conditional**, not unconditional.

Recommended first discriminator input:

- real branch:
  - `(x, d, c, Δ_real)`
- fake branch:
  - `(x, d, c, Δ_fake)`

The discriminator learns whether the residual came from the real channel or
from the generator, given the same conditioning context.

This is the key reason the model should be called a **conditional residual
cVAE-GAN**.

## Recommended Training Strategy

### Stage 1 — Point-wise adversarial residual model

Implement only:

- `delta_residual_adv`

Do **not** start with the seq model.

Why:

- easier to stabilize
- cheaper protocol iterations
- fewer moving parts when debugging adversarial failure modes
- we already know `delta_residual` is the best point-wise baseline

### Stage 2 — Temporal adversarial residual model

Only after the point-wise version is stable:

- extend to `seq_bigru_residual_adv`

This should be treated as a second project step, not the initial patch.

## Loss Design Recommendation

### Generator loss

Keep the current cVAE loss as the base:

- `recon / heteroscedastic residual objective`
- `KL`

Then add:

- `λ_adv * adversarial_generator_loss`

Start simple:

- keep `β` and `free_bits` exactly as they are
- do not redesign the probabilistic core in the first patch

### Discriminator loss

Recommended choice:

- **hinge GAN** or **WGAN-GP**

Pragmatic recommendation:

- start with **hinge**
  - simpler and easier to integrate
- use `WGAN-GP` only if hinge shows obvious instability

## What The Discriminator Should See

### First version

Use a compact MLP discriminator on:

- `concat(x, d, c, Δ)`

This is enough for the first point-wise implementation.

### Later option

If spectral and temporal structure remain weak:

- add a discriminator over short residual windows
- or use a two-branch critic:
  - point-wise residual branch
  - short-window residual branch

But this is **not** part of the first implementation.

## Protocol-First Constraint

This variant must be implemented inside the **existing protocol-first
pipeline**, not in a separate training-only experiment path.

Non-negotiable:

- use `src.protocol.run`
- support `--train_once_eval_all`
- preserve save/load compatibility
- preserve `encoder`, `prior_net`, `decoder` layer names
- keep output artifacts and summary tables inside the current protocol layout

## Proposed File-Level Implementation

### New / updated model code

- `src/models/cvae.py`
  - dispatch `arch_variant="delta_residual_adv"`
- `src/models/losses.py`
  - adversarial generator loss terms
- `src/models/discriminator.py` (new)
  - conditional residual discriminator
- `src/models/adversarial.py` (optional new helper)
  - helper functions / training utilities

### Training loop changes

The current compile/fit path is too simple for a real GAN objective.
We will likely need one of:

- a custom `Model.train_step`
- or a dedicated adversarial training wrapper

Recommended approach:

- implement a custom training model/wrapper with alternating updates:
  - 1 discriminator step
  - 1 generator step

### Grid integration

- `src/training/grid_plan.py`
  - add smoke preset
  - add reduced exploratory preset

### Protocol integration

- `src/protocol/run.py`
  - should not need conceptual changes
  - only ensure save/load and ranking continue to work

### Evaluation

No special-casing in evaluation unless required.
The new variant should still be judged by the same protocol metrics/gates.

## Implementation Checklist

### Phase 0 — Contract

- add `arch_variant="delta_residual_adv"`
- keep `delta_raw = Y - X` semantics unchanged
- keep protocol-first flow as the only supported experiment path
- preserve model save/load compatibility for:
  - full model
  - decoder
  - prior net

### Phase 1 — Minimal adversarial model

- implement a conditional residual discriminator
- implement generator adversarial loss
- implement discriminator loss
- add alternating train step with:
  - 1 discriminator update
  - 1 generator update
- keep inference identical to current `delta_residual`:
  - predict `Δ̂`
  - reconstruct `Ŷ = X + Δ̂`

### Phase 2 — Protocol integration

- add smoke preset for `delta_residual_adv`
- add reduced exploratory preset for `delta_residual_adv`
- ensure `src.protocol.run --train_once_eval_all` works end-to-end
- ensure leaderboard/ranking still works with the new variant

### Phase 3 — Evaluation and decision

- compare against the strongest current `delta_residual` candidates
- measure whether adversarial training improves:
  - `G5`
  - `G6`
  - `PSD`
  - `rho_hetero`
  - `JSD`
- reject the variant if it improves visual realism but hurts `EVM/SNR`

## Proposed Milestones

### Milestone A — Architectural smoke

Goal:

- build model
- run one train/eval smoke
- save/load works
- inference works

Acceptance:

- no regressions to existing variants
- model can be saved and reloaded by the current protocol pipeline
- one reduced protocol run completes without custom post-processing

### Milestone B — Reduced protocol exploratory sweep

Protocol:

- `configs/one_regime_1p0m_300mA_sel4curr.json`
- `--train_once_eval_all`

Compare:

- best current `delta_residual`
- new `delta_residual_adv`

Goal:

- determine whether adversarial loss improves:
  - `G5`
  - `G6`
  - `PSD`
  - `MMD/Energy`
  - `JSD`
  - `rho_hetero`

### Milestone C — Only if justified

Extend to:

- `seq_bigru_residual_adv`

## Initial Hyperparameter Plan

Do not open a huge adversarial grid immediately.

Start with:

- `λ_adv in {0.01, 0.05, 0.1}`
- keep:
  - `beta=0.001`
  - `free_bits=0.0`
  - `layer_sizes` around current best `delta_residual`
  - `latent_dim` around `4–6`

If unstable:

- reduce `λ_adv`
- lower discriminator capacity
- reduce discriminator update ratio

## Main Risks

- adversarial instability
- mode collapse
- degradation of signal fidelity (`EVM`, `SNR`) while trying to improve tails
- critic overpowering the cVAE objective
- harder reproducibility across GPUs / TF stacks

## Acceptance Criteria For First Merge

The first implementation should be merged only if all of the following hold:

- existing `delta_residual` and `seq_bigru_residual` variants still build
- `delta_residual_adv` trains and saves inside `src.protocol.run`
- protocol summary tables are generated without special-case hacks
- no output-layout regressions are introduced
- at least one reduced-protocol run shows competitive `EVM/SNR`
- the new variant does not require a separate training-only entrypoint

## Decision Rule

This variant is worth continuing only if it improves the current residual line
on the protocol metrics that still hurt us most:

- `G5`
- `G6`
- `PSD`
- higher-order residual fidelity

If it only makes plots “look more realistic” while worsening `EVM/SNR` or
destabilizing training, it is not a win.

## Recommended Next Implementation Step

Implement:

- `delta_residual_adv`
- point-wise only
- protocol-first
- small reduced-protocol exploratory sweep

Do **not** start with:

- pure GAN
- seq adversarial model
- a massive adversarial grid
