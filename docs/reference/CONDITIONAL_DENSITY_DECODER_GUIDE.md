# Conditional Density Decoder Guide

This guide records the project decision boundary around decoder families for
learning the residual-noise distribution.

Use this file when the question is:

- which model family should learn the noise law?
- was "flow" already tried here?
- what exactly failed, and what still remains scientifically open?

## Short Answer

Yes, a **flow decoder line was already implemented and tested** in this
project.

But it was a **specific flow family**:

- branch:
  - `feat/conditional-flow-decoder`
- decoder style:
  - conditional `sinh-arcsinh` flow per output axis
- status:
  - plumbing worked
  - scientific result was negative

That result does **not** mean "all conditional density models failed".

It only means:

- the current implemented `sinh-arcsinh` per-axis flow line was not good enough

## What Was Already Done

Historical flow branch:

- `feat/conditional-flow-decoder`

Key historical commits:

- `2f3c54f`
  - `docs(plan): add conditional flow decoder branch plan`
- `c41fd7e`
  - `feat(flow): add seq conditional flow decoder smoke`
- `e7eab4a`
  - `feat(flow): add quick proof preset`
- `20f52c9`
  - `feat(flow): add micro proof preset`
- `517de2b`
  - `feat(flow): add phase-3 stabilization grid`
- `8aea10c`
  - `docs(flow): scaffold phase-4 full compare`
- `607f984`
  - `docs(branch): discard flow line and return to mdn`

Historical plan files:

- [archive/active/MDN_G5_RECOVERY_PLAN.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/archive/active/MDN_G5_RECOVERY_PLAN.md)
- [archive/agents/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/archive/agents/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md)

Historical flow-specific plan:

- from branch `feat/conditional-flow-decoder`:
  - `docs/FLOW_DECODER_PLAN.md`

Project reading from those materials:

- the branch proved the flow plumbing works
- the exact tested family was a small conditional flow proof
- the current implementation family was **not scientifically competitive**
- the branch was therefore retired in favor of the stronger MDN line

Documented negative runs:

- `outputs/exp_20260326_034522`
  - plain flow micro proof
  - `0/12`
- `outputs/exp_20260326_035723`
  - stabilized flow quick grid
  - best candidate still `0/12`

## What That Historical Flow Actually Was

The historical flow decoder was **not** a full expressive 2-D coupling flow.

It was documented as:

- conditional `sinh-arcsinh` flow per output axis
- exact likelihood
- exact stochastic sampling
- deterministic representative point via base sample

That is an important distinction.

This project has **not** yet established a negative result for:

- conditional `RealNVP`
- conditional `MAF`
- conditional neural spline flow
- conditional diffusion / score-based decoder

## Why This Matters For The Noise Problem

The current scientific problem is not just mean/covariance mismatch.

The stronger reading from the recent runs is:

- the residual cloud is too clean
- the marginal noise shape is wrong
- tails / asymmetry / local shape are not matched reliably
- Gaussian decoders saturate
- MDN improved a lot, but still leaves shape mismatch in hard regimes

That makes the problem a **conditional density estimation** problem:

- learn `p(r | context)` where `r = y - x`

and not just:

- predict `E[r | context]`
- or predict a single Gaussian scale

## Recommended Decoder Families Going Forward

### 1. Best next structural test

Use a **conditional normalizing flow** over the residual.

Project-specific recommended version:

- keep the current sequential context encoder
- keep the latent path if useful
- predict a conditional 2-D residual law
- use an exact-likelihood 2-D flow that is more expressive than the retired
  per-axis `sinh-arcsinh` head

Best starting family:

- conditional neural spline flow

Practical alternatives:

- conditional `RealNVP`
- conditional `MAF`

Why this is the best next structural test:

- output dimension is only `2` (`I`, `Q`)
- exact likelihood is available
- shape flexibility is much better than Gaussian and usually more stable than a
  huge mixture sweep
- it is the closest structural extension to the historical flow branch without
  repeating the exact failed family

### 2. Higher-cost, higher-fidelity path

Use a **conditional diffusion / score-based decoder**.

This is stronger if the only priority is shape fidelity, but it is slower and
heavier operationally.

Use this if:

- a richer conditional flow still cannot match the hard residual law
- or the team is willing to trade speed for generative fidelity

## What Not To Reopen Blindly

Do not reopen, as if it were still unresolved:

- the current `sinh-arcsinh` per-axis flow line
- its old quick proofs
- a broad replay of the same historical flow branch

That question is already answered.

The open question is narrower:

- does a **more expressive conditional density decoder** solve the noise-shape
  mismatch?

## Project-Specific Implementation Target

If this route is reopened, the first target should be:

- `arch_variant="seq_bigru_residual"`
- model residual:
  - `r_t = y_t - x_t`
- condition on:
  - sequence state
  - latent sample or deterministic latent context
  - center sample `x_t`
  - regime scalars `d`, `c`
- decoder:
  - conditional 2-D spline flow

Secondary route:

- same idea for `seq_imdd_graybox`

But the first proof should happen in the stronger historical family:

- `seq_bigru_residual`

## Literature Pointers

These are the main paper families that support the current recommendation.

Conditional density via mixtures:

- Bishop, *Mixture Density Networks* (1994)
  - https://research.aston.ac.uk/en/publications/mixture-density-networks/

Exact-likelihood normalizing flows:

- Dinh et al., *Density Estimation using Real NVP* (2016)
  - https://arxiv.org/abs/1605.08803
- Papamakarios et al., *Masked Autoregressive Flow for Density Estimation* (2017)
  - https://arxiv.org/abs/1705.07057
- Durkan et al., *Neural Spline Flows* (2019)
  - https://arxiv.org/abs/1906.04032
- Winkler et al., *Learning Likelihoods with Conditional Normalizing Flows* (2019)
  - https://arxiv.org/abs/1809.00080

Diffusion / score-based generative modeling:

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020)
  - https://arxiv.org/abs/2006.11239
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (2021)
  - https://arxiv.org/abs/2011.13456

Communications-specific signal that this direction is relevant:

- Letafati et al., *Denoising Diffusion Probabilistic Models for Hardware-Impaired Communications* (2023)
  - https://arxiv.org/abs/2309.08568
- Wagle et al., *Physics-based Generative Models for Geometrically Consistent and Interpretable Wireless Channel Synthesis* (2025)
  - https://arxiv.org/abs/2506.00374

## Decision Rule

Use this rule in future planning:

- if someone asks "flow already failed, so should we drop generative density models?"
  - answer: no
- if someone asks "what exactly failed?"
  - answer: the old per-axis `sinh-arcsinh` conditional flow line
- if someone asks "what should be tried next for noise-distribution learning?"
  - answer: conditional 2-D spline flow first, conditional diffusion second
