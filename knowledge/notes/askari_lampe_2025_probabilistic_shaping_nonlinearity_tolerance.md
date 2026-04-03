# askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance

## Citation

- title: Probabilistic Shaping for Nonlinearity Tolerance
- authors: Mohammad Taha Askari and Lutz Lampe
- year: 2025
- venue: Journal of Lightwave Technology

## Why It Matters Here

- This is the strongest conceptual bridge between shaping and a real,
  non-ideal channel.
- It explains why the optimal signaling strategy depends not only on marginal
  distributions, but also on moments, channel memory, receiver processing, and
  sequence-level effects.
- For this repo, it supports moving from "which support is easier to learn" to
  "which support/sequence structure is better matched to the actual channel."

## Core Idea

- problem: conventional probabilistic shaping optimized for AWGN can be
  suboptimal in nonlinear channels
- method: tutorial and analysis based on perturbation models, EGN intuition,
  PAS, finite-length shaping, filter interpretation, and sequence selection
- principal hypothesis: nonlinearity tolerance depends on both signal moments
  and temporal structure; sequence-aware shaping can outperform purely i.i.d.
  marginal optimization

## What To Extract For This Repo

- architecture: not an ML architecture paper, but highly relevant to what a
  digital twin must represent
- loss: indirect guidance for loss design through moments, tails, and
  sequence-level behavior
- inferencia: not applicable in the VAE sense
- tratamento do residual: highlights multiplicative plus additive nonlinear
  distortion components and their relation to energy sequences
- metricas: effective SNR, AIR, shaping gain, EDI, windowed moments
- sinal de colapso: not applicable directly, but warns against oversimplified
  marginal modeling
- resultados por regime ou condicao: finite-length shaping, mapping, and CPR
  all change the apparent nonlinearity tolerance

## Key Claims

- Linear shaping gain and nonlinear shaping gain should be treated separately.
- Higher standardized moments can hurt nonlinearity tolerance even when the
  marginal distribution is good for AWGN.
- Finite-length shaping and sequence selection can improve performance because
  the channel has memory and responds to temporal energy structure.

## Useful Details For Prompts

- "linear shaping gain"
- "nonlinear shaping gain"
- "AMIN"
- "windowed moments"
- "energy dispersion index"
- "sequence selection"
- "carrier-phase recovery interaction"
- "PAPR and mapping matter"

## Relevant Quotes / Pages

- p.: linear vs nonlinear shaping gain decomposition
- p.: finite-length effects and the usefulness of windowed moments / EDI
- p.: sequence selection and sign-bit-aware metrics

## Limitations

- The paper is for optical fiber communication, not directly for VLC hardware.
- It is a conceptual and analytical guide, not a ready-made algorithm for this
  repository.
- Direct transfer to VLC should be done carefully and with experimental
  validation.

## Relevance To VLC Digital Twin

- conexao direta: argues that a faithful channel model should care about more
  than pointwise amplitude mapping
- conexao indireta: motivates sequence-aware digital twins and future
  autoencoder constraints on moments, tails, and temporal structure
- nao aplicavel: does not directly specify the exact signaling geometry for the
  current bench

## Follow-up Actions

- testar no repo: add radius- and amplitude-conditioned diagnostics, then test
  whether edge/corner regions behave like higher-order-moment penalties
- comparar com resultado atual: relate current `edge-gap` behavior to the idea
  that bulk statistics can look good while extreme structure remains wrong
- usar em sintese: use as the main paper motivating digital-twin-first,
  constellation-second research

