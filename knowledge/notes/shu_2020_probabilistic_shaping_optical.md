# shu_2020_probabilistic_shaping_optical

## Citation

- title: Probabilistic shaping: a step closer to the Shannon limit in channel capacity for optical communications
- authors: Chester Shu and Qiulin Zhang
- year: 2020
- venue: Electronics Letters

## Why It Matters Here

- It gives the practical framing for why shaping is attractive in optical links.
- It also gives the caution that the "best" format is not determined by AWGN
  shaping gain alone.
- For this repo, it supports the idea that geometry and probability should be
  matched to the actual hardware/channel constraints, not optimized in the
  abstract.

## Core Idea

- problem: improve optical transmission efficiency beyond conventional uniform
  QAM
- method: short tutorial-style comparison of probabilistic shaping (PS),
  geometric shaping (GS), and practical limitations in optical systems
- principal hypothesis: PS provides real gains, but practical resource limits
  can change which modulation format is actually best

## What To Extract For This Repo

- architecture: not an ML paper
- loss: not applicable
- inferencia: not applicable
- tratamento do residual: indirect only; discusses nonlinear phase noise and
  practical penalties
- metricas: NGMI, SNR / OSNR, reach
- sinal de colapso: not applicable
- resultados por regime ou condicao: optimality changes with linewidth and
  ADC resolution

## Key Claims

- PS can approach the classical shaping gain of about `1.53 dB` for AWGN-like
  conditions.
- PS is easier to integrate than GS because the DSP chain can stay compatible
  with standard QAM geometries.
- Higher-cardinality PS is not always best in practice; laser linewidth and
  converter resolution can reverse the preferred choice.

## Useful Details For Prompts

- "probabilistic shaping"
- "geometric shaping"
- "NGMI"
- "laser linewidth"
- "PAPR"
- "ADC ENOB"
- "non-power-of-two constellation for PS"

## Relevant Quotes / Pages

- p.: discussion of PS vs GS and the compatibility advantage of PS
- p.: discussion of phase-noise sensitivity and ADC-resolution sensitivity

## Limitations

- Very short paper; mainly conceptual and illustrative.
- Does not provide a full sequence-level treatment of shaping under channel
  memory.
- Does not solve the design problem for a specific hardware platform such as
  the VLC bench in this repo.

## Relevance To VLC Digital Twin

- conexao direta: supports the need to evaluate signaling geometry under real
  hardware limits rather than under idealized AWGN assumptions
- conexao indireta: motivates comparing support geometries with different PAPR
  and edge behavior
- nao aplicavel: does not itself define an ML training recipe

## Follow-up Actions

- testar no repo: compare `full_square` and `full_circle` under equal average
  power and inspect edge distortion
- comparar com resultado atual: use this paper to justify why corner-heavy
  supports may be physically suboptimal even if they probe the channel well
- usar em sintese: pair with Askari and Lampe (2025) for the memory /
  nonlinearity-aware view

