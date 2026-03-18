# Review Instructions

Use this file for Claude Code review-specific behavior on this repository.

If using Claude Code interactively, prefer invoking `/seq-review` so the review
starts with the correct scope and output format.

## Primary Review Focus

Prioritize:

1. behavioral regressions
2. protocol-invariant violations
3. train/validation leakage
4. inference mismatches between train and eval
5. scientific-invalidity risks, even when code looks clean

## For The seq-BiGRU Residual Project

When reviewing work related to `feat/seq-bigru-residual-cvae`, focus on:

- windowing happens only after temporal per-experiment split
- windows never cross:
  - experiment boundaries
  - train/val boundaries
- the sequence path preserves one output per original sample
- deterministic inference still supports `EVM/SNR`
- stochastic `mc_concat` still supports distribution metrics
- the saved model still exposes compatible layer names:
  - `encoder`
  - `prior_net`
  - `decoder`
- legacy paths for `concat` and `channel_residual` remain intact
- new grids and defaults do not silently change baseline project behavior
- tests cover:
  - window construction
  - no leakage
  - model build
  - inference path
  - protocol compatibility

Review output expectations:

- findings first
- ordered by severity
- include file and line references
- explicitly call out missing tests or unverified assumptions
- keep summaries brief and secondary

## Out Of Scope

Do not spend review effort on:

- `knowledge/`
- local paper indexing
- generated experiment outputs under `outputs/`
- local dataset contents under `data/`

unless the user explicitly asks for those areas.
