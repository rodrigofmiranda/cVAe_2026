# AI Agent Guide

Shared project context for Codex, Claude, Gemini, and any other secondary
assistant.

Tool-specific root files still exist because some tools auto-discover them:

- [CODEX.md](/workspace/2026/feat_seq_bigru_residual_cvae/CODEX.md)
- [CLAUDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/CLAUDE.md)

Those root files should stay short and point here for common context.

## Current Focus

- active branch: `feat/mdn-g5-recovery`
- goal:
  recover the remaining `G5` failures near `0.8 m` without destabilizing the
  best MDN line

## Minimal Read Order

Read only these first:

1. [README.md](/workspace/2026/feat_seq_bigru_residual_cvae/README.md)
2. [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
3. [active/WORKING_STATE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/WORKING_STATE.md)
4. [reference/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/PROTOCOL.md)
5. [reference/EXPERIMENT_WORKFLOW.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/EXPERIMENT_WORKFLOW.md)

Add a sixth file only if the task needs it.

## Current Scientific Anchors

- stable Gaussian reference:
  - `outputs/exp_20260324_023558`
  - `10/12`
- best MDN line so far:
  - `outputs/exp_20260325_230938`
  - `9/12`
- negative lines already tested:
  - `sample-aware MMD`
  - current `sinh-arcsinh` flow line
  - pure regime-weighted resampling

## Active Architecture Families

- `seq_bigru_residual`
- `delta_residual`
- `legacy_2025_zero_y`

## Non-Negotiable Invariants

- split remains per experiment, temporal `head=train`, `tail=val`
- windowing happens after split
- windows never cross experiment or train/val boundaries
- deterministic inference must still support `EVM/SNR`
- stochastic inference must still support distribution metrics
- saved-model layer names remain compatible:
  - `encoder`
  - `prior_net`
  - `decoder`
- serious experimentation should go through `src.protocol.run`

## Canonical Entry Point

```bash
python -m src.protocol.run
```

## Run Review Minimum

When analyzing a finished run, inspect:

- `tables/protocol_leaderboard.csv`
- `tables/summary_by_regime.csv`
- `tables/residual_signature_by_regime.csv`
- `tables/stat_fidelity_by_regime.csv`
- `train/tables/gridsearch_results.csv`

Do not accept a line based only on train-side ranking.

## Quick Session Restart

Use this at handoff:

```text
Read:
1. README.md
2. PROJECT_STATUS.md
3. docs/active/WORKING_STATE.md
4. docs/reference/PROTOCOL.md
5. docs/reference/EXPERIMENT_WORKFLOW.md

Then inspect the latest exp_* and summarize:
- winner
- pass/fail by regime
- what changed vs the current reference
- the single best next test
```

## Tool-Specific Additions

- Codex-specific behavior:
  [CODEX.md](/workspace/2026/feat_seq_bigru_residual_cvae/CODEX.md)
- Claude-specific behavior:
  [CLAUDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/CLAUDE.md)
- review-only criteria:
  [docs/agents/REVIEW.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/REVIEW.md)
