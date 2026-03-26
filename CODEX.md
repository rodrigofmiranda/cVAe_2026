# Codex Session Handoff

This file stays at repo root because Codex-style tooling commonly looks for
`CODEX.md` automatically.

## Shared Project Context

Read the common guide first:

- [docs/agents/AI_AGENT_GUIDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/AI_AGENT_GUIDE.md)
- [docs/active/WORKING_STATE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/WORKING_STATE.md)

## Codex-Specific Start

On a fresh Codex session:

1. read [docs/agents/AI_AGENT_GUIDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/AI_AGENT_GUIDE.md)
2. inspect `git status --short`
3. if a run may be active, inspect `pgrep -af "src.protocol.run|src.training.train"`
4. run `python scripts/analysis/summarize_experiment.py` on the latest `outputs/exp_*`

## Codex-Specific Notes

- serious experimentation should happen through `src.protocol.run`
- prefer `--reuse_model_run_dir` when re-evaluating a shared trained model
- do not treat train-side ranking alone as scientific acceptance

## Out Of Scope

- do not use all 9 currents by default
- do not move `release/cvae-online` based on the archived adversarial line
