---
name: seq-review
description: Review seq_bigru_residual changes for protocol, leakage, compatibility, and validation risks
argument-hint: [scope]
context: fork
agent: Explore
disable-model-invocation: true
---

Review the current changes for this scope:

$ARGUMENTS

Use `REVIEW.md` and `docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md` as the project-specific review criteria.

Primary review focus:

1. behavioral regressions
2. protocol-invariant violations
3. train/validation leakage
4. inference mismatches between train and eval
5. scientific-invalidity risks, even when code looks clean

Required output format:

- findings first, ordered by severity
- each finding should include file and line references
- open questions or residual risks after findings
- brief change summary only at the end

Important:

- Prioritize correctness and scientific validity over style.
- Ignore `knowledge/`, local paper indexing, generated `outputs/`, and local dataset contents unless the user explicitly asks for those areas.
