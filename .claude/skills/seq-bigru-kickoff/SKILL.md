---
name: seq-bigru-kickoff
description: Start or resume the seq_bigru_residual project with the correct reading order and constraints
argument-hint: [phase-or-focus]
context: fork
agent: Plan
disable-model-invocation: true
---

You are starting or resuming work on the `seq_bigru_residual` project.

Requested phase or focus:

$ARGUMENTS

Follow this workflow:

1. If interactive, run `/memory` and confirm project memory is loaded.
2. Read `docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md`.
3. Read the files listed in Section 2 of that plan in the exact order given there.
4. Summarize:
   - current architecture surface
   - protocol invariants
   - exact write scope
   - the smallest next safe change
5. If the user asked for implementation, start with the smallest change that unlocks the requested phase.
6. Verify with the lightest relevant command from the plan before moving on.

Non-negotiable:

- Preserve `concat` and `channel_residual`.
- Do not touch `knowledge/`, `data/`, or generated `outputs/` unless the user explicitly asks.
- Prefer phase-by-phase progress over broad rewrites.
