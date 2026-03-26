# Claude Code Instructions

This file stays at repo root because Claude-style tooling commonly looks for
`CLAUDE.md` automatically.

## Shared Project Context

Read the common guide first:

- [docs/agents/AI_AGENT_GUIDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/AI_AGENT_GUIDE.md)
- [docs/active/WORKING_STATE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/WORKING_STATE.md)

Then read:

- [docs/agents/REVIEW.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/REVIEW.md)

## Claude-Specific Start

- run `/memory` and confirm project memory and rules are loaded
- this project is configured to start in Plan Mode via `.claude/settings.json`
- use `/rename <task-name>` early so the session is easy to resume later

Useful project skills:

- `/seq-bigru-kickoff [phase-or-focus]`
- `/seq-review [scope]`

## Claude-Specific Hygiene

- use `/compact` during long sessions to preserve key decisions
- use `/clear` between unrelated tasks
- for parallel investigations, prefer separate worktrees

## Ignore Unless Asked

- `knowledge/`
- `scripts/index_knowledge_chroma.py`
- `.gitignore` changes related to local paper/index tooling

## Out Of Scope

- do not edit `knowledge/`, `data/`, or generated `outputs/` unless explicitly asked
- do not spend time extending local paper-ingestion or retrieval tooling in this branch
