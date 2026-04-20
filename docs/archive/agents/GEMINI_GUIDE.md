# Gemini Guide

Shared project context now lives in:

- [AI_AGENT_GUIDE.md](../../agents/AI_AGENT_GUIDE.md)

This file remains only as a Gemini-oriented wrapper plus prompt templates.

It supersedes the archived notes in:

- `docs/archive/agents/GEMINI_BOOTSTRAP.md`
- `docs/archive/agents/GEMINI_PLAYBOOK.md`
- `docs/archive/agents/GEMINI_PROMPTS.md`

## Gemini Read Order

Ask Gemini to read:

1. [docs/agents/AI_AGENT_GUIDE.md](../../agents/AI_AGENT_GUIDE.md)
2. one task-specific doc if needed

## Bootstrap Template

```text
Read first:
1. docs/agents/AI_AGENT_GUIDE.md

Then do:
- <specific task>

Do not:
- treat train-side ranking alone as final evidence
- ignore summary_by_regime.csv or stat_fidelity_by_regime.csv
```

## Run-Analysis Template

```text
Analyze this run:
- outputs/exp_YYYYMMDD_HHMMSS

Use:
- tables/protocol_leaderboard.csv
- tables/summary_by_regime.csv
- tables/residual_signature_by_regime.csv
- tables/stat_fidelity_by_regime.csv
- train/tables/gridsearch_results.csv

Answer:
1. did the run complete?
2. which candidate won?
3. which regimes passed or failed?
4. what changed vs <reference run>?
5. what single next test is best?
```
