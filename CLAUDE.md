# Claude Code Instructions

## Scope

- Active architecture project: `feat/seq-bigru-residual-cvae`
- Main implementation handoff: `docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md`
- Review-specific criteria: `REVIEW.md`

Ignore these local changes unless the user explicitly asks for them:

- `knowledge/`
- `scripts/index_knowledge_chroma.py`
- `.gitignore` changes related to local paper/index tooling

## Start Of Session

- Run `/memory` and confirm project memory and rules are loaded.
- This project is configured to start in Plan Mode via `.claude/settings.json`.
- Before any non-trivial code change, read `docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md`.
- Use `/rename <task-name>` early so the session is easy to resume later.

Useful project skills:

- `/seq-bigru-kickoff [phase-or-focus]`
- `/seq-review [scope]`

## Core Constraints

- Preserve existing `concat` and `channel_residual` behavior.
- The new target architecture is `seq_bigru_residual`.
- Split remains per experiment, temporal `head=train`, `tail=val`.
- Windowing must happen after split.
- Windows must never cross experiment boundaries or train/val boundaries.
- Sequence inference must preserve one output per original sample.
- Deterministic inference must still support `EVM/SNR`.
- Stochastic `mc_concat` must still support distribution metrics.
- Keep saved-model layer names compatible: `encoder`, `prior_net`, `decoder`.

## Verification

- Success is scientific protocol improvement, not lower loss alone.
- Prefer targeted verification after each phase instead of large unverified rewrites.
- Default quick verification commands:
  - `python -m pytest tests -q`
  - `python -m src.training.train --dataset_root data/dataset_fullsquare_organized --output_base outputs --run_id seq_bigru_residual_smoke --grid_preset seq_residual_smoke --max_grids 1 --max_experiments 1 --max_samples_per_exp 200000 --max_epochs 2 --keras_verbose 2`
  - `python -m src.protocol.run --dataset_root data/dataset_fullsquare_organized --output_base outputs --protocol configs/one_regime_1p0m_300mA.json --train_once_eval_all --grid_preset seq_residual_smoke --max_grids 1 --max_epochs 2 --stat_tests --stat_mode quick`

## Session Hygiene

- Use `/compact` during long sessions to preserve the key decisions and reduce context noise.
- Use `/clear` between unrelated tasks.
- Use `/rewind` or checkpoints if the session drifts.
- For parallel investigations, prefer separate worktrees rather than mixing tasks in one session.

## Out Of Scope

- Do not edit `knowledge/`, `data/`, or generated `outputs/` unless the user explicitly asks.
- Do not spend time extending the local paper-ingestion or retrieval tooling in this branch.
