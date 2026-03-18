# Codex Session Handoff

## Scope

- Active branch: `feat/seq-bigru-residual-cvae`
- Main architecture handoff: `docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md`
- Review constraints: `REVIEW.md`
- Codex role in this branch: observer / reviewer / test guide unless the user explicitly asks Codex to edit files
- Claude is currently the primary implementation agent

Ignore these local changes unless the user explicitly asks for them:

- `knowledge/`
- `scripts/index_knowledge_chroma.py`
- `.gitignore` changes related to local paper/index tooling

## Current Checkpoint

- Architecture in progress: `seq_bigru_residual`
- User recollection: work likely passed Phase 6 of the plan
- Repository state confirms local work exists for Phases 4-8 scope:
  - sequence model: `src/models/cvae_sequence.py`
  - windowing: `src/data/windowing.py`
  - training integration: `src/training/pipeline.py`, `src/training/gridsearch.py`
  - inference/protocol integration: `src/evaluation/engine.py`, `src/protocol/run.py`
  - grid integration: `src/training/grid_plan.py`
  - tests added: `tests/test_windowing.py`, `tests/test_seq_cvae_build.py`, `tests/test_seq_engine.py`, `tests/test_seq_serialization.py`, `tests/test_pipeline_seq.py`
- Exact completion status of each phase is not yet verified in-session; treat “passed Phase 6” as a working assumption until tests confirm it

## Non-Negotiable Invariants

- Preserve existing `concat` and `channel_residual` behavior
- Split remains per experiment, temporal `head=train`, `tail=val`
- Windowing must happen after split
- Windows must never cross experiment boundaries or train/val boundaries
- Sequence path must preserve one output per original sample
- Deterministic inference must still support `EVM/SNR`
- Stochastic `mc_concat` must still support distribution metrics
- Saved-model layer names must remain compatible: `encoder`, `prior_net`, `decoder`

## Start Of Session

On a fresh Codex session, begin with:

1. read `CODEX.md`
2. read `REVIEW.md`
3. read `docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md`
4. inspect current `git status --short`
5. inspect changed files before making any claim about the active phase

If the user wants Codex only as reviewer/observer:

- do not make code edits unless explicitly asked
- prioritize:
  - checking phase completeness
  - identifying regressions or invariant violations
  - proposing next verification commands
  - summarizing what remains before the next phase

## Suggested Resume Prompt

Use this at session restart:

`Leia CODEX.md, REVIEW.md e docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md. Atue como observador/revisor apenas, sem editar nada até resumir o estado atual da branch feat/seq-bigru-residual-cvae, dizer em que fase estamos e quais testes devem rodar a seguir.`

## Verification Focus

Prefer verifying in this order:

1. windowing invariants
2. sequence model build / serialization
3. inference and protocol compatibility
4. grid exposure and smoke-run wiring

Useful commands when the user asks for validation:

- `python -m pytest tests/test_windowing.py tests/test_seq_cvae_build.py -q`
- `python -m pytest tests/test_pipeline_seq.py tests/test_seq_engine.py tests/test_seq_serialization.py -q`
- `python -m src.training.train --dataset_root data/dataset_fullsquare_organized --output_base outputs --run_id seq_bigru_residual_smoke --grid_preset seq_residual_smoke --max_grids 1 --max_experiments 1 --max_samples_per_exp 200000 --max_epochs 2 --keras_verbose 2`
- `python -m src.protocol.run --dataset_root data/dataset_fullsquare_organized --output_base outputs --protocol configs/one_regime_1p0m_300mA.json --train_once_eval_all --grid_preset seq_residual_smoke --max_grids 1 --max_epochs 2 --stat_tests --stat_mode quick`

## Out Of Scope

- Do not treat `knowledge/`, `data/`, or generated `outputs/` as implementation targets unless the user explicitly asks
- Do not spend time on local paper-ingestion or retrieval tooling in this branch
