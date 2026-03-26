# SESSION_STATE (refactor_architecture)

## Objective
Refactor the VLC cVAE research repo into a modular, reproducible structure **without changing algorithms**.
Keep existing outputs, directory names, JSON/state formats, and training/eval behavior.

## Branch
exp/refactor_architecture

## Recent commits (high-level)
- Commit 2: extracted run bootstrap to `src/training/logging.py` (`bootstrap_run()` creates run dirs + writes `_last_run.txt`).
- Commit 3A: moved low-level helpers to `src/data/loading.py` (`ensure_iq_shape`, `read_metadata`, `parse_dist_curr_from_path`).
- Commit 3B: consolidated dataset discovery/loading in `src/data/loading.py`:
  `discover_experiments`, `is_valid_dataset_root`, `find_dataset_root(dataset_root_hint=...)`,
  `reduce_experiment_xy(reduction_config)`, `load_experiments_as_list(reduction_config=None|dict)`.

## Runner scripts (important)
To avoid `ModuleNotFoundError: src`, scripts now:
- `cd $REPO_ROOT`
- `export PYTHONPATH="$REPO_ROOT"`
- run via `python -m ...`
Files: `scripts/train.sh`, `scripts/eval.sh`

## Invariants (must not break)
- OUTPUT tree: `outputs/run_YYYYmmdd_HHMMSS/{models,logs,plots,tables,state_run.json}`
- Env vars: `DATASET_ROOT`, `OUTPUT_BASE`, optional `RUN_ID`
- Split policy: **per-experiment head/tail** remains unchanged.
- No changes to model, loss, metrics, grid-search plan.

## Context window note (LLM)
When the chat “Context Window” gets near limit, prioritize:
(1) invariants above, (2) commit log summary, (3) current failing error trace.
Do NOT reintroduce removed duplicate functions; import from `src/data/loading.py`.

## Next step
Smoke tests:
- `bash scripts/train.sh` (can stop after first epoch starts)
- `bash scripts/eval.sh`
Then proceed to next refactor commit only after tests are green.
