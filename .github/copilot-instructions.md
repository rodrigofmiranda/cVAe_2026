# Copilot Workspace Instructions – `/workspace/2026`

## Big picture
- This is a PhD-level research repository for a VLC channel **digital twin** centered on a conditional VAE workflow.
- The current production flow is script-driven and mostly monolithic:
  - Training + grid search: `src/training/cvae_TRAIN_documented.py`
  - Run re-analysis/evaluation: `src/evaluation/analise_cvae_reviewed.py`
  - Dataset capture (GNU Radio/UHD): `src/data/channel_dataset.py`
- `REFACTOR_PLAN.md` defines the target modular architecture. Implement changes against **current runtime behavior** unless explicitly asked to change algorithms.

## Core objective
Model the conditional distribution (digital twin of the physical channel):

    p(y | x, d, c)

Where:
- `x`: transmitted complex I/Q baseband signal
- `y`: received signal after VLC channel
- `d`: distance
- `c`: LED current (or regime identifier)

The goal is **distributional fidelity** (not only mean mapping), preserving VLC nonlinearities and noise characteristics learned from synchronized experimental data.

## Critical constraints (compatibility-first)
- **Do NOT change algorithmic behavior** unless explicitly requested.
- Focus on engineering structure, modularization, readability, and reproducibility.
- Preserve paths and environment variables:
  - `DATASET_ROOT` → dataset path (default points to `data/dataset_fullsquare_organized`)
  - `OUTPUT_BASE` → `/workspace/2026/outputs`
  - `RUN_ID` optional
- Preserve run directory layout:
  - `outputs/run_YYYYMMDD_HHMMSS/{models,logs,plots,tables,state_run.json}`
  - Keep `outputs/_last_run.txt` behavior.
- Preserve output file names and JSON keys unless explicitly requested to change.
- Keep dependencies minimal; avoid heavy new frameworks unless strictly necessary.

## Core data/artifact flow
- Input dataset under `data/dataset_fullsquare_organized/dist_*/curr_*/.../IQ_data/`.
- Required arrays per experiment:
  - `sent_data_tuple.npy`
  - one accepted received filename (e.g., `received_data_tuple_sync-phase.npy`)
- Training writes artifacts to `outputs/run_*/{models,logs,plots,tables}`.
- Compatibility contract: keep `state_run.json` keys stable:
  - `normalization`, `data_split`, `training_config`, `artifacts`, `eval_protocol`
- Evaluation reads `state_run.json` when available and must remain backward-compatible with fallback behavior when keys/files are missing.

## Developer workflows (actual commands)
- Preferred wrappers:
  - `bash scripts/train.sh`
  - `bash scripts/eval.sh`
- Direct entrypoints used today:
  - `python -u src/training/cvae_TRAIN_documented.py`
  - `python -u src/evaluation/analise_cvae_reviewed.py`
  - `python src/evaluation/non_gaussianity_by_regime.py --dataset_root <path>`

## Project-specific conventions to preserve
- Split policy is intentionally per-experiment (`head_tail`), not global shuffle (reduce temporal leakage).
- Normalization is computed from train split and stored in `state_run.json`; evaluation should reuse it.
- Model artifact names are part of downstream tooling:
  - `best_model_full.keras`, `best_decoder.keras`, `best_prior_net.keras`, `training_history.json`
- Keras deserialization depends on custom objects in evaluation (e.g., `Sampling`, `CondPriorVAELoss`);
  preserve class/function **names and signatures** when editing model code.
- Keep run outputs and table/plot filenames stable unless a task explicitly requests changes.

## Refactoring strategy (minimize risk)
Refactor incrementally with small, reviewable commits:
1. Scaffolding only: new folders/modules + CLI entrypoints + wrappers (no logic moved)
2. Move IO/run_dir/state_run writing
3. Move data loading/splits/normalization
4. Move model/loss/callbacks
5. Move evaluation metrics/plots/reporting
Each step must preserve behavior and outputs.

## Quality standards for changes
- Add docstrings and type hints for new modules/functions.
- Avoid overengineering (no microservices / unnecessary abstraction).
- Prefer pragmatic research-grade architecture suitable for publications and reproducibility.
- If an I/O contract must change, update training and evaluation in the same change and document it.

## Validation expectations
- Always run the smallest relevant validation first:
  - Prefer `scripts/eval.sh` on an existing run after refactors.
- Ensure training still produces the same run folder structure and key artifacts.
- Ensure evaluation works on older runs with partial/missing `state_run.json` keys.