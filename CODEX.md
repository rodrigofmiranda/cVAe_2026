# Codex Session Handoff

## Scope

- Active branch: `feat/seq-bigru-residual-cvae`
- Current focus: advance the digital twin with `seq_bigru_residual` and
  `delta_residual` inside the protocol-first workflow
- Main notes:
  - `docs/ACTIVE_CONTEXT.md`
  - `PROJECT_STATUS.md`
  - `docs/DELTA_RESIDUAL_STATUS.md`
  - `docs/FUTURE_ADVERSARIAL_STRATEGY.md`
  - `docs/PROTOCOL.md`
  - `docs/RUN_REANALYSIS_PLAYBOOK.md`
  - `TRAINING_PLAN.md`

## Current Checkpoint

- active architectures in this worktree:
  - `seq_bigru_residual`
  - `delta_residual`
  - legacy support variants
- the adversarial line was intentionally removed from the active code path
- if that strategy is revisited, use `docs/FUTURE_ADVERSARIAL_STRATEGY.md`
  and the historical branch name `feat/delta-residual-adv`
- the branch now also includes the recent protocol infrastructure:
  - reduced 12-regime protocol: `configs/all_regimes_sel4curr.json`
  - model reuse: `--reuse_model_run_dir`
  - operational training dashboard:
    - `train/tables/grid_training_diagnostics.csv`
    - `train/plots/training/dashboard_analysis_complete.png`
  - tmux-managed container helpers:
    - `scripts/run_tf25_gpu.sh`
    - `scripts/enter_tf25_gpu.sh`
    - `scripts/stop_tf25_gpu.sh`

## Scientific State

- strongest current temporal reference:
  - run id: `exp_20260322_193738`
  - winner: `S4seq_W7_h64_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
  - summary:
    - `6/12` total passes
    - `0/4` at `0.8 m`
    - `2/4` at `1.0 m`
    - `4/4` at `1.5 m`
    - best current multi-regime scientific reference in this workspace
- latest comparison run that did not overtake the reference:
  - run id: `exp_20260323_210309`
  - winner: `S4seq_W7_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
  - reading:
    - more seq capacity did not improve the final protocol result
- historical all-gates-passed seq reference:
  - run id: `exp_20260318_204149`
  - tag: `S2seq_W7_h64_lat4_b0p001_lmmd1p0_fb0p10_lr0p0003_L128-256-512`
- best current point-wise non-adversarial scientific reference:
  - run id: `exp_20260318_235319`
  - tag: `D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`
- strongest current point-wise anchor carried into protocol-first comparisons:
  - `COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256`
- current recommended overnight preset:
  - `seq_overnight_12h`
  - target: roughly `10–12 h`
  - command:
    - `python -m src.protocol.run --dataset_root data/dataset_fullsquare_organized --output_base outputs --protocol configs/all_regimes_sel4curr.json --train_once_eval_all --grid_preset seq_overnight_12h --max_epochs 120 --patience 12 --reduce_lr_patience 6 --stat_tests --stat_mode full --stat_max_n 5000 --no_data_reduction`

## Non-Negotiable Invariants

- preserve `concat`, `channel_residual`, `delta_residual`,
  `seq_bigru_residual`, and `legacy_2025_zero_y`
- split remains per experiment, temporal `head=train`, `tail=val`
- pipeline order remains `split -> cap/reduce(train only) -> train`
- no condition-label misalignment after post-split reduction
- saved-model layer names must remain compatible: `encoder`, `prior_net`, `decoder`
- serious experimentation should happen through `src.protocol.run`, not `src.training.train`
- `seq_bigru_residual` requires `--no_data_reduction`
- `--reuse_model_run_dir` is the preferred way to re-evaluate a trained shared model without retraining

## Output Hygiene

- target layout:
  - `train/`
  - `eval/`
  - `logs/`
  - `tables/`
  - `plots/best_model/`
- champion-only visual artifacts:
  - `train/plots/champion/analysis_dashboard.png`
  - `plots/best_model/heatmap_gate_metrics_by_regime.png`
- experiment-wide operational artifacts:
  - `train/tables/grid_training_diagnostics.csv`
  - `train/plots/training/dashboard_analysis_complete.png`

## Start Of Session

On a fresh Codex session, begin with:

1. read `CODEX.md`
2. read `docs/ACTIVE_CONTEXT.md`
3. read `PROJECT_STATUS.md`
4. read `docs/PROTOCOL.md`
5. read `docs/FUTURE_ADVERSARIAL_STRATEGY.md` only if the task is about reviving the archived GAN line
6. inspect `git status --short`
7. if a run is active, inspect `pgrep -af "src.protocol.run|src.training.train"`
8. run `python scripts/summarize_experiment.py` on the latest `outputs/exp_*`

## Suggested Resume Prompt

Use this at session restart:

`Leia CODEX.md, docs/ACTIVE_CONTEXT.md, PROJECT_STATUS.md e docs/DELTA_RESIDUAL_STATUS.md. Rode python scripts/summarize_experiment.py no último exp_*. Resuma o estado atual da branch feat/seq-bigru-residual-cvae, diga qual é a melhor referência seq e acompanhe apenas o experimento em andamento sem mudar a configuração.`

## Verification Focus

Prefer verifying in this order:

1. `--reuse_model_run_dir` still skips retrain and resolves `models/best_model_full.keras`
2. reduced 12-regime protocol remains correctly selected
3. inspect:
   - `train/tables/grid_training_diagnostics.csv`
   - `train/plots/training/dashboard_analysis_complete.png`
   - `plots/best_model/heatmap_gate_metrics_by_regime.png`

## Out Of Scope

- do not use all 9 currents by default
- do not move `release/cvae-online` based on the adversarial branch yet
