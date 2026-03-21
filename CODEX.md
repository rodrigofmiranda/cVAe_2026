# Codex Session Handoff

## Scope

- Active branch: `feat/delta-residual-adv`
- Current focus: align the experimental point-wise cVAE-GAN line with the
  current protocol-first workflow and compare it fairly against the strongest
  non-adversarial references
- Main notes:
  - `docs/ACTIVE_CONTEXT.md`
  - `PROJECT_STATUS.md`
  - `docs/DELTA_RESIDUAL_ADV_STATUS.md`
  - `docs/DELTA_RESIDUAL_STATUS.md`
  - `docs/PROTOCOL.md`
  - `docs/RUN_REANALYSIS_PLAYBOOK.md`
  - `TRAINING_PLAN.md`

## Current Checkpoint

- `delta_residual_adv` exists in the main pipeline
  - implementation commit: `8c2bd41`
- adversarial training was corrected in:
  - commit: `ee2681f`
  - critic now sees sampled residuals instead of only residual means
  - MC ranking no longer uses the optimistic pre-fix mean-of-samples shortcut
  - adversarial save/load now preserves the full wrapper path
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
  - run: `/workspace/2026/outputs/exp_20260320_171510`
  - winner: `S2seq_W7_h64_lat4_b0p003_lmmd1p0_fb0p10_lr0p0003_L128-256-512`
  - summary:
    - passes all regimes at `1.5 m`
    - fails all regimes at `0.8 m`
    - best current multi-regime scientific reference in this workspace
- historical all-gates-passed seq reference:
  - run: `/workspace/2026/outputs/exp_20260318_204149`
  - tag: `S2seq_W7_h64_lat4_b0p001_lmmd1p0_fb0p10_lr0p0003_L128-256-512`
- best current point-wise non-adversarial scientific reference:
  - run: `/workspace/2026/outputs/exp_20260318_235319`
  - tag: `D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`
- strongest current point-wise anchor carried into protocol-first comparisons:
  - `COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256`

## Important Caveat For The GAN Line

Do not treat these adversarial runs as final scientific references:

- `/workspace/2026/outputs/exp_20260320_012223`
- `/workspace/2026/outputs/exp_20260320_014614`
- `/workspace/2026/outputs/exp_20260320_020652`

Reason:

- they were produced before the March 20 adversarial fixes
- the original adversarial critic was not pressuring the stochastic residual head correctly
- the comparison path used stale MC ranking logic

These runs are useful only as historical debugging evidence.

## Non-Negotiable Invariants

- preserve `concat`, `channel_residual`, `delta_residual`, `delta_residual_adv`,
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
5. read `docs/DELTA_RESIDUAL_ADV_STATUS.md`
6. inspect `git status --short`
7. if a run is active, inspect `pgrep -af "src.protocol.run|src.training.train"`
8. run `python scripts/summarize_experiment.py` on the latest `outputs/exp_*`

## Suggested Resume Prompt

Use this at session restart:

`Leia CODEX.md, docs/ACTIVE_CONTEXT.md, PROJECT_STATUS.md e docs/DELTA_RESIDUAL_ADV_STATUS.md. Rode python scripts/summarize_experiment.py no último exp_*. Resuma o estado atual da branch feat/delta-residual-adv, diga quais runs adversariais estão obsoletos, qual é a melhor referência seq e acompanhe apenas o experimento em andamento sem mudar a configuração.`

## Verification Focus

Prefer verifying in this order:

1. adversarial critic still consumes sampled residuals
2. MC ranking in grid search uses per-draw metrics rather than the optimistic mean shortcut
3. `--reuse_model_run_dir` still skips retrain and resolves `models/best_model_full.keras`
4. reduced 12-regime protocol remains correctly selected
5. inspect:
   - `train/tables/grid_training_diagnostics.csv`
   - `train/plots/training/dashboard_analysis_complete.png`
   - `plots/best_model/heatmap_gate_metrics_by_regime.png`

## Out Of Scope

- do not treat the pre-fix March 20 adversarial runs as scientific references
- do not use all 9 currents by default
- do not move `release/cvae-online` based on the adversarial branch yet
