# Codex Session Handoff

## Scope

- Active branch: `feat/seq-bigru-residual-cvae`
- Current focus: protocol-first mixed-family comparison
- Main notes:
  - `PROJECT_STATUS.md`
  - `docs/DELTA_RESIDUAL_STATUS.md`
  - `docs/PROTOCOL.md`
  - `TRAINING_PLAN.md`
  - `REVIEW.md`

## Current Checkpoint

- `delta_residual` is implemented and integrated in the 2026 pipeline
  - commit: `c32ee63`
- `seq_bigru_residual` now has a short-window non-cuDNN fallback for newer GPUs
  - commit: `e1807d2`
- protocol-selected experiment filters are now portable across clone roots
  - commit: `5428c9d`
- current reduced 4-current protocol:
  - `configs/one_regime_1p0m_300mA_sel4curr.json`
  - currents: `100/300/500/700 mA`
  - distances: `0.8/1.0/1.5 m`
- best current `delta_residual` scientific reference:
  - run: `/workspace/2026/outputs/exp_20260318_235319`
  - winner: `D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`
  - pivot summary: `ΔEVM=-3.678 pp`, `ΔSNR=+1.113 dB`, `Δmean_l2=0.0090`, `PSD_L2=0.2399`, `MMD²=0.001129`, `Energy=0.000787`
- all-gates-passed seq reference:
  - run: `/workspace/2026/outputs/exp_20260318_204149`
  - tag: `S2seq_W7_h64_lat4_b0p001_lmmd1p0_fb0p10_lr0p0003_L128-256-512`
  - source: `eval_final_model.py` / `eval_final_gates.json`
- current protocol-first mixed-family preset:
  - preset: `best_compare_large`
  - commit: `b9f53d2`
  - compares:
    - strongest current point-wise `delta_residual` candidates
    - strongest current `seq_bigru_residual` candidates, including `lambda_mmd`

## Main Lessons So Far

- explicit residual target is more promising than `legacy_2025_zero_y`
- the best point-wise region so far is:
  - `beta=0.001`
  - `free_bits=0.0`
  - `layer_sizes=[128,256,512]` or nearby compact MLPs from the capacity sweep
  - `latent_dim` in the small range `4–6`
- the seq line remains scientifically important because it is still the only
  historical reference with all gates passing
- protocol-first comparison is now preferred over training-only ranking
- on newer GPUs like the RTX 5090, the seq line must use the non-cuDNN GRU path

## Non-Negotiable Invariants

- preserve existing `concat`, `channel_residual`, `seq_bigru_residual`, and `legacy_2025_zero_y`
- split remains per experiment, temporal `head=train`, `tail=val`
- pipeline order remains `split -> cap/reduce(train only) -> train`
- no condition-label misalignment after post-split reduction
- saved-model layer names must remain compatible: `encoder`, `prior_net`, `decoder`
- serious experimentation should happen through `src.protocol.run`, not `src.training.train`
- `seq_bigru_residual` requires `--no_data_reduction`

## Valid Reference Outputs

Prefer these runs when comparing or resuming:

- `/workspace/2026/outputs/exp_20260318_204149`
- `/workspace/2026/outputs/exp_20260318_235319`
- `/workspace/2026/outputs/exp_20260318_231458`
- `/workspace/2026/outputs/exp_20260318_233023`
- `/workspace/2026/outputs/exp_20260318_193036`
- `/workspace/2026/outputs/exp_20260318_195010`

Do not use this run as scientific reference:

- `/workspace/2026/outputs/exp_20260318_190337`
  - invalid due to the pre-fix D/C reduction misalignment

## Output Hygiene

- the target layout is now:
  - `train/`
  - `eval/`
  - `logs/`
  - `tables/`
  - `plots/best_model/`
- champion-only visual artifacts:
  - `train/plots/champion/analysis_dashboard.png`
  - `plots/best_model/heatmap_vae_vs_real_metric_diffs.png`
- old `global_model/` / `studies/` paths are legacy output layout, not the target

## Start Of Session

On a fresh Codex session, begin with:

1. read `CODEX.md`
2. read `PROJECT_STATUS.md`
3. read `docs/PROTOCOL.md`
4. read `docs/DELTA_RESIDUAL_STATUS.md`
5. inspect `git status --short`
6. if a run is active, inspect `pgrep -af "src.protocol.run|src.training.train"`
7. inspect the latest `outputs/exp_*/train/tables/gridsearch_results.xlsx`

## Suggested Resume Prompt

Use this at session restart:

`Leia CODEX.md, PROJECT_STATUS.md, docs/PROTOCOL.md e docs/DELTA_RESIDUAL_STATUS.md. Resuma o estado atual da branch feat/seq-bigru-residual-cvae, diga qual é a melhor referência point-wise, qual é a melhor referência seq com todos os gates passando e acompanhe apenas o experimento em andamento sem mudar a configuração.`

## Verification Focus

Prefer verifying in this order:

1. reduced 4-current protocol remains correctly selected
2. selected_experiments filtering remains portable across clone roots
3. seq runs on newer GPUs avoid cuDNN and use the fallback GRU path
4. compare `delta_residual` candidates by:
   - `ΔEVM`
   - `ΔSNR`
   - `cvae_delta_mean_l2`
   - `cvae_psd_l2`
   - `stat_mmd2`
   - `stat_energy`
5. inspect `train/plots/champion/analysis_dashboard.png` and `plots/best_model/heatmap_vae_vs_real_metric_diffs.png` first

## Out Of Scope

- do not use all 9 currents by default
- do not treat old legacy smoke runs as release candidates
- do not move `release/cvae-online` yet based on `delta_residual`
