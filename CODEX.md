# Codex Session Handoff

## Scope

- Active branch: `feat/seq-bigru-residual-cvae`
- Current focus: scientific evaluation of `arch_variant="delta_residual"`
- Main notes:
  - `docs/DELTA_RESIDUAL_STATUS.md`
  - `docs/LEGACY_2025_ZERO_Y_STATUS.md`
  - `docs/LEGACY_2025_ZERO_Y_RUNBOOK.md`
  - `REVIEW.md`

## Current Checkpoint

- `delta_residual` is implemented and integrated in the 2026 pipeline
  - commit: `c32ee63`
- current reduced 4-current protocol:
  - `configs/one_regime_1p0m_300mA_sel4curr.json`
  - currents: `100/300/500/700 mA`
  - distances: `0.8/1.0/1.5 m`
- first operational smoke:
  - run: `/workspace/2026/outputs/exp_20260318_230955`
  - outcome: variant works end-to-end, but poor scientific fidelity
- best current `delta_residual` candidate:
  - run: `/workspace/2026/outputs/exp_20260318_235319`
  - winner: `D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`
  - pivot summary: `ΔEVM=-3.678 pp`, `ΔSNR=+1.113 dB`, `Δmean_l2=0.0090`, `PSD_L2=0.2399`, `MMD²=0.001129`, `Energy=0.000787`
- local refinement around the winner:
  - preset commit: `fb7fbd2`
  - run: `/workspace/2026/outputs/exp_20260318_233023`
  - winner by grid score: `D2delta_lat4_b0p0005_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`
  - scientific outcome: worse than `/workspace/2026/outputs/exp_20260318_231458`
- local latent/batch refinement around the scientific winner:
  - preset commit: `05e2774`
  - run: `/workspace/2026/outputs/exp_20260318_235319`
  - winner by grid score: `D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`
  - scientific outcome: slightly better overall than `/workspace/2026/outputs/exp_20260318_231458`

## Main Lessons So Far

- explicit residual target is more promising than `legacy_2025_zero_y`
- the best region so far is:
  - `beta=0.001`
  - `free_bits=0.0`
  - `layer_sizes=[128,256,512]`
  - `latent_dim` in the small range `4–5`
- `batch_size=16384` remains competitive in the best current run, but `8192` still looked healthier in several intermediate sweeps
- lowering `beta` to `0.0005` improved grid rank metrics but hurt the final pivot regime:
  - worse `ΔEVM`
  - much worse `PSD_L2`
  - much worse `MMD²` and `Energy`
- current grid ranking is not perfectly aligned with final twin-quality metrics

## Non-Negotiable Invariants

- preserve existing `concat`, `channel_residual`, `seq_bigru_residual`, and `legacy_2025_zero_y`
- split remains per experiment, temporal `head=train`, `tail=val`
- pipeline order remains `split -> cap/reduce(train only) -> train`
- no condition-label misalignment after post-split reduction
- saved-model layer names must remain compatible: `encoder`, `prior_net`, `decoder`

## Valid Reference Outputs

Prefer these runs when comparing or resuming:

- `/workspace/2026/outputs/exp_20260318_235319`
- `/workspace/2026/outputs/exp_20260318_231458`
- `/workspace/2026/outputs/exp_20260318_233023`
- `/workspace/2026/outputs/exp_20260318_193036`
- `/workspace/2026/outputs/exp_20260318_195010`

Do not use this run as scientific reference:

- `/workspace/2026/outputs/exp_20260318_190337`
  - invalid due to the pre-fix D/C reduction misalignment

## Output Hygiene

- `outputs/` was cleaned on 2026-03-18
- new runs group plots by purpose:
  - `reports/`
  - `core/`
  - `distribution/`
  - `latent/`
  - `training/`
  - `legacy/`

## Start Of Session

On a fresh Codex session, begin with:

1. read `CODEX.md`
2. read `REVIEW.md`
3. read `docs/DELTA_RESIDUAL_STATUS.md`
4. inspect `git status --short`
5. if a run is active, inspect `pgrep -af "src.protocol.run|src.training.train"`
6. inspect the latest `outputs/exp_*/global_model/tables/gridsearch_results.xlsx`

## Suggested Resume Prompt

Use this at session restart:

`Leia CODEX.md, REVIEW.md e docs/DELTA_RESIDUAL_STATUS.md. Resuma o estado atual da branch feat/seq-bigru-residual-cvae, diga qual run delta_residual e a melhor referencia cientifica atual e acompanhe apenas o experimento em andamento sem mudar a configuracao.`

## Verification Focus

Prefer verifying in this order:

1. reduced 4-current protocol remains correctly selected
2. normalization uses all intended distances/currents after reduction
3. compare `delta_residual` candidates by:
   - `ΔEVM`
   - `ΔSNR`
   - `cvae_delta_mean_l2`
   - `cvae_psd_l2`
   - `stat_mmd2`
   - `stat_energy`
4. inspect `plots/README.txt` and `plots/core/overlay_constellation.png` first

## Out Of Scope

- do not use all 9 currents by default
- do not treat old legacy smoke runs as release candidates
- do not move `release/cvae-online` yet based on `delta_residual`
