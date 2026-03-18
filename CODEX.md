# Codex Session Handoff

## Scope

- Active branch: `feat/seq-bigru-residual-cvae`
- Current focus: scientific evaluation of `arch_variant="legacy_2025_zero_y"`
- Main implementation/research notes:
  - `docs/LEGACY_2025_ZERO_Y_RUNBOOK.md`
  - `docs/LEGACY_2025_ZERO_Y_STATUS.md`
  - `docs/LEGACY_2025_BATCHSIZE_PROTOCOL.md`
  - `docs/LEGACY_2025_BATCHSIZE_RESULTS.md`
- Review constraints: `REVIEW.md`

## Current Checkpoint

- `legacy_2025_zero_y` is implemented and integrated in the 2026 pipeline
- reduced-data pivot benchmark after the D/C alignment fix:
  - run: `/workspace/2026/outputs/exp_20260318_193036`
  - valid reference run for the legacy port
- batch-size sweep completed on the reduced 4-current protocol:
  - `bs4096` reference: `/workspace/2026/outputs/exp_20260318_193036`
  - `bs8192` accepted: `/workspace/2026/outputs/exp_20260318_195010`
  - `bs16384` rejected: `/workspace/2026/outputs/exp_20260318_195709`
- current operational batch size for legacy runs: `8192`
- current scientific screening in progress:
  - run: `/workspace/2026/outputs/exp_20260318_204208`
  - mode: `train_once_eval_all`
  - protocol: `configs/one_regime_1p0m_300mA_sel4curr.json`
  - preset: `legacy2025_large`
  - search width: `12` grids

## Non-Negotiable Invariants

- preserve existing `concat`, `channel_residual`, and `seq_bigru_residual`
- split remains per experiment, temporal `head=train`, `tail=val`
- pipeline order remains `split -> cap/reduce(train only) -> train`
- no condition-label misalignment after post-split reduction
- saved-model layer names must remain compatible: `encoder`, `prior_net`, `decoder`
- `legacy_2025_zero_y` keeps standard-normal latent semantics:
  - no learned conditional prior in the loss
  - `free_bits` must stay incompatible with this variant

## Valid Reference Outputs

Prefer these runs when comparing or resuming:

- `/workspace/2026/outputs/exp_20260318_141109`
- `/workspace/2026/outputs/exp_20260318_193036`
- `/workspace/2026/outputs/exp_20260318_195010`
- `/workspace/2026/outputs/exp_20260318_195709`

Do not use this run as scientific reference:

- `/workspace/2026/outputs/exp_20260318_190337`
  - invalid due to the pre-fix D/C reduction misalignment

## Output Hygiene

- `outputs/` was cleaned conservatively on 2026-03-18
- smoke runs, dry-runs, exploratory leftovers, many incomplete `exp_*`, and all legacy `run_*`
  directories were removed
- current output footprint was reduced from about `3.9G` to about `1.8G`
- plot bundles are now grouped by purpose in new runs:
  - `reports/`
  - `core/`
  - `distribution/`
  - `latent/`
  - `training/`
  - `legacy/` for champion-only executive plots

## Start Of Session

On a fresh Codex session, begin with:

1. read `CODEX.md`
2. read `REVIEW.md`
3. read `docs/LEGACY_2025_ZERO_Y_STATUS.md`
4. inspect `git status --short`
5. if a run is active, inspect `pgrep -af "src.protocol.run|src.training.train"`
6. inspect the latest `outputs/exp_*/global_model/tables/gridsearch_plan.xlsx`

## Suggested Resume Prompt

Use this at session restart:

`Leia CODEX.md, REVIEW.md e docs/LEGACY_2025_ZERO_Y_STATUS.md. Resuma o estado atual da branch feat/seq-bigru-residual-cvae, diga quais runs validos devemos usar como referencia e acompanhe apenas o experimento em andamento sem mudar a configuracao.`

## Verification Focus

Prefer verifying in this order:

1. reduced 4-current protocol remains correctly selected
2. normalization uses all intended distances/currents after reduction
3. batch size stays at the accepted ceiling (`8192`) for wider sweeps
4. compare candidate grids by:
   - `best_val_loss`
   - `cvae_delta_evm_%`
   - `cvae_delta_mean_l2`
   - `cvae_psd_l2`
   - `stat_mmd2`
5. inspect `plots/README.txt` and `plots/core/overlay_constellation.png` first

## Out Of Scope

- do not treat `knowledge/`, `data/`, or generated `outputs/` as implementation targets unless the user explicitly asks
- do not revive deleted smoke/exploratory runs unless the user explicitly wants them regenerated
- do not use all 9 currents for legacy grid sweeps by default; prefer the reduced 4-current protocol first
