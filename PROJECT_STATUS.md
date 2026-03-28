# PROJECT_STATUS

> Updated on 2026-03-28.
> This is the official inventory of worktrees, active branches, and current
> scientific status for the repository.

## Worktrees

Registered git worktrees:

1. `/workspace/2026/feat_seq_bigru_residual_cvae`
   - branch: `feat/seq-imdd-graybox-mdn`
   - role: implement and test `seq_imdd_graybox + MDN`
   - latest branch result: `outputs/exp_20260328_023302`
   - latest branch result score: `5/12`

2. `/workspace/2026/feat_seq_bigru_residual_mdn_route`
   - branch: `feat/seq-bigru-residual-mdn-route`
   - role: isolate `seq_bigru_residual + MDN` reruns and reproducibility checks
   - latest branch result: `outputs/exp_20260328_041729`
   - latest branch result score: `4/12`

## Canonical Entrypoint

- canonical experiment command:
  - `python -m src.protocol.run`

The public experiment flow remains:

- global train plus per-regime evaluation:
  - `--train_once_eval_all`
- reuse of a trained model:
  - `--reuse_model_run_dir`

## Execution Contexts

The current experiment history was produced across two machines:

1. remote `5090` host
   - long runs commonly executed through the documented `tf25_gpu` container
     path and host-side `tmux`
2. local `A6000` workstation
   - current reruns and reproducibility checks executed from the local runtime
     available in this workspace

Important provenance rule:

- not every `5090` experiment exists as a full runnable local environment here
- some `5090` runs were only copied back as experiment artifacts
- the remote `5090` host may be on a different branch from the local worktrees
  at the same time
- when comparing anchors, always treat host provenance as part of the
  experimental context, not just the git commit

Current reported remote snapshot:

- date: `2026-03-28`
- host: remote `5090`
- repo path: `~/RODRIGO/cVAe_2026`
- branch: `feat/mdn-g5-recovery`
- worktree state: clean

## Current Scientific Position

Main verified anchors:

- stable Gaussian reference:
  - run: `outputs/exp_20260324_023558`
  - result: `10/12`
- best historical MDN:
  - run: `outputs/exp_20260325_230938`
  - result: `9/12`
- previous-branch MDN benchmark:
  - run: `outputs/exp_20260327_161311`
  - result: `9/12`
- current gray-box Gaussian anchor:
  - run: `outputs/exp_20260327_172148`
  - result: `6/12`
- current gray-box + MDN branch result:
  - run: `outputs/exp_20260328_023302`
  - result: `5/12`
- current dedicated seq MDN rerun:
  - run: `outputs/exp_20260328_041729`
  - result: `4/12`

Current reading:

- `seq_imdd_graybox + MDN` is now implemented and validated end to end.
- That route did not beat the best gray-box Gaussian anchor.
- The dedicated `seq_bigru_residual + MDN` rerun did not reproduce the older
  `9/12` MDN anchors.
- The immediate problem is now reproducibility drift, not lack of architecture
  options.
- Part of that drift is operational:
  - some strong anchors came from the remote `5090` lane
  - current investigations are being executed on the local `A6000` lane
  - only a subset of remote experiments was imported locally
  - the remote lane may also be on another branch in parallel, so branch
    provenance must be checked explicitly
- The unresolved scientific region remains concentrated around `0.8 m`,
  especially G5 and G6 at lower currents.

## Tested Route Summary

Use [docs/active/ROUTES_AND_RESULTS.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/ROUTES_AND_RESULTS.md)
as the detailed matrix for:

- tested route families
- branch purpose and role
- recent benchmark runs
- promoted vs negative lines
- current queue and do-not-reopen list

## Model Families Available

- `concat`
  - original point-wise cVAE
- `channel_residual`
  - point-wise residual decoder
- `delta_residual`
  - point-wise residual-target decoder
- `seq_bigru_residual`
  - main temporal family
- `seq_imdd_graybox`
  - IM/DD gray-box temporal family
  - now supports Gaussian and MDN decoders
- `legacy_2025_zero_y`
  - controlled historical comparison

## Operational Attention Points

- For sequence runs with `max_samples_per_exp` and/or
  `max_val_samples_per_exp`, the `df_split` counts must stay synchronized with
  the post-cap arrays before windowing.
- If the evaluation environment lacks `matplotlib`, plotting can fail even
  when the numeric evaluation is otherwise valid.
- The seq pipeline now contains GRU retry logic for some cuDNN runtime
  failures, but that does not remove the need to compare environment behavior
  across GPU stacks carefully.

## Canonical Artifacts

Judge a run with:

- `tables/protocol_leaderboard.csv`
- `tables/summary_by_regime.csv`
- `tables/residual_signature_by_regime.csv`
- `tables/stat_fidelity_by_regime.csv`
- `train/tables/gridsearch_results.csv`
- `train/tables/grid_training_diagnostics.csv`

## Documentation Map

Start with:

1. [README.md](/workspace/2026/feat_seq_bigru_residual_cvae/README.md)
2. [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
3. [docs/active/WORKING_STATE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/WORKING_STATE.md)
4. [docs/active/ROUTES_AND_RESULTS.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/ROUTES_AND_RESULTS.md)
5. [docs/reference/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/PROTOCOL.md)

## Quick Resume

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
git status -sb
git worktree list
python scripts/analysis/summarize_experiment.py "$(ls -td outputs/exp_* | head -1)"
```
