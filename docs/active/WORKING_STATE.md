# Working State

This is the single active working note for the repository.

If you want to understand the current branch without reading historical plans,
start here.

## Current Worktree

- active worktree:
  - `/workspace/2026/feat_seq_bigru_residual_cvae`
- active branch:
  - `feat/mdn-g5-recovery`
- git worktree count:
  - `1`

## Current Scientific Anchors

- stable Gaussian reference:
  - `outputs/exp_20260324_023558`
  - `10/12`
- best MDN line so far:
  - `outputs/exp_20260325_230938`
  - `9/12`

## What Was Already Explored

- stronger old Gaussian loss:
  - negative result
- `sample-aware MMD`:
  - negative result
- aggressive MDN + hybrid loss:
  - unstable / over-dispersed
- conservative and exploratory MDN:
  - best result so far: `9/12`
- `conditional flow decoder`:
  - current implementation discarded
- pure regime-weighted resampling:
  - negative result

## Current Reading

The remaining gap is now narrow:

- the best MDN already passes `1.0 m` and `1.5 m`
- the unresolved zone is `0.8 m`
- the remaining problem is mostly `G5`
- the next intervention should target marginal shape more directly
  instead of reopening broad weighting or new decoder families

## Current Direction

Use the best MDN as the anchor and try only interventions that are local to the
remaining gap:

- per-axis shape control
- quantile / tail-aware regularization
- other direct `G5`-oriented corrections

The current implementation branch now includes an `MDN v2` path:

- `lambda_coverage` for direct marginal coverage / tail calibration
- `mini_protocol_v1` ranking for grid champion selection
- finite `decoder_sensitivity` for seq Gaussian / seq MDN
- `latent_summary` kept as audit-only telemetry, not a search criterion
- an opt-in throughput compare preset:
  - `seq_mdn_v2_perf_compare_quick`
  - control path keeps `seq_gru_unroll=True`
  - faster path tries `batch_size=8192`, `batch_infer=16384`
  - experimental GRU path tries `seq_gru_unroll=False`
  - keep the conservative default when moving to another GPU stack, especially the RTX 5090 machine
- the latest throughput compare selected the faster operational baseline:
  - `batch_size=8192`
  - `batch_infer=16384`
  - `seq_gru_unroll=False`
  - continuity preset for the next scientific quicks:
    - `seq_mdn_v2_fastbase_quick`
- the first scientific quick on top of that faster baseline improved the line:
  - run: `outputs/exp_20260327_021632`
  - champion: `S22 ... cov0.05 / t=0.03 ...`
  - protocol result: `5/12`
  - main gain: `G6` recovery compared with the fastbase anchor
  - remaining gap: `G5` still concentrated at `0.8 m`
- local follow-up preset:
  - `seq_mdn_v2_g5_followup_quick`
- the local follow-up improved the line again:
  - run: `outputs/exp_20260327_032019`
  - champion: `S23 ... cov0.06 / t=0.03 ...`
  - protocol result: `6/12`
  - main gain: `0.8m / 700mA` moved to pass
  - overnight decision preset:
    - `seq_mdn_v2_overnight_decision_quick`
    - mixes S23-local refinement and small exploratory probes

## Operational Attention Point

For `seq_bigru_residual`, any branch that uses per-experiment caps must be
checked for the post-cap `df_split` fix.

- affected configuration:
  - `max_samples_per_exp`
  - `max_val_samples_per_exp`
  - train-side sequence windowing
  - protocol-side sequence quick evaluation
- failure mode:
  - window center stays correct
  - left/right context can cross experiment boundaries
- safe configurations:
  - full runs without per-experiment caps
  - point-wise models without sequence windowing

When resuming or cherry-picking to another branch, verify that the equivalent of
commit `a1660e2` is present before trusting a quick sequential run.

Do not reopen:

- `sample-aware MMD`
- the current `sinh-arcsinh` flow line
- pure regime-resampling as the main intervention

## Minimal Read Order

1. [README.md](/workspace/2026/feat_seq_bigru_residual_cvae/README.md)
2. [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
3. [reference/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/PROTOCOL.md)
4. [reference/EXPERIMENT_WORKFLOW.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/EXPERIMENT_WORKFLOW.md)

## Archived Sources For This Working State

If you need the older detailed notes, they were archived here:

- [archive/active/ACTIVE_CONTEXT_legacy.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/archive/active/ACTIVE_CONTEXT_legacy.md)
- [archive/active/MDN_G5_RECOVERY_PLAN.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/archive/active/MDN_G5_RECOVERY_PLAN.md)
- [archive/active/TRAINING_PLAN_legacy.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/archive/active/TRAINING_PLAN_legacy.md)
- [archive/research/NOISE_DISTRIBUTION_AUDIT.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/archive/research/NOISE_DISTRIBUTION_AUDIT.md)
