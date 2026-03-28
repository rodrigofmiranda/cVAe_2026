# Working State

This is the single active working note for the repository.

If you want to understand the current branch without reading historical plans,
start here.

## Current Worktree

- active worktree:
  - `/workspace/2026/feat_seq_bigru_residual_cvae`
- active branch:
  - `feat/seq-imdd-graybox-mdn`
- git worktree count:
  - `2`

## Queued Architecture Routes

These are the current routes to try, in order, and they should stay on
separate branches.

- current route:
  - branch: `feat/seq-imdd-graybox-mdn`
  - objective: keep the IM/DD gray-box inductive bias but replace the
    Gaussian residual law with `MDN`
  - entry presets:
    - `seq_imdd_graybox_mdn_smoke`
    - `seq_imdd_graybox_mdn_guided_quick`
- next route:
  - branch: `feat/seq-bigru-residual-mdn-route`
  - objective: return to the stronger `seq_bigru_residual + MDN` family if
    gray-box + MDN still fails to copy the residual shape / constellation
    thickness
  - use this as the immediate fallback benchmark lane, not as a broad
    exploratory branch

## Current Scientific Anchors

- stable Gaussian reference:
  - `outputs/exp_20260324_023558`
  - `10/12`
- best MDN line so far:
  - `outputs/exp_20260325_230938`
  - `9/12`
- current gray-box anchor:
  - `outputs/exp_20260327_172148`
  - `6/12`
- gray-box + MDN branch result:
  - `outputs/exp_20260328_023302`
  - `5/12`

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

2026-03-28 route pivot:

- the gray-box Gaussian family is no longer the main search lane
- the new first-priority route is `seq_imdd_graybox + MDN`
- if that still leaves the constellation visibly too clean or the residual
  law visibly too Gaussian, move immediately to `seq_bigru_residual + MDN`
  on its dedicated branch instead of reopening another Gaussian sweep

2026-03-28 gray-box + MDN result:

- integration succeeded end to end on this branch
- smoke validation:
  - run: `outputs/exp_20260328_003030`
  - purpose: plumbing only
  - result: `0/12`
- first real guided run:
  - run: `outputs/exp_20260328_023302`
  - champion: `SGBM1 ... cov0.06 / lr=2e-4 / W7 / h32 / lat6`
  - protocol result: `5/12`
- training diagnostics on the four candidates did not indicate
  undertraining or posterior collapse
- practical reading:
  - `seq_imdd_graybox + MDN` is a valid implemented route
  - it did not beat the gray-box Gaussian anchor
  - it also did not justify staying as the main branch for the next sweep

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
  - 5090-safe overnight preset:
    - `seq_mdn_v2_overnight_5090safe_quick`
    - keeps `seq_gru_unroll=False` only on the validated `W7 / h64` branch
    - forces `seq_gru_unroll=True` on structural probes (`h96`, `W11`, combined probes)
  - A600 complementary exploratory preset:
    - `seq_mdn_v2_a600_tail_explore_quick`
    - opens a dedicated `tail_levels` sweep
    - keeps structural probes on the faster `gruroll0` path
    - meant to run in parallel with the 5090-safe overnight, not instead of it
  - A600 tail exploration result:
    - run: `outputs/exp_20260327_050422`
    - champion: `S26 ... lat6 ... tail02-98 ...`
    - protocol result: `5/12`
    - reading: negative for the hypothesis that a separate `tail_levels` sweep
      alone unlocks the remaining `0.8 m` gap
  - 5090-safe overnight result:
    - run: `outputs/exp_20260327_050158`
    - train-side winner: `S25 ... h96 / lat6 / gruroll1 ...`
    - protocol result is not scientifically valid yet
    - reason: evaluation environment was missing `matplotlib`, so every regime
      finished with `eval_status=failed`
    - useful signal that remains:
      - the strongest candidate came from a structural probe
      - `gate_g6` signal was the strongest seen so far in this MDN v2 branch
    - correct next action:
      - re-evaluate the trained model from `exp_20260327_050158/train`
      - do not open another 5090 grid before that re-evaluation

Current branch reading after these two runs:

- `S23` remains the best valid MDN v2 result: `6/12`
- the A600 tail-specific branch did not improve on it
- the 5090 structural branch may still have headroom, but that claim is blocked
  on environment parity, not on training quality
- the branch now includes a permanent runtime fix for this class of issue:
  - seq candidates that hit the cuDNN GRU runtime failure retry automatically
    on a compatibility backend
  - this removes the need to hand-curate grids just to dodge the RTX 5090 GRU
    failure mode
- the evaluation path also no longer invalidates a whole protocol just because
  `matplotlib` is missing; plots are skipped and the metrics still count

## Gray-Box Hyperparameter Gates

Use these as the official decision checklist for the next `seq_imdd_graybox`
sweeps.

- choose the winner by protocol result first; use train-side grid ranking only
  as a pre-protocol filter
- if `flag_undertrained=True`, first increase `max_epochs` / `patience`
  before changing `beta`, `free_bits`, or latent structure
- if `flag_posterior_collapse=True` or `active_dim_ratio` drops materially,
  adjust `free_bits`, `beta`, or `latent_dim`
- if `flag_overfit=True`, do not promote the candidate even if point metrics
  look good
- only lower the initial `lr` when `flag_lr_floor=True` and the late
  validation slope is still negative
- only increase capacity when training is otherwise stable and structural
  error remains high
- keep `train/tables/grid_training_diagnostics.csv` and
  `train/tables/gridsearch_results.csv` as mandatory artifacts for gray-box
  sweeps
- periodic per-regime train diagnostics are useful for live monitoring, but
  they are observability; the final hyperparameter decision comes from the
  saved diagnostics tables above

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

Also verify environment parity before trusting a completed protocol:

- `matplotlib` must be installed in the evaluation environment
- otherwise the run can train successfully, write reanalysis JSONs, and still
  end with `eval_status=failed`, which invalidates `G1-G3`

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
