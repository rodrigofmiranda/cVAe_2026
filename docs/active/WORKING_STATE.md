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

- stable Gaussian reference (old, 100k/exp):
  - `outputs/exp_20260324_023558`
  - `10/12`
- best valid MDN v2 (mini_protocol, 100k/exp):
  - `outputs/exp_20260327_161311`
  - `9/12` (`gate_g5_pass=10`, `gate_g6_pass=12`)
- **best valid MDN v2 (full protocol + G6, 8.6M train)**:
  - `outputs/exp_20260327_225213`
  - **`8/12`** champion: `S26full_lat8` (`latent_dim=8`, MDN 3 components)
  - all `1.0m` and `1.5m` pass; all `0.8m` fail
- Gaussian reference (full protocol + G6, 8.6M train):
  - `outputs/exp_20260328_023430`
  - `4/12` champion: `S26gauss_lat8`
  - MDN v2 gains +4 regimes over Gaussian (all `1.0m`)
- 0.8m isolation diagnostic:
  - `outputs/exp_20260328_042412`
  - `0/4` pass — confirms 0.8m failure is partly intrinsic, not only inter-distance conflict

## What Was Already Explored

- stronger old Gaussian loss:
  - negative result
- `sample-aware MMD`:
  - negative result
- aggressive MDN + hybrid loss:
  - unstable / over-dispersed
- conservative and exploratory MDN:
  - best result so far: `9/12` (mini_protocol), `8/12` (full protocol with G6)
- `conditional flow decoder`:
  - current implementation discarded
- pure regime-weighted resampling:
  - negative result
- ceiling probe axes (mdn_components, deeper decoder, higher beta, higher lambda_axis):
  - all negative or marginal vs lat8
- Gaussian decoder as reference:
  - `4/12` full protocol — MDN v2 clearly better (+4 regimes on all `1.0m`)
- 0.8m isolation (trained only on 0.8m data):
  - `0/4` — G3 fixed by isolation but G5/G6 remain; under-dispersion persists

## Current Reading

The universal under-dispersion problem:

- `Δcov95 ≈ -0.17` across **all** 12 regimes, including those that pass the
  protocol gates
- the cVAE generates lighter tails and a slightly broader centre than the real
  distribution — visible in all noise distribution plots and constellation
  overlays
- even at regimes where the model "passes", the shape is not fully correct:
  the real distribution has a sharper peak and heavier tails simultaneously
  (leptokurtic profile)
- this is a systematic under-dispersion driven by the ELBO objective: the
  model prioritises the bulk of the distribution and does not have enough
  gradient pressure on the tails

The 0.8m problem has two independent layers:

- **G3 (covariance)**: caused by inter-distance conflict in the global model;
  isolation fixes it (`covVar` drops from `0.32` to `0.07` at `100mA`)
- **G5/G6 (shape + statistical fidelity)**: intrinsic to 0.8m low-current
  regimes; the real distribution is near-Gaussian at low currents, so any
  mismatch in the kurtosis/skew dimension becomes a large relative error;
  the isolated model oscillates between "too Gaussian" and "too
  non-Gaussian" without converging to the correct shape; `MMDq` collapses
  to `~0.007` even in isolation

Current active intervention:

- `seq_coverage_tail_sweep` (S27) — in progress
- sweeps `lambda_coverage` from `0.06` to `0.40` and `tail_levels` from
  `[0.05, 0.95]` to `[0.01, 0.99]`, plus a harder `coverage_temperature=0.01`
- hypothesis: the current gradient from coverage loss is too weak relative to
  reconstruction loss; a stronger coverage signal should force the model to
  place more probability mass in the tails

## Current Direction

Use `S26full_lat8` (exp_20260327_225213) as the anchor and target the
tail/coverage calibration gap directly:

- `lambda_coverage` sweep: `0.06 → 0.15 → 0.25 → 0.40`
- `tail_levels` extension: `[0.05, 0.95] → [0.01, 0.99]`
- `coverage_temperature` hardening: `0.03 → 0.01`
- do not reopen architecture search until coverage calibration is resolved

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
  - 5090-safe overnight historical result:
    - run: `outputs/exp_20260327_050158`
    - train-side winner: `S25 ... h96 / lat6 / gruroll1 ...`
    - protocol result is not scientifically valid yet
    - reason: evaluation environment was missing `matplotlib`, so every regime
      finished with `eval_status=failed`
    - useful signal that remains:
      - the strongest candidate came from a structural probe
      - `gate_g6` signal was the strongest seen so far in this MDN v2 branch
    - next action that was executed:
      - re-evaluation run: `outputs/exp_20260327_161311`
  - valid re-evaluation result:
    - run: `outputs/exp_20260327_161311`
    - champion: `S25 ... W7 / h64 / lat6 / gruroll1 ...`
    - protocol result: `9/12` (mini_protocol)
    - remaining failures: all at `0.8m`

- MDN v2 ceiling analysis (S26 series):
  - preset: `seq_mdn_v2_ceiling_probe_quick`
  - axes probed: `mdn_components=5`, deeper decoder, `latent_dim=8`,
    `beta=0.005`, `lambda_axis=0.05`
  - run (100k/exp, mini_protocol): `outputs/exp_20260327_191958`
    - winner: `lat8` — `5/12` mini; control (S25 clone): `4/12`
    - finding: significant stochastic variance between runs with identical
      config; the `9/12` result from `exp_20260327_161311` is an optimistic
      draw, not a stable mode
  - run (full data 8.6M, full protocol + G6): `outputs/exp_20260327_225213`
    - preset: `seq_mdn_v2_ceiling_full`
    - winner: `S26full_lat8` — **`8/12` full protocol**
    - all `1.0m` and `1.5m` regimes pass; all `0.8m` fail
    - G6 now computed — passes everywhere except `0.8m`
    - `score_v2=0.327`

- Gaussian decoder reference (full data 8.6M, full protocol + G6):
  - preset: `seq_gaussian_reference_full`
  - run: `outputs/exp_20260328_023430`
  - winner: `S26gauss_lat8` — `4/12`
  - only `1.5m` regimes pass; all `1.0m` fail (vs MDN which passes all `1.0m`)
  - MDN v2 advantage is clear and consistent: +4 regimes on all `1.0m`
  - MDN advantage mechanism: better JB/skew calibration at intermediate
    distances (`JBrel` 0.03–0.11 for MDN vs 0.22–0.37 for Gaussian)

- 0.8m isolation diagnostic:
  - preset: `seq_mdn_v2_0p8m_isolation`
  - protocol: `configs/regimes_0p8m_only.json`
  - run: `outputs/exp_20260328_042412`
  - result: `0/4`
  - findings:
    - G3 (covariance) is **fixed** by isolation: `covVar` drops from `0.32` to
      `0.07` at `100mA` — inter-distance conflict in global model confirmed
    - G5 (JB/shape) is **not fixed**: model oscillates — at low currents
      (`100mA`, `300mA`) prediction becomes too non-Gaussian (`JBrel +1.6`,
      `+1.3`) while global model is too Gaussian (`JBrel -0.51`)
    - G6 (MMD/Energy) **worsens**: `MMDq` collapses to `0.007` in isolation
      vs `0.024` in global — sample fidelity is poor even with specialised model
    - universal under-dispersion persists: `Δcov95 ≈ -0.20` (worse than
      global `-0.17`); tails remain too short regardless of isolation

Current branch reading (2026-03-28):

- `S26full_lat8` (`exp_20260327_225213`) is the best result under fair conditions
  (full protocol, G6 computed, full training data): **`8/12`**
- MDN v2 decisively better than Gaussian decoder: `8/12` vs `4/12`
- the remaining `4/12` gap is `0.8m` only and has two causes:
  - G3: inter-distance covariance conflict (addressable with better conditioning)
  - G5/G6: systematic under-dispersion + shape calibration failure at low-current
    near-Gaussian regimes (under-coverage Δcov95 ≈ -0.17 universal)
- active intervention: `seq_coverage_tail_sweep` (S27) targeting `lambda_coverage`
  and `tail_levels`

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

**Critical operational note**: always use `--max_samples_per_exp 100000` (not
the default `20000`) to match the training data volume of `exp_20260327_161311`.
Using the default `20000` gives `240k` train samples vs `1.2M`, which degrades
results dramatically (observed: `1/12` vs `8/12` for identical config).

When resuming or cherry-picking to another branch, verify that the equivalent of
commit `a1660e2` is present before trusting a quick sequential run.

Also verify environment parity before trusting a completed protocol:

- `matplotlib` should stay installed for full dashboard generation
- if `matplotlib` is missing, plotting can be skipped; in this branch the
  protocol metrics can still be counted
- if `eval_status=failed` appears, inspect run logs before discarding the run

Do not reopen:

- `sample-aware MMD`
- the current `sinh-arcsinh` flow line
- pure regime-resampling as the main intervention
- ceiling probe axes (mdn_components, deeper decoder, higher beta, lambda_axis)
  — all tested negative vs lat8 baseline

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
