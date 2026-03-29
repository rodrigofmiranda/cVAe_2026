# Working State

This is the single active working note for the repository.

If you want to understand the current branch without reading historical plans,
start here.

## Current Worktree

- active worktree:
  - `/workspace/2026/feat_seq_bigru_residual_spline_flow_v2`
- active branch:
  - `feat/seq-bigru-residual-spline-flow-v2`
- git worktree count:
  - `5`

## Current Flow Route Snapshot

- route purpose:
  - isolate the next materially different conditional-density line on its own
    branch/worktree
- decoder families currently wired in this lane:
  - `flow_family="spline_2d"`:
    - new default here
    - per-axis rational-quadratic spline flow with decoder-conditioned
      parameters
  - `flow_family="coupling_2d"`:
    - legacy compatibility / carry-over from flow v1
  - `flow_family="sinh_arcsinh"`:
    - legacy control / compatibility only
- new presets:
  - `seq_flow_spline_smoke`
  - `seq_flow_spline_guided_quick`
- first structural smoke in this lane:
  - `outputs/exp_20260329_015508`
  - result: `0/12`
  - reading:
    - integration success
    - scientific result still negative
    - diagnostics: `flag_undertrained=True`, `flag_posterior_collapse=True`,
      `active_dim_ratio=0.0`
- first meaningful quick in this lane:
  - `outputs/exp_20260329_015815`
  - preset: `seq_flow_spline_guided_quick`
  - result: `0/12`
  - reading:
    - not just a smoke failure
    - all 4 grid candidates also ended `0/12` on the mini protocol
    - `gate_g5_pass=0`
    - diagnostics no longer point to undertraining or collapse
    - the present `spline_2d` formulation is a formal negative result
- branch-local decision:
  - stop sweeping the current `flow_family="spline_2d"`
  - preserve this branch as a documented negative route
  - if flow returns later, use a materially different family
  - next serious global route: conditional diffusion

## Current Scientific Anchors

- stable Gaussian reference (old, 100k/exp):
  - `outputs/exp_20260324_023558`
  - `10/12`
- best valid MDN v2 (mini_protocol, 100k/exp):
  - `outputs/exp_20260327_161311`
  - `9/12` (`gate_g5_pass=10`, `gate_g6_pass=12`)
- best valid MDN v2 (full protocol + G6, 8.6M train) — superseded:
  - `outputs/exp_20260327_225213`
  - `8/12` champion: `S26full_lat8` (`latent_dim=8`, MDN 3 components)
  - all `1.0m` and `1.5m` pass; all `0.8m` fail
- **best valid MDN v2 (full protocol + G6, 8.6M train)**:
  - `outputs/exp_20260328_153611`
  - **`10/12`** champion: `S27cov_lc0p25_tail95_t0p03` (`lambda_coverage=0.25`)
  - passes: all `1.5m`, all `1.0m`, `0.8m/500mA`, `0.8m/700mA`
  - fails: `0.8m/100mA` (JBrel=3.59), `0.8m/300mA` (JBrel=0.36) — G5 only
  - G3 now passes on all 12 regimes; G6 now passes on all 12 regimes
  - `gate_g5_pass=10`, `gate_g6_pass=12`, `mean_stat_mmd_qval=0.37`
- Gaussian reference (full protocol + G6, 8.6M train):
  - `outputs/exp_20260328_023430`
  - `4/12` champion: `S26gauss_lat8`
  - MDN v2 gains +6 regimes over Gaussian (now with S27 anchor)
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
  - best result: `9/12` (mini_protocol), `8/12` → `10/12` (full protocol with G6)
- `conditional flow decoder`:
  - old `sinh-arcsinh`: negative
  - `coupling_2d`: negative
  - `spline_2d`: negative
- pure regime-weighted resampling:
  - negative result
- ceiling probe axes (mdn_components, deeper decoder, higher beta, higher lambda_axis):
  - all negative or marginal vs lat8
- Gaussian decoder as reference:
  - `4/12` full protocol — MDN v2 clearly better (+6 regimes with S27 anchor)
- 0.8m isolation (trained only on 0.8m data):
  - `0/4` — G3 fixed by isolation but G5/G6 remain; under-dispersion persists
- `lambda_coverage` sweep (S27, 6 candidates):
  - **winner: `lambda_coverage=0.25`, `10/12` full protocol** — new best
  - `lambda_coverage=0.40`: collapses back to `4/12` — too aggressive, destabilises ELBO
  - `tail_levels=[0.01,0.99]` (B/C candidates): negative — broader tails hurt more than help
  - the optimal point is narrow: `0.25` wins, `0.15` gives only `5/12`, `0.40` regresses
  - G3 and G6 now pass everywhere; G5 still fails at `0.8m/100mA` and `0.8m/300mA`

## Current Reading

The universal under-dispersion problem — updated after S27:

- with `lambda_coverage=0.06` (previous anchor): `Δcov95 ≈ -0.14` avg across
  12 regimes — coverage_95 ≈ 0.806 mean (target: 0.950)
- with `lambda_coverage=0.25` (S27 winner): `Δcov95 ≈ -0.18` avg — coverage
  actually regressed slightly in the quantile sense, **but G5 (JB/shape)
  improved dramatically** across all regimes except 0.8m/100mA and 0.8m/300mA
- interpretation: the stronger coverage signal does not increase the quantile
  coverage of the marginal, but it reshapes the gradient so that the MDN
  components learn a **better shape** (lower skew/kurt error, lower JBrel)
  at the cost of slightly narrower marginal quantiles — net gain: +2 regimes
- the shape mismatch is not solved by coverage loss alone; it is reformulated

Residual shape gap (visible in analysis dashboards, 0.8m regimes):

- **0.8m / 100mA, 300mA** (fail, G5✗): the real distribution has a very sharp,
  narrow peak and heavier tails simultaneously (leptokurtic); the model
  produces a flatter, more uniform bell — the MDN cannot capture this with
  3 components at this regime; JBrel = 3.59 (100mA) and 0.36 (300mA)
- **0.8m / 500mA, 700mA** (pass, G5✓): JBrel = 0.17 — just under threshold;
  the noise distribution shape is better but the **residual constellation
  overlay is still visibly different**: the model's point cloud is more
  uniform/spherical, the real data shows more structure at the edges
- **even passing regimes (1.0m, 1.5m)**: the constellation overlay still
  reveals the same gap — the model is always slightly more uniform, the real
  data has residual spatial structure at the extremes that the cVAE does not
  reproduce
- this is a systematic pattern: the model captures the **bulk** of the
  distribution well but misses the **leptokurtic edge structure** regardless
  of distance or current; visible in every noise distribution Q-Q and every
  constellation overlay

The 0.8m problem has two independent layers:

- **G3 (covariance)**: caused by inter-distance conflict — now resolved at
  `lambda_coverage=0.25`; G3 passes in all 12 regimes including 0.8m
- **G5 (shape)**: intrinsic to 0.8m low-current regimes; the real distribution
  is near-Gaussian with high kurtosis at low currents; any shape mismatch
  becomes a large relative JB error; `lambda_coverage=0.25` is insufficient
  to fix 100mA and 300mA

Current branch reading (2026-03-28, after S27):

- `S27cov_lc0p25_tail95_t0p03` (`exp_20260328_153611`) is the new best:
  **`10/12`** full protocol
- remaining `2/12` gap is `0.8m/100mA` and `0.8m/300mA`, failing only G5
- G6 now passes everywhere — the statistical fidelity (MMD/Energy) is solid
- the residual constellation overlay gap is **systematic and not resolved** by
  the current approach; even in passing regimes the model point cloud is too
  uniform relative to real data

Current global reading after the dedicated flow lanes:

- changing the flow family alone has not solved the universal shape gap
- the negative flow lines are now:
  - old `sinh-arcsinh`
  - `coupling_2d`
  - `spline_2d`
- the next serious global generative line is conditional diffusion

## Current Direction

New anchor: `S27cov_lc0p25_tail95_t0p03` (`exp_20260328_153611`, `10/12`).

S27 has closed the coverage/lambda axis. The residual gap is:

1. G5 at `0.8m/100mA` and `0.8m/300mA` — JBrel still above threshold; the
   model cannot reproduce the leptokurtic shape of low-current 0.8m noise
2. Constellation overlay gap — universal, present even in passing regimes;
   the model point cloud is always too uniform relative to real data

Open questions for the next intervention:

- **more MDN components** at 0.8m low-current regimes (targeted, not global)?
  Previously tested globally (negative); targeted via regime-conditioning not
  yet tried
- **explicit kurtosis / higher-moment loss term**? Would add direct pressure on
  the 4th moment rather than relying on JB indirectly through coverage
- **accept 10/12 and move to next scientific milestone**? The remaining failure
  is intrinsic to the 0.8m/low-current regime and has resisted all structural
  interventions so far

Do not reopen:
- `tail_levels=[0.01,0.99]` — tested negative in S27
- `lambda_coverage>0.25` — 0.40 collapses to 4/12
- `coverage_temperature=0.01` alone — marginal gain vs complexity

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
- S27 result: `seq_coverage_tail_sweep` — `10/12` full protocol at
  `lambda_coverage=0.25`; remaining failures: `0.8m/100mA` and `0.8m/300mA`
  (G5 only, JBrel too high); G3 and G6 now pass in all 12 regimes

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
