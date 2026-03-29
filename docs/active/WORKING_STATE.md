# Working State

This is the single active working note for the repository.

If you want to understand the current branch without reading historical plans,
start here.

## Current Worktree

- active worktree:
  - `/workspace/2026/feat_seq_bigru_residual_diffusion`
- active branch:
  - `feat/seq-bigru-residual-diffusion`
- git worktree count:
  - `6`

## Current Route Launch

- route purpose:
  - open the next global generative family after the MDN and flow limits became
    clear
- base anchor inherited from the parent worktree:
  - `outputs/exp_20260328_153611`
  - `S27cov_lc0p25_tail95_t0p03`
  - `10/12`
- scientific reason for opening this route:
  - the residual-shape mismatch is global, not only local to the last two
    failing regimes
  - even passing regimes still show a constellation that is too uniform
  - three flow lines are now negative:
    - `sinh-arcsinh`
    - `coupling_2d`
    - `spline_2d`
- current branch status:
  - no diffusion implementation committed yet
  - no diffusion smoke or quick run yet
  - this worktree is the staging lane for that integration

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
- latest regime-conditioning probe (full protocol + G6, 8.6M train):
  - `outputs/exp_20260328_233844`
  - `5/12` champion: `S30ce_embed16` (`cond_embed_dim=16`)
  - passes: all `1.5m`, `1.0m/700mA`
  - fails: all `0.8m`, `1.0m/100mA`, `1.0m/300mA`, `1.0m/500mA`
  - reading: decoder-side regime embedding avoided the worst control collapse,
    but regressed sharply against the S27 anchor and did not fix the two
    remaining low-current `0.8m` G5 failures
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
- **explicit kurtosis loss `lambda_kurt` sweep (S28, 4 candidates)**:
  - run: `outputs/exp_20260328_191811`
  - base: exact S27 A2 config (`lambda_coverage=0.25`, MDN3, lat8)
  - **CTRL (lk=0.0) regressed to `4/12` mini** — expected `~9/12` from S27 A2; **confirms
    HIGH TRAINING VARIANCE**: the S27 A2 `10/12` result is at the optimistic tail of the
    initialization distribution, not a stable mode
  - G3 regressed from `12/12` (S27) to `5/12` (CTRL): the covariance fix from
    `lambda_coverage=0.25` is also variance-dependent, not guaranteed
  - **lambda_kurt does NOT improve G5**: stuck at `4/12` across all 4 candidates
    (only `1.5m` regimes pass G5 regardless of lambda_kurt)
  - **lambda_kurt at moderate values is destructive**: lk=0.10 causes JBrel
    explosion at `0.8m` (JBrel_I/Q up to 588/462) — catastrophic shape degradation
  - G3 and G6 **do improve** with lambda_kurt: at lk=0.20, G3=9/12 and G6=9/12,
    better than CTRL (G3=5, G6=5) — the kurtosis pressure reshapes the MDN
    components in a way that helps distributional matching but not JB shape
  - **conclusion: lambda_kurt is not the right lever for G5**; the JB test failure
    at 0.8m is not addressable by isolated 4th-moment pressure; the gap is rooted in
    the MDN's inability to reproduce the leptokurtic structure at low current
- **MDN components sweep (S29, 3 candidates: k=3/5/8)**:
  - run: `outputs/exp_20260328_210953`
  - base: exact S27 A2 config (`lambda_coverage=0.25`, lat8)
  - CTRL k=3: `8/12` full protocol — **3rd independent seed**: distribution of full protocol
    across 3 seeds of S27-A2 config = `{10, 4, 8}`, expected value ~7, high variance
  - CTRL k=3 mini = `7/12` | G3=11 G5=7 G6=12 — better than S28 CTRL (4/12),
    confirming S28 was a bad draw not a systematic regression
  - **k=5 (A1): regresses to `5/12` mini** | G3=9 G5=5 G6=9 — strictly worse
    than k=3; more components destabilise G3 and G6 at this learning rate
  - **k=8 (A2): `6/12` mini** | G3=10 G5=6 G6=12 — marginal G5 improvement vs
    k=3 (6 vs 7), G6 recovers to 12/12, but 0.8m/100mA and 300mA still fail G5
  - **0.8m/300mA JBrel is extreme in all seeds**: CTRL=8.07, k=5=10.47, k=8=3.45 —
    this regime's leptokurtic shape does not yield to any MDN capacity increase tested
  - **conclusion: more MDN components do NOT fix G5**; k=5 hurts, k=8 is marginal;
    the shape constraint at 0.8m low-current is architecture-agnostic within MDN family
- **decoder regime-conditioning embedding sweep (S30, 3 candidates: embed=0/16/32)**:
  - run: `outputs/exp_20260328_233844`
  - base: exact S27 A2 config (`lambda_coverage=0.25`, lat8, MDN3)
  - CTRL `embed=0`: collapsed to `1/12` mini / severe full-protocol regression —
    another strong sign of high training variance in this family
  - `embed=16`: selected champion, but only `5/12` full protocol
  - `embed=32`: no scientific gain over `embed=16`; mini stays at `4/12`
  - **conclusion: a small shared decoder conditioning embedding does NOT fix G5
    at `0.8m/100mA` and `0.8m/300mA`; it broadens regression into `1.0m`**

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
  to fix 100mA and 300mA; tested MDN k=3/5/8 — none fixes it (S29)

Current branch reading (2026-03-29, after S27 + S28 + S29 + S30):

- `S27cov_lc0p25_tail95_t0p03` (`exp_20260328_153611`) is the best known result:
  **`10/12`** full protocol
- three independent seeds of S27-A2 config: `{10, 4, 8}` full protocol passes
  → E[pass] ≈ 7, high variance; the `10/12` is the optimistic tail
- S29 CTRL k=3 = `8/12` full / `7/12` mini — better than S28 CTRL (`4/12`),
  confirming S28 was an unlucky draw; typical expected value is ~7–8/12
- remaining `2/12` gap is `0.8m/100mA` and `0.8m/300mA`, failing only G5;
  tested lambda_axis, lambda_kurt, lambda_coverage, k=3/5/8 MDN — none fixes it
- G5 failure at 0.8m/300mA is extreme (JBrel=3–10 across seeds) and persistent
- the residual constellation overlay gap is **systematic and not resolved**;
  even in passing regimes the model point cloud is too uniform relative to real data
- S30 `cond_embed_dim` was the first decoder-side regime-specific conditioning
  probe and regressed to `5/12`; the best candidate (`embed=16`) still fails
  all `0.8m` regimes and also loses `1.0m/100mA`, `300mA`, `500mA`
- the current evidence says a **small shared regime embedding is too weak** to
  address the low-current leptokurtic shape gap; if this axis is revisited, it
  likely needs a stronger regime-specific head/expert instead of a shallow
  shared embedding

Global reading that motivates this branch:

- the benchmark gap is still visible in `G5`, but the scientific gap is wider
- no current family reproduces the full residual distribution shape globally
- the next route should therefore change the generative family itself, not just
  local MDN knobs

## Current Direction

Anchor: `S27cov_lc0p25_tail95_t0p03` (`exp_20260328_153611`, `10/12` best known).

The next route is **conditional diffusion**, because the open problem is
global: the current families do not reproduce the full residual distribution
shape, even when they pass the protocol.

Immediate branch goals:

- keep the current seq backbone and conditioning path as much as possible
- replace the decoder-family logic with a conditional diffusion route
- start with a minimal structural milestone:
  - build
  - train
  - save/load
  - deterministic and stochastic inference
  - smoke run through `src.protocol.run`
- only then open a guided quick against the `S27` anchor

Implementation focus:

- seq-only first; do not generalize to every architecture family immediately
- residual-domain generation first (`y = x + residual_sample`)
- deterministic path should use a reproducible sampler, not ad-hoc averaging
- preserve protocol compatibility and the existing artifact layout

Do not reopen:
- local sweeps of the three negative flow lines
- global MDN knob sweeps already exhausted in S28/S29/S30

Do not reopen:
- `tail_levels=[0.01,0.99]` — tested negative in S27
- `lambda_coverage>0.25` — 0.40 collapses to 4/12
- `coverage_temperature=0.01` alone — marginal gain vs complexity
- **`lambda_kurt`** — S28 negative; catastrophic at lk=0.10; not the right lever
- **`mdn_components` global sweep** — S29 negative; k=5 hurts, k=8 marginal;
  more components globally do not fix 0.8m G5
- **small decoder `cond_embed_dim` sweep** — S30 negative; `embed=16/32` do not
  recover the remaining two 0.8m G5 regimes and regress 1.0m coverage

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

1. [README.md](/workspace/2026/feat_seq_bigru_residual_diffusion/README.md)
2. [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_diffusion/PROJECT_STATUS.md)
3. [reference/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_diffusion/docs/reference/PROTOCOL.md)
4. [reference/EXPERIMENT_WORKFLOW.md](/workspace/2026/feat_seq_bigru_residual_diffusion/docs/reference/EXPERIMENT_WORKFLOW.md)

## Archived Sources For This Working State

If you need the older detailed notes, they were archived here:

- [archive/active/ACTIVE_CONTEXT_legacy.md](/workspace/2026/feat_seq_bigru_residual_diffusion/docs/archive/active/ACTIVE_CONTEXT_legacy.md)
- [archive/active/MDN_G5_RECOVERY_PLAN.md](/workspace/2026/feat_seq_bigru_residual_diffusion/docs/archive/active/MDN_G5_RECOVERY_PLAN.md)
- [archive/active/TRAINING_PLAN_legacy.md](/workspace/2026/feat_seq_bigru_residual_diffusion/docs/archive/active/TRAINING_PLAN_legacy.md)
- [archive/research/NOISE_DISTRIBUTION_AUDIT.md](/workspace/2026/feat_seq_bigru_residual_diffusion/docs/archive/research/NOISE_DISTRIBUTION_AUDIT.md)
