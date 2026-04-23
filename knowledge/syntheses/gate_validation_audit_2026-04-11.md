# Gate Validation Audit

Date: 2026-04-11

## Purpose

This note audits the current `G1..G6` gate system used to validate digital-twin
results in the full-circle line.

The goal is to answer a narrower question before further optimisation around
`G6`:

- which parts of the gate system are sound as engineering validation helpers
- which parts are statistically well-grounded
- which parts are currently being interpreted too strongly

## Current Implementation

Canonical implementation:

- thresholds: `src/evaluation/validation_summary.py`
- stat-test suite: `src/evaluation/stat_tests/__init__.py`
- MMD test: `src/evaluation/stat_tests/mmd.py`
- Energy test: `src/evaluation/stat_tests/energy.py`
- FDR correction: `src/evaluation/stat_tests/fdr.py`
- direct evaluation path: `src/evaluation/engine.py`
- protocol path: `src/protocol/run.py`

Current thresholds:

| Gate | Rule | Intended meaning |
|---|---|---|
| `G1` | `cvae_rel_evm_error < 0.10` | direct signal fidelity in EVM |
| `G2` | `cvae_rel_snr_error < 0.10` | direct signal fidelity in SNR |
| `G3` | `cvae_mean_rel_sigma < 0.10` and `cvae_cov_rel_var < 0.20` | residual mean/dispersion fidelity |
| `G4` | `cvae_psd_l2 < 0.25` | spectral fidelity of the residual |
| `G5` | `cvae_delta_skew_l2 < 0.30` and `cvae_delta_kurt_l2 < 1.25` and `delta_jb_stat_rel < 0.20` | higher-order residual-shape fidelity |
| `G6` | `stat_mmd_qval > 0.05` and `stat_energy_qval > 0.05` | no statistically detected residual-distribution mismatch under the configured test budget |

Status rule:

- `validation_status = fail` if any available gate is `False`
- `validation_status = pass` if all available gates are `True`
- `validation_status = partial` otherwise

This means the current validation system gives all gates equal veto power.

## What Is Strong In The Current Design

### 1. The ladder itself is scientifically reasonable

The structure

- `G1/G2` direct signal fidelity
- `G3/G4/G5` residual-structure fidelity
- `G6` formal two-sample testing

is a good validation ladder for a digital twin.

It matches two ideas that are consistent with the project literature:

- model quality should be judged against measured channel behaviour under
  multiple operating conditions, not just training loss
- higher-order moments, tails, and sequence- or residual-level structure
  matter in nonlinear channels

Local support includes the Askari/Lampe and Shu shaping notes plus the VLC
shaping methodology synthesis already present in the wider project history.

### 2. The project is not relying on a single metric

This is a real strength.

The code does not reduce validation to:

- only MSE
- only EVM
- or only one hypothesis test

That is much closer to engineering validation practice than a single-score
selection rule.

### 3. The chosen formal tests are legitimate two-sample tools

The `G6` tests are not ad hoc:

- `MMD` is a standard kernel two-sample test
- `Energy` distance is a standard metric/test for equality of distributions
- `BH/FDR` is a standard multiplicity correction

So the issue is not that the tests are invalid.
The issue is how strongly we interpret passing them.

## What Needs To Be Reframed

### 1. `G6` is currently interpreted too strongly

In the code comments and plots, `G6` is described as:

- `formal distributional indistinguishability`

That wording is too strong for the procedure actually implemented.

What the current procedure really establishes is:

- the model and real residuals were compared with two null-hypothesis
  two-sample tests
- after BH correction, the tests did not reject at `q > 0.05`

That is not the same as proving equality or equivalence of distributions.

Methodological reason:

- failing to reject a null hypothesis is not the same as demonstrating
  equivalence
- p-values do not quantify effect size or practical agreement by themselves

This matters a lot for thesis language.

Safer wording:

- `G6 = no statistically detected mismatch under the configured MMD/Energy test budget`

Risky wording:

- `G6 = distributions are indistinguishable`
- `G6 = model is equivalent to the real channel`

### 2. `G6` will generally get harder as sample size increases

This is not a bug.
It is how two-sample testing works when there is real but small mismatch.

In our pipeline:

- `quick` uses `n_perm=200` and `stat_max_n=5000`
- `full` uses `n_perm=2000` and `stat_max_n=50000`
- the evaluation path sub-samples residuals before running `MMD` and `Energy`

So a model can:

- pass `G6` in a lighter configuration
- and fail `G6` in a stronger configuration

without any contradiction.

This means `G6` is partly a property of:

- the model
- the regime
- the sample size
- the permutation budget
- and the kernel bandwidth heuristic

not only of the model.

### 3. `G5` is more engineering-useful than statistically canonical

`G5` combines:

- skew mismatch
- kurtosis mismatch
- a JB-derived relative gap

This is defensible as an engineering gate for tail and shape fidelity.
But it should be presented as:

- a project-specific residual-shape acceptance gate

not as a standard textbook validation theorem.

### 4. `validation_status` currently mixes different kinds of evidence

Right now a regime only gets `pass` if:

- engineering-fidelity gates pass
- and the NHST-based `G6` gate passes

This creates a very strict operational rule, which is fine for screening.
But it mixes:

- practical signal fidelity
- residual-shape fidelity
- formal multiple-testing output

into one binary veto.

That is acceptable for model selection.
It is less ideal for thesis claims, where these layers should be reported
separately.

## Audit Of Each Gate

### `G1` and `G2`

Strengths:

- directly tied to physically interpretable communication metrics
- easy to explain
- regime-specific

Weaknesses:

- thresholds appear hard-coded unless a calibration memo is attached

Audit verdict:

- keep them
- document them as engineering acceptance thresholds
- explicitly state they are project-calibrated, not universal

### `G3`

Strengths:

- uses channel-relative scale (`sigma_real`, `var_real_delta`)
- much better than a raw unnormalized mean/cov threshold

Weaknesses:

- still threshold-based, with no explicit calibration report attached

Audit verdict:

- keep it
- this is one of the better-designed gates
- document how the thresholds were chosen or calibrate them from accepted runs

### `G4`

Strengths:

- spectral fidelity is important in this channel family
- nicely complements pointwise and moment metrics

Weaknesses:

- threshold still looks project-defined rather than literature-derived

Audit verdict:

- keep it
- calibrate and document the threshold empirically

### `G5`

Strengths:

- explicitly targets shape mismatch beyond covariance
- consistent with the project concern about tails, kurtosis, and edge/corner
  behaviour
- conceptually aligned with the shaping/nonlinearity literature

Weaknesses:

- mixes several related but not identical shape indicators
- `JB`-based terms can become sensitive with large `N`
- threshold justification is not yet formalized in active docs

Audit verdict:

- keep it as an engineering gate
- stop presenting it as if it were a generic textbook gold standard
- formalize calibration

### `G6`

Strengths:

- uses legitimate two-sample tests
- adds formal distribution-level scrutiny after the engineering gates
- BH correction is a serious step and avoids naive multiple-testing use

Weaknesses:

- `q > 0.05` is not equivalence
- outcome depends on `N`, `n_perm`, and test configuration
- code currently describes it too strongly
- the current single-threshold use makes "pass/fail" vulnerable to
  configuration effects

Audit verdict:

- keep `G6`
- change how we describe it
- for thesis claims, do not use `G6 pass` as proof of equivalence

## Specific Audit Findings In The Current Pipeline

### FDR is applied jointly across both test families

The code concatenates:

- all `MMD` p-values
- all `Energy` p-values

and runs one BH correction over the combined list.

That is conservative and defensible if the intention is:

- to control the overall false discovery rate across all formal tests in the
  protocol

But it must be documented, because it is stricter than "one BH per family".

### `G6` is run on residuals, not raw outputs

This is a good design decision.

The tests compare:

- `res_real = Y_true - X_center`
- `res_pred = Y_pred - X_center`

So `G6` is about residual-channel fidelity, not just absolute output-space fit.

This is more aligned with the digital-twin purpose of the repo.

### `quick` versus `full` is not a mere speed knob

It changes the evidential meaning of `G6`.

Therefore every `G6` claim should be reported with:

- `stat_mode`
- `stat_max_n`
- `stat_n_perm`
- if relevant, `mmd_bandwidth`

Otherwise two `G6` results are not directly comparable.

## What The Literature Supports

The local `knowledge/` base supports:

- hardware-aware evaluation rather than abstract AWGN optimality
- attention to higher-order moments, tails, and memory
- measured-channel validation across operating conditions

This strongly supports `G1..G5` as a meaningful engineering ladder.

External statistical methodology supports the current ingredients, but also a
more careful interpretation:

- `MMD` is a valid two-sample test for distribution comparison
- `Energy` distance is a valid distribution-comparison metric/test
- `BH` is a valid FDR correction
- non-significant p-values do not establish equivalence

## Recommended Reframing For The Thesis

### Recommended wording

Use:

- `G6 tests whether the measured and generated residual samples show no statistically detected mismatch under MMD and Energy tests after FDR correction.`

Avoid:

- `G6 proves the distributions are identical`
- `G6 proves the model is equivalent to the channel`
- `G6 establishes indistinguishability` without qualification

### Recommended reporting split

For thesis/reporting, separate:

1. engineering fidelity
   - `G1..G5`
2. formal statistical screen
   - `G6`
3. operational decision
   - combined `validation_status`

That preserves the usefulness of the current pipeline without overstating what
`G6` means.

### Recommended next methodological step

If we want a truly stronger final acceptance claim than "no detected mismatch",
we should add one of these:

1. equivalence-style acceptance on domain metrics
   - define practically negligible bounds for selected residual metrics
   - test whether confidence intervals stay inside those bounds
2. bootstrap confidence bounds on effect-size metrics
   - not only p-values
3. threshold calibration memo from repeated accepted runs
   - to justify `G1..G5` quantitatively

For this project, the most practical path is:

- keep `G1..G6` for screening and internal iteration
- add an explicit calibration and reporting layer for thesis claims

## Recommended Immediate Changes

1. Rename the interpretation of `G6` in docs from
   `formal distributional indistinguishability`
   to
   `no detected residual-distribution mismatch under MMD/Energy + BH`.

2. Keep `validation_status` operationally, but in analysis tables also expose:
   - `engineering_pass = G1..G5`
   - `stat_screen_pass = G6`

3. Create a threshold-calibration note for:
   - `G1`
   - `G2`
   - `G3`
   - `G4`
   - `G5`

4. For final model claims, always report:
   - `stat_mode`
   - `stat_max_n`
   - `stat_n_perm`
   - number of regimes tested
   - whether BH was joint across test families

## External References

These references are the most relevant anchors for the audit:

- Gretton et al. (2012), *A Kernel Two-Sample Test*.
  - https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
- Rizzo and Szekely (2016), *Energy distance*.
  - https://real.mtak.hu/44201/
- Benjamini and Hochberg (1995), *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*.
  - https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Lakens (2017), *Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses*.
  - https://pubmed.ncbi.nlm.nih.gov/28736600/
- Oberkampf and Barone (2006), *Measures of agreement between computation and experiment: Validation metrics*.
  - https://www.sciencedirect.com/science/article/abs/pii/S0021999106001860
- ASA statement on p-values (summary of the 2016 statement).
  - https://www.stat.purdue.edu/news/2016/ASA_Pvalue1.html

## Bottom Line

The current gate system is good enough to keep using for model development.

But, before centering the next decision around `G6`, we should change the
project language from:

- `G6 proves indistinguishability`

to:

- `G6 is a conservative formal screen that did not detect mismatch at the
  configured sample size and permutation budget`

That keeps the pipeline scientifically honest and makes it much easier to
defend in a thesis.