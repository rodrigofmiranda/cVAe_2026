# S38-S40 Plan

This file is the concrete restart plan for the next architecture line after the
`S37` radial probe.

If we stop work and need to resume later, start from this document.

## Why This Plan Exists

The 2025 project in `/workspace/2025/vae-vlc-multicurrent` suggests a useful
lesson:

- the old model was not a strong digital twin by current standards;
- but it was very good at learning the **local conditional law**
  `p(y | x, d, c)` in a point-wise setting;
- that local point-conditioned modeling is the main idea worth transferring.

The goal of this plan is to import the useful part of 2025 into the 2026
pipeline without giving up the stronger protocol and audit standards.

## Core Hypothesis

The current seq-MDN line learns global/contextual behavior reasonably well, but
still smooths the border and misses the true residual shape in difficult
regimes.

The most promising transferable idea from 2025 is:

- add a **local conditional expert** that sees the center symbol explicitly;
- let that expert specialize in the local residual distribution;
- keep the 2026 evaluation protocol unchanged.

## What We Reuse From 2025

We reuse the following ideas, not the full 2025 training setup:

- point-wise conditioning on `x_center`
- explicit conditioning on `distance` and `current`
- heteroscedastic output head
- wide practical decoder variance range

We do **not** reuse:

- global random split as scientific validation
- weak protocol criteria
- purely mean-output analysis as proof of digital-twin quality

## Execution Order

Run the next work in this order:

1. `S38`: 2025-style point-wise baseline inside the 2026 protocol
2. `S39`: seq backbone + local heteroscedastic expert
3. `S40`: seq backbone + local MDN expert

This order matters because each step isolates a different question.

## S38: 2025-Style Point-Wise Baseline Under 2026 Protocol

### Goal

Answer a clean question:

- if we revive the old local conditional modeling style inside the current
  protocol, does the border behavior reappear?

### Scientific Question

Is the missing border fidelity mainly caused by:

- the sequence architecture; or
- the stricter protocol and data split; or
- both?

### Architecture

Create a point-wise branch that models:

- inputs: `x_center`, `distance`, `current`
- outputs: heteroscedastic Gaussian parameters for `y` or residual `delta`

Keep it simple and close to 2025:

- no GRU
- no window context
- no MDN in the first pass
- no radial specialization in the first pass

### Recommended Branch

- `feat/pointwise-2025-revival`

### Recommended Tag Prefix

- `S38`

### Likely Files To Touch

- `src/models/`
- `src/training/grid_plan.py`
- `src/evaluation/`

### Minimum Presets

- `pointwise_2025_revival_smoke`
- `pointwise_2025_revival_quick`
- `pointwise_2025_revival_full`

Current branch implementation status:

- canonical preset names now available in code:
  - `s38_pointwise_2025_smoke`
  - `s38_pointwise_2025_quick`
  - `s38_pointwise_2025_full`
- current branch-local continuation:
  - `s38_pointwise_small_local`
- first branch-local readings:
  - `S38 smoke` (`exp_20260402_165714`) gave `10/12`
  - `S38 quick` (`exp_20260402_234032`) gave `3/12`
  - `S38b` capped local sweep (`exp_20260404_150425`) gave `9/12`
  - `S38A` full confirmation (`exp_20260404_154011`) gave `9/12`
  - `S38bB` full confirmation (`exp_20260404_155322`) recovered `10/12`
  - current interpretation: the local point-wise idea is alive, but the
    larger 2025-like hyperparameter region is not; the viable standalone
    region is small and narrow

### Success Criteria

`S38` is useful if it does at least one of these:

- improves border visuals substantially at `0.8m`
- improves `G5` on `0.8m` relative to current seq baseline
- recovers local border spread without collapsing `G1-G4`

### Failure Criteria

Treat `S38` as negative if:

- it reproduces only the mean but not the border spread;
- or it cannot beat the current seq line even locally on `0.8m`;
- or it only looks good visually but fails `G5/G6` the same way.

Current status against those criteria:

- `S38` is not negative yet
- it already matched the current `10/12` ceiling with `S38 smoke` under caps
  and `S38bB` on full data
- however, it still fails `G5` on two `0.8m` regimes, so it should be treated
  as a live but incomplete standalone route

### Interpretation

- If `S38` helps: the local point-wise law is genuinely missing in the seq line.
- If `S38` does not help: the problem is not solved by reviving 2025-style
  conditioning alone.
- Current read:
  - the local point-wise law does help
  - but the standalone point-wise route still does not close the remaining
    border-shape gap by itself

## S39: Seq Backbone + Local Heteroscedastic Expert

### Goal

Keep the sequence backbone but add the local conditional mechanism from 2025.

### Scientific Question

Can a seq model improve the border if it delegates the final local residual law
to a point-wise expert?

### Architecture

Shared idea:

- seq backbone extracts contextual features from the window
- local head sees `x_center`, `distance`, `current`, and backbone context
- output is heteroscedastic Gaussian for local residual or local `y`

One acceptable formulation:

- seq encoder produces context vector `h_ctx`
- local expert receives `concat(h_ctx, x_center, d, c, optional_r)`
- local head predicts `mu/logvar`

### Recommended Branch

- `feat/seq-local-hetero-expert`

### Recommended Tag Prefix

- `S39`

### Likely Files To Touch

- `src/models/cvae_sequence.py`
- `src/models/losses.py`
- `src/training/grid_plan.py`
- `src/evaluation/report.py`

### Minimum Presets

- `seq_local_hetero_expert_smoke`
- `seq_local_hetero_expert_quick`
- `seq_local_hetero_expert_full`

### Success Criteria

`S39` is promising if:

- `G1-G4` stay strong
- `G5` improves on `0.8m`
- border spread visibly improves against `S36`/`S37`

### Failure Criteria

Treat `S39` as negative if:

- it behaves like the current seq model with no visible border gain;
- or it regresses signal fidelity without meaningful distribution benefit.

## S40: Seq Backbone + Local MDN Expert

### Goal

Move from local heteroscedastic Gaussian to a local mixture model that can
represent asymmetry, heavy tails, and multi-scale dispersion.

### Scientific Question

Is the remaining border failure caused by the local head being too Gaussian even
when it sees the right conditioning?

### Architecture

Shared backbone:

- same seq context extraction as `S39`

Local expert:

- receives `h_ctx`, `x_center`, `d`, `c`, optional `r`
- predicts local MDN parameters

This is the first step in the plan where we explicitly target:

- asymmetric local noise
- heavy tails
- multi-component local residual laws

### Recommended Branch

- `feat/seq-local-mdn-expert`

### Recommended Tag Prefix

- `S40`

### Likely Files To Touch

- `src/models/cvae_sequence.py`
- `src/models/losses.py`
- `src/training/grid_plan.py`
- `src/evaluation/validation_summary.py`

### Minimum Presets

- `seq_local_mdn_expert_smoke`
- `seq_local_mdn_expert_quick`
- `seq_local_mdn_expert_full`

### Success Criteria

`S40` is promising if:

- `G5` improves in `0.8m/100mA` and `0.8m/300mA`;
- border shape gets visibly closer to the real halo;
- `G6` does not collapse while `G5` improves.

### Failure Criteria

Treat `S40` as negative if:

- it adds complexity but does not improve `G5`;
- or it improves visuals while breaking protocol reliability elsewhere.

## Operational Rules

Use the same general discipline as the current branch:

- keep the 2026 protocol unchanged
- prefer quick -> full progression
- do not promote a visual-only improvement without protocol evidence
- preserve clear negative-result documentation

## Recommended Resume Point

If work stops here and resumes later:

1. start with `S38`
2. do not open `S39` before `S38` has a clean readout
3. do not open `S40` before `S39` confirms that the local-expert direction is
   scientifically useful

## Short Decision Tree

- `S38` good:
  - local conditional law matters
  - proceed to `S39`

- `S38` neutral:
  - local 2025 revival alone is insufficient
  - still consider `S39` only if seq context is believed to be necessary

- `S39` good but `G5` still limited:
  - proceed to `S40`

- `S39` bad:
  - local expert did not solve the problem
  - reassess before opening more complex local heads

## Current Summary

The next best step is not to keep tweaking the same seq-MDN head.

The best transfer from 2025 is:

- explicit local conditional modeling around `x_center`

The concrete execution path is:

- `S38` point-wise revival
- `S39` seq + local heteroscedastic expert
- `S40` seq + local MDN expert
