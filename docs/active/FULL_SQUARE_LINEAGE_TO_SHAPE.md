# Full-Square Lineage To Shape

Date: 2026-04-20

## Purpose

This note reconstructs, in one place, how the project evolved from the
traditional point-wise cVAE framing to the later `shape` line, while keeping
the focus on the original `full_square` dataset.

The goal is not to retell every branch exhaustively. The goal is to establish
the minimum historical reading needed to restart `full_circle` from zero in a
clean order, beginning from the simplest hypotheses instead of jumping directly
to the most engineered support-aware variants.

## Scope

This document is about the `full_square` research line and the branches that
changed how the team understood the modeling problem.

It does **not** treat the later `full_circle` results as part of the same line.
Those belong to the separate `research/full-circle` worktree and should be read
as a new validation track, not as the default continuation of `full_square`.

## Branch-Level Map

The branch guide gives the high-level route:

| Branch / worktree | Main role in the story | Full-square reading |
| --- | --- | --- |
| `feat/seq-bigru-residual-cvae` | first major sequential residual cVAE line | moved the project beyond pure point-wise modeling into short-window temporal channel modeling |
| `feat/mdn-g5-recovery` | main MDN recovery line | established that the hard failure is concentrated in `0.8m`, especially `G5` shape mismatch |
| `research/mdn-return-20260416` | current coordinated return line | continued the MDN family with fast-path triage and cleaner documentation of the remaining ceiling |
| `feat/pointwise-2025-revival` | controlled replay of the useful 2025 local point-wise idea | showed that a narrow local point-wise law still has value under the 2026 protocol |
| `feat/probabilistic-shaping-nonlinearity` (`shape`) | support-geometry hypothesis line built on the same `full_square` data | re-read the problem as partly a support-geometry issue and introduced support-aware proxy interventions |

## Starting Point: Traditional cVAE Reading

The oldest useful baseline in this story is the traditional point-wise cVAE
reading of the problem:

- model the conditional law `p(y | x, d, c)` directly from synchronized IQ data
- use `full_square` as the standard acquisition geometry
- treat the central problem as distributional fidelity, not only mean mapping

Important baseline assumptions from the docs:

- `full_square` is a dense square support in the IQ plane, not QAM and not
  probabilistic shaping in the PAS/CCDM sense
- the original point-wise families were `concat`, `channel_residual`, and later
  `delta_residual`
- the project already knew that passing the protocol required more than low MSE:
  it required matching covariance, tails, and shape-oriented gates

What survived from this era:

- the point-wise law is scientifically meaningful
- residual-oriented modeling is more useful than naive direct-output decoding
- `full_square` is a broad probing dataset and should be interpreted as channel
  identification data, not as a communications-optimal signaling format

What did **not** survive:

- the idea that a simple point-wise decoder family would fully solve the hard
  short-distance shape failures by itself

## First Major Shift: Sequential Residual cVAE

The first major conceptual jump was from point-wise decoding to the sequential
family `seq_bigru_residual`.

Why that mattered:

- it acknowledged that the channel is not purely point-wise
- it inserted a short temporal window into the prior/encoder/decoder path
- it became the main line for the strongest `full_square` digital-twin runs

The practical reading from the branch guide and status docs is that this was
the first family the team treated as the main temporal baseline rather than a
side experiment.

What this changed scientifically:

- the project stopped treating the remaining gap as a generic cVAE weakness
- the remaining failures became localized enough to discuss by regime and by
  gate, especially around `0.8m`

## Second Major Shift: MDN Recovery On Full-Square

The `feat/mdn-g5-recovery` line is where the project learned the most about the
real bottleneck inside `full_square`.

Main findings from the active docs:

- the MDN family clearly beat the Gaussian decoder reference on the strict full
  protocol
- `lambda_coverage=0.25` in the `S27` family became the strongest known
  full-square result
- the best full-square anchor was:
  - `outputs/exp_20260328_153611`
  - champion: `S27cov_lc0p25_tail95_t0p03`
  - result: `10/12`

What that result meant:

- `G3` and `G6` were effectively recovered across all `12` regimes
- the remaining `2/12` gap was no longer broad failure
- it was concentrated at `0.8m/100mA` and `0.8m/300mA`
- those failures were `G5` failures, not general collapse

This is the critical historical simplification:

- the full-square problem is **not** “the model cannot learn the channel at all”
- the full-square problem became “the model still misses a specific low-current,
  short-distance residual shape mismatch”

## What The MDN Line Ruled Out

The MDN branch is important not only for what worked, but for what it closed.

According to the active notes, the following directions were already treated as
negative, exhausted, or non-central for the `full_square` bottleneck:

- stronger old Gaussian loss
- `sample-aware MMD`
- the current conditional-flow decoder implementation
- pure regime-weighted resampling
- simply increasing MDN components globally
- isolated kurtosis pressure via `lambda_kurt`
- broad `cond_embed` as a general search direction

The strong conclusion was this:

- the unresolved `full_square` gap behaved like a **shape** problem near the
  support extremes, not merely a covariance problem or a missing-capacity
  problem

## MDN Return: What Changed And What Did Not

The current `research/mdn-return-20260416` worktree did not restart the science
from zero. It clarified and operationalized the same `full_square` line.

What it added:

- fast-path triage (`S35`, `S36`) to continue the MDN family more efficiently
- clearer documentation of the anchor hierarchy
- explicit closure that broad `cond_embed` had already been over-explored

What it did **not** change:

- the remaining ceiling was still the same `0.8m` low-current shape problem
- the line still lived inside `full_square`
- it did not yet validate `full_circle`

Important clarification for future audits:

- `S35` in this worktree was a `full_square` run, not a `full_circle` run
- therefore it belongs to the same historical line summarized here

## Point-Wise Revival: Why It Still Matters

The `feat/pointwise-2025-revival` line matters because it prevents an overly
simple historical reading such as “point-wise is dead, only the sequential MDN
line matters.”

That branch showed a narrower claim:

- large 2025-like point-wise revivals were negative
- but a **small local point-wise family** remained scientifically alive

Best branch-local full result:

- `outputs/exp_20260404_155322`
- champion: `S38bB_pw25_local_lat4_b0p01_fb0p0_lr0p0003_bs2048_anneal3_L32-64`
- result: `10/12`

Why this matters for the full-square story:

- it shows the project never fully escaped the possibility that the missing law
  is partly local and point-conditioned
- it weakens any claim that “more sequence + more MDN” is the only path worth
  testing
- it suggests the shape problem may be representational, not only temporal

This is one reason a future `full_circle` restart should begin with simple
baselines again instead of assuming the best `full_square` MDN stack is the
only legitimate starting point.

## Shape: The Support-Geometry Reinterpretation

The `shape` branch changed the question more than it changed the backbone.

Its key reframing was:

- the current `full_square` dataset is excellent for broad probing
- but the square corners may be creating the hardest and least regular part of
  the channel
- the residual mismatch seen in `0.8m` may therefore be partly a
  support-geometry problem, not only a model-family problem

This line did **not** begin by collecting real `full_circle` data.
It first used support-aware interventions on the existing `full_square` data as
proxy tests:

- `support_feature_mode`
- `support_weight_mode`
- `support_filter_mode`
- `disk_l2`
- `geom3`
- edge/corner upweighting

That is the correct reading of `shape`:

- not a new dataset yet
- not strict probabilistic shaping yet
- but a support-aware ablation program on top of the same `full_square` line

## What Shape Actually Learned On Full-Square

The surviving lesson from the `shape` docs is not “geometry solved it.”
The stronger lesson is narrower.

What the branch supported:

- support-aware pressure can help
- edge weighting was the first clearly useful support-aware intervention
- the model seems to benefit when training spends more budget on edge/corner
  regions of the square support

What the branch did **not** prove:

- that hard geometry priors are the scientifically correct final answer
- that filtering to a disk inside the square solves the real channel problem
- that the project should abandon `full_square` as the identification dataset

The current support hyperparameter reading is especially important:

- `edge_rinf_corner` weighting survived as the strongest practical idea
- `geom3` alone did not solve the problem
- hard filtering like `disk_l2` was useful diagnostically, but should be read
  as a probe rather than the main scientific answer

So the honest historical reading is:

- `shape` found **engineering leverage** inside `full_square`
- but it did not cleanly settle whether the true scientific gain should come
  from a new acquisition geometry or from a better model of the existing one

## Full-Square Evolution In One Sentence Per Stage

1. Traditional cVAE: learn `p(y | x, d, c)` point-wise on `full_square`.
2. Residual families: reparameterize the target because direct output modeling
   is not enough.
3. Sequential residual cVAE: add short temporal context because the channel is
   not purely point-wise.
4. MDN recovery: improve the conditional law enough to isolate the residual
   bottleneck to `0.8m` low-current `G5` shape mismatch.
5. Point-wise revival: show that a narrow local point-wise law is still alive,
   so the representation question is not closed.
6. Shape: reinterpret part of the remaining problem as support geometry on the
   same `full_square` dataset and use proxy support-aware interventions.

## What Must Carry Forward Into A Clean Full-Circle Restart

If the team restarts `full_circle` from zero, the historical lessons above
imply the following should carry forward.

Keep:

- the strict 2026 protocol logic
- the regime-level reading of failures instead of only aggregate score
- the distinction between covariance recovery and shape recovery
- the recognition that `0.8m` low-current is the decisive stress test
- the possibility that both sequence structure and local point-wise law matter

Do not carry forward as assumptions:

- that the winning `shape` proxy settings are already scientifically validated
  for a real disk acquisition
- that `full_circle` should start from hard geometry priors or filters
- that the best `full_square` MDN stack is automatically the only correct
  baseline for a new support geometry

## Recommended Clean Restart Order For Full-Circle

The clean restart should proceed from the simplest honest hypotheses to the
most engineered ones.

### Stage A: data and fairness first

1. Acquire or validate the real `full_circle` dataset under matched average
   power and matched acquisition conditions.
2. Compare bench diagnostics against `full_square` before changing the model.
3. Train the simplest defensible baselines with no support-aware tricks.

### Stage B: simplest model baselines

1. point-wise baseline on `full_circle`
2. residual point-wise baseline on `full_circle`
3. sequential residual baseline on `full_circle`

Purpose:

- answer whether the support change alone already regularizes the problem

### Stage C: strongest full-square families, but still clean

1. import the strongest clean MDN family without shape-specific priors
2. import the strongest narrow local point-wise family without shape-specific
   priors
3. compare them under the same `full_circle` protocol

Purpose:

- answer whether the new support geometry changes the architecture ranking

### Stage D: only then test geometry-aware help

1. soft support-aware features
2. soft weighting
3. only later, if still needed, stronger geometry priors or filters

Purpose:

- keep the scientific interpretation clean
- avoid confusing “better because the acquisition is better” with “better
  because we reintroduced an engineered bias”

## Bottom-Line Historical Reading

The project did not move from traditional cVAE to `shape` because the earlier
lines were useless.

It moved that way because the `full_square` line became precise enough to show
the remaining error was:

- localized
- shape-sensitive
- edge-sensitive
- and plausibly tied to the support geometry itself

So `shape` should be read as the last major `full_square` reinterpretation
before the honest scientific move to a real `full_circle` dataset.

That is why the right next step for `full_circle` is not to continue from the
most complex support-aware proxy. It is to restart from the simplest matched
baselines and rebuild the ladder in a clean order.