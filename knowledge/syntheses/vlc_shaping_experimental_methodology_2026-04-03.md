# VLC Shaping Experimental Methodology

Date: 2026-04-03

## Purpose

This document defines the experimental methodology for the current
probabilistic-shaping-inspired investigation in the VLC bench.

The goal is not yet to validate probabilistic shaping in the strict PAS/CCDM
sense. The immediate goal is to determine whether a support geometry better
matched to the physical channel can improve:

- bench-level transmission behavior
- digital-twin fidelity
- the quality of future learned constellations and demodulators

## Research Questions

Primary question:

- for fixed average transmit power, does replacing the current `full_square`
  excitation by a matched `full_circle` excitation improve both physical
  transmission quality and digital-twin learnability?

Secondary question:

- if `full_circle` helps, is the gain due to better channel matching, or only
  because the experiment avoids the hardest square-corner samples?

## Experimental System

The physical link under study is:

- `USRP TX -> Bias-T -> LED -> free-space optical path -> photodetector/TIA -> USRP RX`

The model-facing dataset is not built directly from the oscilloscope.
Instead, it follows the repository pipeline:

- GNU Radio transmission and reception generate `*_sent.c64` and `*_recv.c64`
- post-processing applies startup skip, lag correction, global phase
  correction, and anchor normalization
- the final supervised arrays are `X.npy` and `Y.npy`

The oscilloscope CSVs are used as auxiliary physical diagnostics.

## Experimental Factors

### Controlled Factors

- symbol rate
- `sps`
- pulse-shaping configuration
- acquisition duration
- current and distance regimes
- post-processing pipeline
- train/validation/test protocol

### Main Factor Of Interest

- input support geometry:
  - `full_square`
  - `full_circle`

### Future Factor

- learned signaling geometry after digital-twin validation

## Baseline And Candidate Geometries

### Baseline: `full_square`

The current GNU Radio source generates random `int16` pairs that are scaled
into a dense square support in the IQ plane.

This baseline should be interpreted as:

- a broad channel-probing excitation
- a support that includes center, side-edge, and corner samples
- a geometry that stresses high-energy operating regions

It should not be interpreted as:

- Gaussian signaling
- discrete QAM
- probabilistic shaping in the PAS/CCDM sense

### Candidate: `full_circle`

The candidate acquisition will generate samples uniformly over a disk in the IQ
plane while preserving the same average transmit power used in the square
comparison.

For the virtual disk experiment derived from the current `full_square` data,
the matched-power radius is:

- `R = a_train * sqrt(4/3)`

This follows from the uniform-support assumption used by the current source:

- square baseline on `[-a_train, a_train]^2`: `E[r^2] = 2 a_train^2 / 3`
- disk baseline of radius `R`: `E[r^2] = R^2 / 2`
- matched average power therefore implies `R^2 = 4 a_train^2 / 3`

This equality is exact for a continuous uniform square and remains a close
approximation for the current dense `int16` excitation.

The candidate is motivated by two expected benefits:

- lower peak stress due to removal of square corners
- smoother and more isotropic support boundary for the digital twin

## Fairness Constraints

The comparison between `full_square` and `full_circle` must keep the following
fixed:

- same average transmit power
- same symbol rate
- same `sps`
- same acquisition duration
- same current and distance regimes
- same pulse-shaping and RF chain settings
- same preprocessing and normalization pipeline
- same model family and training budget when comparing digital twins

If these constraints are not preserved, any observed gain becomes ambiguous.

## Data Acquisition Procedure

For each regime of interest:

1. Acquire a `full_square` dataset with the standard GNU Radio chain.
2. Acquire a `full_circle` dataset under the same physical conditions.
3. Store raw GNU Radio outputs and metadata using the existing dataset
   organization pattern.
4. When available, collect oscilloscope CSVs for the same regime to inspect
   asymmetry, harmonic content, and baseband reconstruction quality.

Priority regimes:

- `0.8m / 100mA`
- `0.8m / 300mA`

These are the regimes where the current repo history already indicates the most
pronounced center-to-edge gap.

## Preprocessing Procedure

All datasets must pass through the same post-processing path:

1. startup skip
2. lag estimation and synchronization
3. global phase correction
4. anchor normalization
5. supervised array generation into `X.npy` and `Y.npy`

No geometry-specific post-processing should be introduced before the first fair
comparison.

## Physical Diagnostics

Bench-level comparison should include:

- peak positive excursion
- peak negative excursion
- asymmetry ratio
- crest factor or equivalent peak-stress indicator
- occupied bandwidth
- baseband reconstruction correlation
- baseband reconstruction NMSE
- visible harmonic content near `2f` and higher components when available

These diagnostics help determine whether a geometry is physically easier for
the link, not only easier for the model.

## Dataset Diagnostics

For each regime and each support geometry, compute:

- input and output radial statistics
- residual mean and covariance
- residual skewness
- residual kurtosis
- Jarque-Bera or equivalent normality-distance measure
- center-vs-edge mismatch
- amplitude-conditioned diagnostics
- radius-conditioned diagnostics

The purpose is to test whether the circular support reduces the extreme
structure that the current model family fails to reproduce.

## Digital-Twin Evaluation

Each support geometry should be used to train the same digital-twin pipeline
under matched conditions.

The comparison should report:

- training and validation losses
- residual-shape fidelity metrics already adopted in the repo
- edge-focused diagnostics
- regime-specific outcomes
- stability across seeds when feasible

The key test is whether `full_circle` reduces the observed edge gap without
simply collapsing useful structure.

For the support-aware ablation battery on the existing `full_square` dataset,
the primary controlled comparison should not rely only on the stage champion.
The preferred reading is:

- matched-family runs (`D3` vs `D3_*`, `S27` vs `S27_*`)
- at least 3 seeds per comparison when claiming an effect
- explicit inspection of `support_region = corner` in addition to the global
  protocol gates, especially for `disk_l2`

## Decision Criteria

`full_circle` should become the preferred support for future signaling
optimization only if it improves all of the following in a meaningful way:

- physical diagnostics on the bench
- residual-shape fidelity in the digital twin
- downstream learned-signaling behavior under fair constraints

If `full_circle` improves only model fit but appears to exclude important
operating regions, it should be treated as a complementary support-specific
dataset rather than a full replacement.

If no meaningful improvement appears, the project should prioritize:

- model architecture
- conditioning strategy
- loss design
- sequence-aware channel representation

instead of changing support geometry.

## Role In The Larger Research Program

This methodology is a bridge between three stages:

Stage 1: channel identification

- learn the physical channel with controlled support geometries

Stage 2: support selection

- decide whether the hardware prefers a smoother operating region

Stage 3: constrained learned modulation

- train an autoencoder or related learned signaling system on the stronger
  digital twin, under explicit power and plausibility constraints

The expected contribution is not merely to show that one geometry fits better.
The intended contribution is to show how physical support geometry, channel
nonlinearity, and digital-twin learnability interact in a VLC bench intended
for learned communication design.
