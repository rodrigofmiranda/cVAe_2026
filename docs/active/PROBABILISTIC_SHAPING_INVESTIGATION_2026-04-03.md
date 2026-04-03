# Probabilistic Shaping Investigation

Date: 2026-04-03

Branch: `feat/probabilistic-shaping-nonlinearity`

## Why This Note Exists

This note consolidates the current investigation around:

- the optical probabilistic shaping literature,
- the actual GNU Radio acquisition used in this repository,
- the current `full_square` dataset geometry,
- the observed modeling failure at the constellation edges,
- the hypothesis that a circular excitation region may be easier both for the
  physical channel and for the digital twin.

It is intended as a working bridge between:

- paper reading,
- bench diagnostics,
- dataset design,
- digital twin modeling,
- and future constellation learning with autoencoders.

Related methodology document:

- `knowledge/syntheses/vlc_shaping_experimental_methodology_2026-04-03.md`

## Current Understanding Of The Bench

The active acquisition path is already documented in
`docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt`, but the key
points for this investigation are:

- `USRP TX -> Bias-T -> LED -> free-space optical path -> photodetector/TIA -> USRP RX`
- the model-facing dataset is born from GNU Radio IQ streams, not directly from
  oscilloscope CSVs
- the saved GNU Radio files are `*_sent.c64` and `*_recv.c64`
- post-processing applies startup skip, lag correction, global phase
  correction, and anchor normalization before producing `X.npy` and `Y.npy`

Important consequence:

- the model learns from aligned and globally normalized `X.npy` / `Y.npy`
- the oscilloscope CSVs remain valuable for physical diagnosis of clipping,
  asymmetry, harmonic content, and envelope effects that are less explicit in
  the model-ready arrays

## What `channel_dataset.py` Really Sends

The current GNU Radio generator lives in `src/data/channel_dataset.py`.

The transmit source is:

- random `int16` values in `[-32767, +32767]`
- converted to complex through `interleaved_short_to_complex(..., 32767)`
- therefore mapped approximately to continuous `I` and `Q` values in `[-1, 1]`

This means the current `full_square` dataset is:

- not Gaussian signaling,
- not discrete QAM,
- not probabilistic shaping in the sense of PAS/CCDM/ESS,
- but a dense, almost continuous excitation of a square support in the IQ plane

Operational interpretation:

- this is excellent for channel identification
- it is not yet a direct hardware validation of probabilistic shaping

## What The Repo History Already Suggests

The current active branch family has repeatedly reported a persistent
center-to-edge mismatch.

From `docs/active/WORKING_STATE.md`:

- the model captures the bulk of the residual distribution reasonably well
- the predicted residual cloud is systematically more uniform / spherical
- the real data keeps additional structure at the extremes
- the hardest unresolved regimes remain `0.8m / 100mA` and `0.8m / 300mA`
- these failures behave like a shape problem, not only a covariance problem

This project therefore already has a strong prior that the current modeling
family struggles with edge structure.

The grid plan also reflects this explicitly via the `edgegap` preset:

- `src/training/grid_plan.py`
- preset `_preset_seq_cond_embed_fast_stage4_edgegap()`

## Why The Square Support May Be Difficult

The present `full_square` source excites:

- center samples,
- edge samples along one axis,
- and corner samples where both axes are extreme simultaneously

This creates two different kinds of "boundary":

- side boundary: one coordinate is extreme
- corner boundary: both coordinates are extreme

For a real VLC channel this can matter because:

- corners produce larger instantaneous energy than interior points
- high-amplitude regions are more exposed to LED / driver / TIA nonlinearities
- the bench may respond differently to equal radius but different angle
- the resulting conditional channel `p(y|x)` becomes more anisotropic and
  harder to model with the current families

## Why A Circular Support Could Help

The current working hypothesis is that replacing the square support by a disk
with matched average power could help for two reasons.

Reason 1: physical simplification

- a circle removes the corners, which are likely the hardest part of the
  hardware operating region
- for equal average power, a disk has lower peak energy than a square support
- lower peak stress should reduce compression, asymmetry, and edge distortion

Reason 2: modeling simplification

- a circle has a smoother single boundary `r = const`
- the channel may become more regular as a function of radius
- the digital twin may need to learn less angle-specific extreme behavior
- this may reduce the current "uniformized edge" failure mode

Important caution:

- if `full_circle` improves the model, this does not automatically mean the
  model became better at the full channel
- it may also mean that we removed the hardest part of the support

Therefore `full_circle` should first be treated as a controlled hypothesis
test, not as an unquestioned replacement for `full_square`

## Papers Read For This Direction

Two papers were read in full and summarized into research cards:

- `knowledge/notes/shu_2020_probabilistic_shaping_optical.md`
- `knowledge/notes/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance.md`

The short version is:

- Shu and Zhang (2020) emphasize that practical optimality depends on laser
  linewidth, PAPR, and converter resolution, not only shaping gain
- Askari and Lampe (2025) move the discussion from "best marginal
  distribution for AWGN" to "best shaping strategy under fiber/channel
  nonlinearity, memory, and sequence effects"

For this repo, the most useful lesson is:

- the best signaling geometry is a constrained channel-matching problem
- not a generic "maximize energy" or "be closer to Gaussian" problem

## Oscilloscope Audit Added During This Session

Server-local folder analyzed:

- `/home/rodrigo/Aquisições`

Main files inspected:

- `analysis/scope_summary.csv`
- `analysis/baseband/baseband_summary.csv`
- `100mA_000.csv`
- `500mA_000.csv`
- `900mA_000.csv`

What these measurements add:

- the analog waveform remains centered near `1 MHz`, consistent with the bench
- baseband occupancy is around `46-50 kHz` (`OBW90`) and `56-59 kHz` (`OBW99`),
  consistent with `50 ksym/s` plus pulse shaping
- low-current operation is visibly more asymmetric in voltage excursion
- higher current improves the recovered baseband correlation and NMSE
- a component around `2 MHz` is visible, suggesting nonlinear harmonic content

Observed tendency from the summary tables:

- `100mA`: most asymmetric and worst baseband agreement
- `500mA`: intermediate
- `900mA`: best baseband agreement among the three inspected files

Working interpretation:

- the hardest regimes are not only noisier
- they are structurally more distorted
- this strengthens the hypothesis that the edge/corner regions of
  `full_square` are physically difficult, not only statistically difficult

## Current Research Position

At this point the project should distinguish three different goals.

Goal A: learn the channel well

- `full_square` remains valuable because it probes a wide support

Goal B: find a support geometry that better matches the hardware

- `full_circle` is now a serious hypothesis

Goal C: learn better constellations / demodulators

- this should happen after a stronger digital twin exists
- ideally with explicit constraints on power, PAPR, and physical plausibility

## Proposed Scientific Framing

Primary question:

- for fixed average transmit power, does replacing a uniform square IQ support
  by a uniform circular support improve both physical transmission quality and
  digital twin learnability?

Secondary question:

- if the circular support helps, does it do so because it better matches the
  physically useful region of the channel, or only because it removes the most
  difficult corner samples?

## Formal Research Frame

### General Objective

- determine whether a support geometry better matched to the VLC bench can
  improve digital-twin fidelity and create a stronger base for learned
  modulation and demodulation

### Specific Objectives

- characterize how the current `full_square` acquisition excites the channel
  and where it fails physically and statistically
- generate a fair `full_circle` alternative with matched average power and
  matched acquisition conditions
- compare `full_square` and `full_circle` at the bench, dataset, and
  digital-twin levels
- determine whether the gain from `full_circle` reflects better
  channel-matching or only avoidance of difficult corner samples
- decide which support geometry should be used as the base distribution for
  future constrained autoencoder-based constellation learning

### Main Hypothesis

- `H1`: for fixed average transmit power, a circular IQ support improves both
  physical transmission quality and digital-twin fit because it reduces corner
  stress, peak-related distortion, and angle-specific boundary effects present
  in `full_square`

### Null Hypothesis

- `H0`: changing the support from square to circle does not materially improve
  the physical channel behavior or the learned twin; the current failure is
  mainly a modeling-capacity or loss-design issue

### Mechanistic Hypotheses

- `H2`: the circular support lowers peak stress and therefore reduces
  asymmetry, compression, and harmonic distortion at the bench
- `H3`: the circular support makes the conditional channel more regular as a
  function of radius, reducing the current center-to-edge mismatch
- `H4`: if `full_circle` improves fit but weakens downstream robustness, then
  it is hiding relevant operating regions rather than solving the channel
  modeling problem

### Research Contribution Statement

- the expected contribution is not simply "circle beats square"
- the contribution is to show how support geometry, physical channel behavior,
  and digital-twin learnability interact in a VLC bench that is later used for
  learned constellation design

## Evaluation Metrics And Fairness Constraints

### Fair Comparison Constraints

- same symbol rate
- same `sps`
- same acquisition duration
- same current and distance regimes
- same alignment and normalization pipeline
- same average transmit power
- same train/validation/test protocol for the digital twin

### Physical Metrics

- peak positive and negative excursion
- asymmetry ratio
- crest factor / effective peak stress
- baseband reconstruction correlation
- baseband reconstruction NMSE
- occupied bandwidth
- harmonic content near `2f` and higher visible components

### Dataset Metrics

- residual mean and covariance
- residual skewness and kurtosis
- Jarque-Bera or equivalent normality distance
- center-vs-edge residual mismatch
- amplitude-conditioned and radius-conditioned diagnostics

### Model Metrics

- validation loss under the current training setup
- residual-shape fidelity metrics already used by the project
- edge-focused diagnostics
- regime-specific performance, with priority on `0.8m / 100mA` and
  `0.8m / 300mA`
- stability of performance across seeds and runs

### Downstream Metrics For Learned Signaling

- average-power-constrained reconstruction quality
- decision-region separability on the twin
- robustness when reinjected into the physical bench
- fair comparison against traditional baselines such as `16QAM`

## Decision Logic

- if `full_circle` improves physical diagnostics and digital-twin fidelity
  without harming downstream learned signaling, it becomes the preferred base
  support for future constellation learning
- if it improves fit but only by excluding difficult regions, then it should be
  kept as a complementary dataset rather than a full replacement
- if it does not improve the key metrics, the next effort should focus on model
  family, conditioning, and loss design rather than on support geometry

## Proposed Experimental Plan

1. Generate a `full_circle` acquisition dataset with the same:

- symbol rate
- `sps`
- acquisition duration
- distance/current regimes
- alignment and normalization pipeline

2. Compare `full_square` vs `full_circle` at three levels.

Physical level:

- voltage asymmetry
- crest factor / peak excursion
- baseband reconstruction quality
- harmonic content near `2f`

Dataset level:

- residual skew / kurtosis / JB
- center-vs-edge mismatch
- amplitude- or radius-conditioned residual diagnostics

Model level:

- residual overlay quality
- G5 / shape metrics
- edge-focused diagnostics
- especially `0.8m / 100mA` and `0.8m / 300mA`

3. Only after this, decide which source should become the default digital twin
training support.

## Implication For The Autoencoder Goal

The long-term goal is not to prove that circles are always best.

The long-term goal is:

- train a digital twin that is faithful enough to the real channel
- then optimize signaling geometry and demodulation jointly
- then validate the learned constellation back on the bench

If `full_circle` wins, that suggests the hardware prefers smoother,
lower-peak support regions.

If `full_square` remains necessary, that suggests the autoencoder must learn
under stronger edge/corner constraints rather than avoiding them.

## Immediate Next Actions

- implement a `full_circle` version of the GNU Radio source
- preserve fair average-power normalization relative to `full_square`
- add radius-conditioned diagnostics in evaluation
- compare digital twin performance between `full_square` and `full_circle`
- only then move to learned constellations / autoencoder-based signaling
