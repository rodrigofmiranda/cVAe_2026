# VLC Probabilistic Shaping Strategy

Date: 2026-04-03

## Objective

Use the current VLC bench and digital twin workflow to eventually learn better
constellations and demodulation strategies than the traditional baselines.

The immediate challenge is that the current dataset geometry and the current
 model family appear to disagree most strongly at the edges of the support.

## Present Acquisition Geometry

The current default acquisition is `full_square`:

- `I` and `Q` are generated independently from a uniform discrete source
- after scaling, the transmitted support is approximately a dense square in
  `[-1, 1] x [-1, 1]`

This should be read as:

- a broad channel-probing source
- not as probabilistic shaping
- not as QAM
- not as Gaussian signaling

## Why This Geometry Is Useful

`full_square` is good for digital-twin identification because it probes many
parts of the input support, including difficult high-energy regions.

This makes it valuable for:

- learning the channel broadly
- exposing nonlinear edge behavior
- stress-testing the model family

## Why This Geometry May Also Be Causing Difficulty

The support includes corners where both `I` and `Q` are extreme.

Those corners:

- have higher instantaneous energy than most of the support
- are plausible candidates for stronger LED / driver / detector nonlinearity
- create angle-dependent edge behavior
- make the conditional channel less regular

Repo evidence is consistent with this:

- current models reproduce the bulk better than the extremes
- the predicted residual cloud is systematically more uniform / spherical
- the real data shows additional structure at the edges

## Square Versus Circle Working Hypothesis

For equal average power, a circular support should:

- remove the square corners
- reduce peak stress / PAPR
- smooth the boundary geometry
- make the support more isotropic
- likely improve both physical transmission quality and digital-twin fit

However, that gain may come from either:

- better matching the physically useful region of the channel
- or simply avoiding the hardest operating region

This distinction must be measured experimentally.

## What The Papers Change In Our Framing

Shu and Zhang (2020):

- practical optimality depends on more than AWGN shaping gain
- geometry and probability should be matched to hardware constraints

Askari and Lampe (2025):

- optimal shaping in a real channel depends on moments, memory, mapping, CPR,
  and sequence-level effects
- a future digital twin should ideally represent not just pointwise distortions
  but also temporal structure

Combined implication for this repo:

- learn the channel first
- then optimize signaling under explicit physical constraints
- do not confuse "easier to fit" with "better for transmission" without bench
  validation

## Research Question And Hypotheses

Primary research question:

- for fixed average transmit power, does replacing the current uniform square
  support by a matched circular support improve both the physical behavior of
  the VLC link and the learnability of the digital twin?

Secondary research question:

- if the circular support helps, is the gain due to better alignment with the
  physically useful operating region, or only because the experiment avoids the
  hardest square-corner samples?

Working hypotheses:

- `H1`: `full_circle` improves transmission and modeling because it reduces
  corner-driven peak stress and makes the channel more regular
- `H0`: support geometry is not the main bottleneck; the dominant problem is
  still the current model family or loss design
- `H2`: any real improvement must survive bench diagnostics, digital-twin
  metrics, and downstream learned-constellation evaluation under fair power
  constraints

## Recommended Research Path

Phase 1: channel identification

- keep `full_square` as the broad probing dataset
- add a matched `full_circle` dataset
- compare their physical and modeling behavior directly

Phase 2: geometry decision

- decide whether `full_circle` should replace `full_square` as the default twin
  source, or whether both should coexist

Phase 3: constrained constellation learning

- train an autoencoder or learned modulation system on the stronger twin
- enforce average-power and peak-related constraints
- compare against traditional baselines such as `16QAM`

Phase 4: bench validation

- re-inject the learned constellation into GNU Radio
- measure whether the expected gain survives on the physical setup

## Evaluation Logic

The comparison must keep symbol rate, `sps`, acquisition duration, regime set,
normalization, and average transmit power fixed.

The decision should use three layers of evidence:

- physical metrics: asymmetry, peak stress, reconstruction correlation / NMSE,
  and harmonic content
- digital-twin metrics: residual-shape fidelity, edge-focused diagnostics, and
  regime-specific performance
- downstream metrics: whether the stronger twin actually supports learned
  signaling that survives on the bench

## Decision Rule

Adopt `full_circle` as the preferred support only if it improves:

- physical diagnostics on the bench,
- residual-shape fidelity in the digital twin,
- and downstream learned-constellation performance under fair constraints

Otherwise:

- keep `full_square` as the main identification source
- and treat `full_circle` as an auxiliary support-specific experiment
