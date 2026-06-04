# Onboarding a new dataset version (e.g. the 5-LED set)

The pipeline is dataset-portable: `--dataset_root` is a CLI arg, experiment
discovery is metadata-driven (`discover_experiments` rglobs `IQ_data/` +
`metadata.json`; no rigid directory naming), and regimes are matched by
`distance_m`/`current_mA` with tolerance. It has already run on multiple datasets
(full_square, full_circle, shape, 16qam).

## Dataset versions

- **V2 2026 (current):** 3 LEDs + 1 half-burnt. `dataset_fullsquare_organized`,
  27 regimes (3 distances × 9 currents); the protocol uses a 12-regime subset.
- **5-LED (incoming):** 5 healthy LEDs — a different physical channel.

> Hypothesis to keep in mind: the V2 half-burnt LED likely contributes the
> asymmetry/nonlinearity behind the hard 0.8 m non-Gaussian **tail** behaviour
> (the G5 failures at 0.8 m/300-500 mA). A healthy 5-LED setup may be **cleaner /
> easier to model** (better gates) — or fail in a *different* regime. Do not assume
> the 5-LED bottleneck is the same 0.8 m/300 mA.

## Required format (so it "just works")

Per experiment:
- `<exp>/IQ_data/X.npy` (sent) + `Y.npy` (received) — 2026 format.
- `<exp>/metadata.json` with `distance_m` (or `dist_m`), `curr_mA` (or
  `current_mA`), ideally `regime_id`.
- Same I/Q normalization/representation as V2 (otherwise the `val_recon` scale
  shifts → recalibrate thresholds; if very different, results aren't directly
  comparable).
- If the data arrives raw (`.c64`), convert to `X.npy`/`Y.npy` first (that step
  lives in the acquisition pipeline, not this repo).

## Option A — separate 5-LED twin (DO THIS FIRST)

Train a fresh twin on the 5-LED data alone. **No code/architecture change.**

1. Place the 5-LED tree somewhere mountable, e.g. `data/dataset_5led_organized/`.
2. Auto-generate its protocol (no hand-editing of absolute paths):
   ```bash
   python3 scripts/twin_search/make_protocol_from_dataset.py \
       data/dataset_5led_organized configs/protocol_5led.json
   ```
3. **Recalibrate the good-basin threshold** (the V2 `-4.3/-4.6` is scale-specific):
   run one deterministic baseline (`twin_base`) on the 5-LED data, look at where
   the good vs bad basin settle, and set the cutoff in the gap. Methodology
   (inventory + flatness early-kill) transfers; the numbers don't.
4. Run the search pointed at the new dataset: same `variant_driver.sh` but with
   `--dataset_root data/dataset_5led_organized`, `--protocol configs/protocol_5led.json`,
   the new threshold, and the mount pointing at the 5-LED data. If the 5-LED
   `regime_id`s differ from `dist_0p8m__curr_*`, update the grid resample-weight
   keys accordingly.
5. Build the 5-LED best-runs inventory (`inventory_best_runs.py` with the 5-LED
   root) and compare gates.

## Option B — unified twin conditioned on LED count (FUTURE PLAN)

> **Explicit future work — not for now.** After Option A produces a good 5-LED
> twin, decide whether a single model is worth it.

Goal: one cVAE `p(y | x, d, c, n_leds)` covering V2-3LED **and** 5-LED, by adding
**LED count as a 3rd conditioning variable**. Requires:
- data labelling: add `n_leds` (and ideally a "healthy/burnt" flag) to every
  experiment's `metadata.json`;
- model change: extend the conditioning input (cond-embedding) from `(d, c)` to
  `(d, c, n_leds)` in `src/models/cvae_sequence.py` (+ schema/runtime plumbing);
- training across the merged dataset; gates evaluated per (regime × LED setup).

**Cross-comparison to justify B:** compare the **specialized** Option-A twins
(one per LED setup) against the **unified** Option-B twin on each setup's gates.
Adopt B only if the unified model matches/beats the specialists (i.e. the setups
share enough structure that joint conditioning helps rather than dilutes). If the
half-burnt-LED hypothesis holds, the two channels may be too different and
specialists win — that is itself the answer.

See also `TWIN_SEARCH_PROGRESS.md`.
