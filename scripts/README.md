# Scripts Map

The canonical script layout is now split into four logical layers:

- `common/`
- `full_square/`
- `full_circle/`
- `benchmarks/modulations/`

Legacy folders remain available for backward compatibility:

- `ops/`
- `analysis/`
- `knowledge/`
- `archive/`

## Canonical entry layers

- `common/`
  - shared wrappers that should stay line-neutral
- `full_square/`
  - launchers and helpers specific to the `full_square` line
- `full_circle/`
  - launchers and helpers specific to the `full_circle` line
- `benchmarks/modulations/`
  - evaluation wrappers and helpers for `16QAM`, `4QAM`, and future modulations

## Compatibility note

Current operational wrappers in `ops/` and `analysis/` are still valid and are
reused by the new canonical subfolders when possible.

The canonical home for `16QAM` benchmark scripts is now
`scripts/benchmarks/modulations/16qam/`. The older
`scripts/analysis/run_eval_16qam_all_regimes.py` and
`scripts/analysis/plot_16qam_raw_constellations.py` paths remain available as
thin compatibility wrappers.
