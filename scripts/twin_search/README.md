# Digital-twin search tooling

Scripts driving the parameter search for the best cVAE digital twin (full_square).
Full context: [`docs/TWIN_SEARCH_PROGRESS.md`](../../docs/TWIN_SEARCH_PROGRESS.md)
and [`docs/BEST_RUNS_INVENTORY.md`](../../docs/BEST_RUNS_INVENTORY.md).

- `inventory_best_runs.py` — scan output roots, rank good-basin runs by gates
  (the maintained "best runs" report). `python3 scripts/twin_search/inventory_best_runs.py`
- `seed_pin_driver.sh` + `inner_seed.sh` — deterministic good-seed search.
- `variant_driver.sh` + `inner_variant.sh` — run tail-fix variants (kurt/mdn/clamp)
  + the S38D base; deterministic, full-12 regimes, reduced train data
  (`--max_samples_per_exp 300000`), flatness-based early-kill.
- `parse_min.py` / `parse_min2.py` — running-min and flatness parsers used by the
  early-kill logic.
- `results.sh` — live dashboard of verdicts + gates.

> ⚠️ Provenance scripts with **host-specific absolute paths** (shared GPU box:
> eduardo repo, rodrigo clones, dataset mounted at the canonical `/workspace/...`
> path). Adapt paths before reuse. Determinism uses the env-var path only
> (`CVAE_DETERMINISTIC=1 CVAE_DET_OPDET=0 CVAE_DET_SETSEED=0`).
