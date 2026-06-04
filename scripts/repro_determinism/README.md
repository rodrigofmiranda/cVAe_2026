# Reproducibility & determinism tooling (run 141943 / S39B)

Scripts used to investigate the reproduction of `exp_20260427_141943`
(cVAE S39B, `grid_tag=S39B_edgegap_lowlr_all08_w18_p120`) and to demonstrate
deterministic training. See `docs/REPRODUCIBILITY_DETERMINISM_141943.md` for the
full write-up.

> ⚠️ These are provenance scripts with **host-specific absolute paths** (the
> shared GPU box: eduardo repo at `/home/eduardo/cVAe_2026`, this clone at
> `/home/rodrigo/cvae_det`, dataset mounted into the container at the canonical
> `/workspace/...` path). Adapt the paths before reuse elsewhere.

## What each script does

- `demo_driver.sh` — runs 4 short trainings (2 non-deterministic + 2
  deterministic, same `seed=42`) then calls `compare.py` to test bit-identity
  of the per-epoch `val_recon_loss`.
- `demo_inner.sh` — the in-container command for one demo run.
- `compare.py` — loads `val_recon_loss` from the 4 runs and reports whether the
  non-deterministic twins differ and the deterministic twins are identical.
- `det_launch.sh` / `det_inner.sh` — launch one deterministic training.
- `det_iso_launch.sh` / `det_iso.sh` — isolation harness to test each
  determinism knob (env vars / `enable_op_determinism` / `set_random_seed`)
  independently.

## The determinism switch

The code change lives in `src/protocol/run.py` and `src/training/pipeline.py`,
gated by env vars (off by default):

- `CVAE_DETERMINISTIC=1` — master switch.
- `CVAE_DET_ENV=1` (default) — set `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`
  before TF import. **This is the working determinism path on this stack.**
- `CVAE_DET_OPDET=1` (default) — also call `tf.config.experimental.enable_op_determinism()`.
  **Known to break this model** (raises `'float' object cannot be interpreted as
  an integer` at model build under TF 2.17 + tf_keras). Set `CVAE_DET_OPDET=0`.
- `CVAE_DET_SETSEED=1` (default) — also call `tf.keras.utils.set_random_seed()`.

Recommended deterministic invocation on this stack:
`CVAE_DETERMINISTIC=1 CVAE_DET_OPDET=0 CVAE_DET_SETSEED=0` (env vars + the
existing `tf.random.set_seed(seed)`).
