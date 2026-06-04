# Digital-twin search — progress log

Living status of the work that started as "reproduce run 141943" and became
"find the best cVAE digital twin (full_square)". Companion docs:
[REPRODUCIBILITY_DETERMINISM_141943.md](REPRODUCIBILITY_DETERMINISM_141943.md),
[BEST_RUNS_INVENTORY.md](BEST_RUNS_INVENTORY.md).

## Arc so far

### Phase 1 — Reproduce eduardo's run `exp_20260427_141943` (S39B, 10/12)
- Config/code/data/packages/architecture/clamp verified **byte-identical** to the
  known-good runs (full src tree md5, dataset md5 over 12 regimes, TF/keras/numpy,
  decoder logvar clamp). The only difference between a "good" and "bad" outcome is
  the trained **weights**.
- The training landscape is **bimodal**: good basin `val_recon ≈ -4.97` (~8-10/12)
  vs bad basin `≈ -3.9` (0/12), a ~25-30 % stochastic draw. Exact reproduction of
  that specific run is **impossible** post-hoc (it was non-deterministic and the
  draw was never pinned). Accepted.

### Phase 2 — Determinism (so future runs are repeatable)
- Added an opt-in switch (`CVAE_DETERMINISTIC=1`) in `src/protocol/run.py` +
  `src/training/pipeline.py`. **Working path = TF env vars**
  (`TF_DETERMINISTIC_OPS`, `TF_CUDNN_DETERMINISTIC`) + the existing
  `tf.random.set_seed`. `tf.config.experimental.enable_op_determinism()` **breaks
  this stack** (TF 2.17 + tf_keras: `'float' object cannot be interpreted as an
  integer` at build) → keep `CVAE_DET_OPDET=0`.
- Proven bit-identical twins (same seed, determinism on → identical per-epoch
  val_recon; off → diverge from epoch 1). Cost: only ~10 % slower (the model
  already avoids the cuDNN GRU via `seq_gru_backend=compat`).

### Phase 3 — Digital-twin parameter search (current)
Goal: beat / match **10/12** by fixing the specific weak gate, not chasing physics.
Decisions: optimize **gates (mini_protocol_v1)**; **deterministic** runs;
**full 12-regime** eval; reduce **training** data for speed.

- **Speed lever:** `--max_samples_per_exp 300000` caps the **train** head contiguously
  (seq-safe) and leaves **validation untouched** → ~2× faster (21 vs 42 s/epoch),
  gates still computed on full val. Validate the final winner on full data.
- **Target:** eduardo's 2 failures are both **G5 at 0.8 m** — 300 mA is a kurtosis
  **OVERSHOOT** (cVAE tails far heavier than the real channel; jb -29 vs -4.6),
  500 mA is marginal. (The G5 gate is *relative* — it does NOT punish the channel's
  normal non-Gaussianity; it fails only when the model mismatches it.)
- **Variants (one axis off S39B), added as grid tags in `grid_plan.py`:**
  `S39B_tailfix_kurt05` (lambda_kurt=0.5), `_kurt10` (1.0), `_mdn5`
  (mdn_components=5), a clamp run (`CVAE_DECODER_LOGVAR_CLAMP_HI=-1.2`), plus the
  alternative base **S38D** (9/12, sampled-MMD + cov30).

## Methodology notes (calibrated from 15 good + ~25 bad runs)

- Good-basin cutoff `val_recon <= -4.3` (final); the bimodal gap is clean
  ([-3.98, -4.6]) so this is safe with margin.
- **Early-kill = flat-plateau only.** A run is killed *only* when stuck on the
  ~-3.9 plateau (rmin > -4.0 AND < 0.05 improvement over the last 30 epochs).
  This protects rare **late-blooming good runs** (one historical good run sat at
  rmin@40 = -2.43 — worse than the bad runs — yet finished -4.825). A high early
  number does NOT mean bad; only `rmin <= -4.0` *guarantees* good.

## Tooling (`scripts/twin_search/`)
- `inventory_best_runs.py` — regenerate the best-runs report.
- `seed_pin_driver.sh` / `inner_seed.sh` — deterministic good-seed search (Phase 0).
- `variant_driver.sh` / `inner_variant.sh` — run the tail-fix variants + S38D,
  flatness early-kill, full-12 deterministic, gates vs eduardo.
- `parse_min.py` / `parse_min2.py` — running-min + flatness parser.
- `results.sh` — live dashboard.
> Host-specific absolute paths (shared GPU box); adapt before reuse.

## Current status (NEGATIVE result — read before re-running)
On the current machine, **new training does not reach the good basin**. ~28 runs
(S39B / twin_base, full AND reduced data, deterministic AND non-deterministic,
including eduardo's exact modified src, and one run with NO early-kill to epoch
114) **all stuck at `val_recon ≈ -3.9`** (recon plateaus while KL shrinks ⇒ partial
posterior collapse); none reached `≈ -4.97`. Ruled out as the cause: reduced data,
determinism, GRU backend (fused→compat auto-retry, all end on compat), the
early-kill (a no-early-kill run stayed flat), GPU health, and NVIDIA driver/CUDA
(no change — Jun 3 apt was only lzma/xz/linux-libc-dev). The historical good runs
used worktree code states whose exact form is no longer recoverable. **Root cause
not identified; treat as environmental/systemic until a good basin is observed
again.**

## Deliverable / most-promising state (resume here)
The **existing good twins on disk** are the current best digital twins:
- `eduardo …/exp_20260427_141943` — **10/12** (val_recon -4.978) ← champion.
- `cvae_repro_141943/outputs/retrain_s39b_seed42_20260602/exp_20260602_203631`
  — -4.977 (good basin). See `BEST_RUNS_INVENTORY.md`.
Reusable assets produced this round: opt-in determinism (validated bit-identical),
dataset-portability tooling (for the 5-LED set), the best-runs inventory, and these
docs.

## Next step
Onboard the **5-LED dataset (Option A)** with the portability tooling below and try
to obtain a *new* good basin there. If 5-LED training also can't reach a good basin,
the blocker is environmental on this box and needs the machine/stack investigated
(driver rollback, different GPU, or a clean container rebuild) before more search.

## Future work — new dataset (5-LED) and LED conditioning
See [NEW_DATASET_ONBOARDING.md](NEW_DATASET_ONBOARDING.md).
- **Option A (do first):** separate twin trained on the incoming 5-LED dataset
  (no code change). Onboard via `scripts/twin_search/make_protocol_from_dataset.py`,
  recalibrate the good-basin threshold, rerun the sweep.
- **Option B (future plan):** unified twin conditioned on LED count
  `p(y|x,d,c,n_leds)` (architecture change). Adopt only if a cross-comparison shows
  the unified model matches/beats the per-setup specialists.
