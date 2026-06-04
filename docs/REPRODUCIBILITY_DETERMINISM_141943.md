# Reproducibility & determinism — run `exp_20260427_141943` (cVAE S39B)

Investigation of why the run `exp_20260427_141943`
(`grid_tag=S39B_edgegap_lowlr_all08_w18_p120`, preset `seq_edgegap_targeted_short`)
could not be reproduced "even with the same seed", and how to make training
deterministic.

Base code: clean commit `c622a33` (the run's recorded commit). Env: image
`vlc/tf25-gpu-ready:1`, TF 2.17.0, NumPy 1.26.4, Python 3.12.3, RTX 5090.

## TL;DR

- **Configuration reproduces exactly**: the resolved `grid_cfg` is bit-identical
  to the historical run.
- **The final training minimum does not**, because training is **non-deterministic
  at the GPU-op level** and the loss landscape is **bimodal/sensitive**. Same
  code + same `seed=42` lands in one of two basins (~50/50 in our sample):
  - **good**: `val_recon ≈ -4.97`, ~390–500 epochs, ~8–10/12 gates;
  - **bad**: stuck `val_recon ≈ -3.9`, early-stops ~epoch 90–140, 0/12.
- **The seed is not enough** because `tf.random.set_seed` fixes the RNG (init,
  shuffle, sampling) but **not the floating-point reduction order** of GPU
  kernels (atomics), which is non-deterministic by default.
- **Determinism is achievable and was demonstrated** (bit-identical twins) by
  setting `TF_DETERMINISTIC_OPS=1` + `TF_CUDNN_DETERMINISTIC=1`. Note:
  `tf.config.experimental.enable_op_determinism()` **breaks this stack**
  (`'float' object cannot be interpreted as an integer` at model build).

## 1. What reproduces

The resolved hyperparameter set `grid_cfg` produced from clean `c622` is
**identical** to the historical run (checked field by field). Same pipeline,
losses, architecture and protocol. So methodological/configurational
reproducibility holds.

## 2. What does not — the bimodal basin (empirical)

Same commit / dataset / `seed=42` / env, multiple independent runs:

| Run | epochs | min `val_recon` | basin |
|---|---|---|---|
| historical 141943 | 492 | **-4.978** | good (~10/12) |
| repro A | 393 | -4.976 | good |
| repro B | 500 | -4.977 | good |
| repro C | 137 | **-3.948** | bad (0/12) |
| independent (clean c622) | 91 | **-3.916** | bad (0/12) |

Both outcomes occurred in the *same clone/environment* → it is **stochastic**,
not environmental. Per-epoch `val_recon` is itself noisy (swings between -1.5
and -4.8 even in good runs), so judging reproduction by a single early epoch is
invalid; the basin only settles after ~epoch 430.

## 3. Where the (in)determinism lives, in code

Seeds **are** fixed — `src/training/pipeline.py`:

```python
np.random.seed(int(runtime.training_config["seed"]))
tf.random.set_seed(int(runtime.training_config["seed"]))
```

These govern the VAE/MDN sampling and shuffles (`src/models/losses.py`,
`src/models/sampling.py`, `src/models/cvae.py`).

What is **absent anywhere in the code**: any call to
`enable_op_determinism` / `TF_DETERMINISTIC_OPS` / `TF_CUDNN_DETERMINISTIC`
(grep → 0 hits). So GPU kernels accumulate floating-point in non-deterministic
order. Because `(a+b)+c ≠ a+(b+c)` in float, ~1e-7 differences appear and
**compound over ~500 epochs**; near the basin boundary this selects different
minima.

Note: the model already avoids the cuDNN GRU kernel (`seq_gru_backend="compat"`
→ `layers.RNN(GRUCell, unroll=False)`, see `src/models/cvae_sequence.py`), which
was for *stability* on the RTX 5090 — not determinism. The remaining
non-determinism is in the gradient reductions/atomics across the graph.

## 4. The determinism switch (this branch)

Added, opt-in via env vars (off by default — deterministic kernels are ~2–3×
slower):

- `src/protocol/run.py` — before TF is imported:
  ```python
  if os.environ.get("CVAE_DETERMINISTIC","0")=="1" and os.environ.get("CVAE_DET_ENV","1")=="1":
      os.environ["TF_DETERMINISTIC_OPS"]   = "1"
      os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
  ```
- `src/training/pipeline.py` — at seed time, optional `set_random_seed` and
  `enable_op_determinism` behind sub-flags `CVAE_DET_SETSEED` / `CVAE_DET_OPDET`.

**Important finding:** `tf.config.experimental.enable_op_determinism()` (TF 2.17
+ tf_keras) makes both baseline and cVAE fail at model build with
`'float' object cannot be interpreted as an integer`. Isolated with the
sub-flags. The **working** determinism path on this stack is the env vars only:

```
CVAE_DETERMINISTIC=1 CVAE_DET_OPDET=0 CVAE_DET_SETSEED=0
```

## 5. Demonstration — bit-identical twins

4 short trainings (`--max_epochs 3`, `seed=42`), per-epoch `val_recon_loss`:

```
determinism OFF  ndet_a : [ 0.00047947, -3.0690284, -3.5099626]
determinism OFF  ndet_b : [ 0.09217115, -2.9140275, -2.6952975]   → differ from epoch 1
determinism ON   det_a  : [-0.01590759, -2.7844505, -2.1324918]
determinism ON   det_b  : [-0.01590759, -2.7844505, -2.1324918]   → bit-identical
```

- non-deterministic twins (same seed): **not identical**;
- deterministic twins (same seed): **identical** (full float precision).

Reproduce with `scripts/repro_determinism/demo_driver.sh`.

## 6. Is determinism good or bad for learning?

Determinism buys **repeatability, not robustness**.

- Expected model quality: **unchanged** — it fixes one trajectory, doesn't train
  better/worse on average. The non-determinism here is not a useful regularizer.
- It **freezes the basin draw**: whatever basin deterministic-seed-42 lands in is
  fixed; if bad, it is bad every time (change the seed).
- Cost: ~2–3× slower; some ops lack a deterministic GPU impl (here,
  `enable_op_determinism` errors).
- The real *learning* problem is the **bimodal/sensitive landscape**, which
  determinism does not fix. To make most seeds reach the good basin, change
  training robustness (longer KL warmup, LR warm restarts, gradient clipping,
  better init, softer loss weighting) — not determinism.

## 7. Why the historical run is not exactly reproducible a posteriori

It did not pin (a) determinism flags, (b) the Docker image digest / cuDNN+driver
versions, (c) separate, logged train vs Monte-Carlo eval seeds. The manifest
records the git commit, but the specific non-deterministic trajectory ("draw")
was not saved and cannot be recovered. Additionally the acceptance gates are
Monte-Carlo permutation tests: the *same* trained model scores 6/12–10/12 across
`stat_seed` 42–46, so the exact gate fingerprint is also seed-sensitive.

**Defensible statement:** the historical `10/12` lies inside the reproduced
model's variability envelope; configuration reproduces exactly; bit-exact
reproduction of that specific run is impossible without the missing determinism
pinning, but training *can* be made fully deterministic going forward (§5).

## 8. Recommendations

1. For reported results: enable determinism (`CVAE_DETERMINISTIC=1
   CVAE_DET_OPDET=0 CVAE_DET_SETSEED=0`) + pin image digest + cuDNN/driver + 1 GPU.
2. Log the dataset hash and separate train/eval seeds.
3. Report the run as a **distribution over seeds** (basin hit-rate, gate mean ± sd),
   not a single point — standard ML reproducibility practice.
