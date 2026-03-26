# Modeling Assumptions — VLC Channel Digital Twin (cVAE)

This document collects the **formal modeling definitions, objectives, and
constraints** that govern the cVAE architecture and training/evaluation
protocols.  All contributors must read and follow these assumptions.

---

## 1  Variable definitions

| Symbol | Domain | Description |
|--------|--------|-------------|
| $x \in \mathbb{R}^2$ | (I, Q) | Transmitted baseband sample. |
| $y \in \mathbb{R}^2$ | (I, Q) | Received sample after the physical VLC channel. |
| $d \in \mathbb{R}^+$ | meters | Distance between LED transmitter and photodetector. |
| $c \in \mathbb{R}^+$ | milliamps | LED drive / bias current. |
| $z \in \mathbb{R}^{d_z}$ | latent | Latent stochastic code ($d_z$ = `latent_dim`). |
| $\hat{y} \in \mathbb{R}^2$ | (I, Q) | Decoder output — the generative channel sample. |

The pair $(d, c)$ is collectively referred to as the **operating condition**
or **experiment regime**.  Each unique $(d, c)$ defines one experiment
sub-folder in the dataset.

---

## 2  Modeling objective

We seek to learn the **conditional channel distribution**:

$$p_\theta(y \mid x, d, c)$$

where $\theta$ are the decoder parameters.

This is **not** a regression objective.  A deterministic model
$\hat{y} = f(x, d, c)$ captures only $\mathbb{E}[y \mid x, d, c]$ and
discards higher-order statistics (variance structure, heavy tails,
multi-modality) that are physically present in VLC measurements.

The cVAE factorizes the generation through a latent variable $z$:

$$p_\theta(y \mid x, d, c) = \int p_\theta(y \mid x, d, c, z)\;
p_\psi(z \mid x, d, c)\;\mathrm{d}z$$

and optimizes the ELBO (Evidence Lower Bound):

$$\mathcal{L} = \mathbb{E}_{q_\phi(z | x,d,c,y)}\!\bigl[
\log p_\theta(y \mid x,d,c,z)\bigr]
- \beta\;\mathrm{KL}\!\bigl(q_\phi(z \mid x,d,c,y)\;\big\|\;
p_\psi(z \mid x,d,c)\bigr)$$

---

## 3  Why heteroscedastic regression is insufficient

A heteroscedastic Gaussian baseline predicts both $\mu(x,d,c)$ and
$\sigma^2(x,d,c)$ — one mean and one variance per input.  While it
accounts for *input-dependent noise magnitude*, it assumes:

- A **unimodal Gaussian** residual at every operating point.
- **No correlation** between the I and Q noise components.
- **Static deviation** per regime (no sample-level latent structure).

In practice:

| Phenomenon | Heteroscedastic handles? | cVAE handles? |
|------------|:------------------------:|:-------------:|
| LED clipping / nonlinearity | Partially (mean shift) | Yes (nonlinear generative) |
| Input-dependent variance | Yes | Yes |
| Correlated I/Q noise | No | Yes (full covariance via decoder) |
| Sample-level stochasticity | No (only regime-level $\sigma$) | Yes (via $z$) |
| Non-Gaussian tails | No | Yes (learned implicit density) |

The cVAE therefore provides a **strictly richer** model class.

---

## 4  Why generative modeling is necessary

The purpose of the digital twin is to **replace the physical channel** in
downstream simulations (equalization, coding, system optimization).  This
requires:

1. **Sampling**: draw $y \sim p_\theta(y \mid x, d, c)$ — impossible
   with a deterministic mapping.
2. **Distributional fidelity**: residual statistics (skewness, kurtosis,
   PSD) must match those of the real channel, not just the first moment.
3. **Generalization across regimes**: the model must interpolate to
   $(d, c)$ combinations not seen during training.

A generative model trained on the ELBO optimizes for distributional
coverage, whereas MSE regression optimizes for point accuracy.

---

## 5  Avoiding posterior collapse

Posterior collapse occurs when $q_\phi(z \mid x, d, c, y) \approx
p_\psi(z \mid x, d, c)$ for all inputs, meaning $z$ carries no useful
information and the decoder ignores it.  The model then degenerates to a
deterministic mapping.

Mitigation strategies implemented in this codebase:

| Strategy | Implementation |
|----------|---------------|
| **$\beta$-annealing** | KL weight ramps linearly from 0 to $\beta$ over `anneal_epochs`. |
| **Free-bits** | Per-dimension minimum KL threshold $\lambda$; dimensions below $\lambda$ contribute $\lambda$ instead of their actual KL. Prevents premature compression. |
| **KL diagnostics** | Active latent dimensions, per-dimension KL, and decoder sensitivity to $z$ are logged and plotted after training. |
| **Conditional prior** | A learnable $p_\psi(z \mid x, d, c)$ (not a fixed $\mathcal{N}(0,I)$) makes it harder for the encoder to match the prior trivially. |

---

## 6  Avoiding label leakage

> **Hard constraint: the decoder never receives $y$.**

The decoder input is $(\mathbf{x},\;d,\;c,\;z)$ only.  This is the
single most critical design constraint.

If $y$ enters the decoder (even indirectly), the model can learn an
identity map $\hat{y} \approx y$ and the latent space becomes
meaningless.  At inference time, $y$ is unavailable (the whole point of
the twin is to *predict* $y$), so any training shortcut through $y$ would
cause catastrophic performance degradation.

**Where $y$ is allowed:**

- **Encoder** $q_\phi(z \mid x, d, c, y)$ — $y$ is the observation that
  drives the approximate posterior.  This is correct: the encoder is
  discarded at inference and replaced by the prior.
- **Loss** — the reconstruction term $\log p_\theta(y \mid x, d, c, z)$
  obviously needs $y$ as the target.

**Where $y$ is forbidden:**

- Decoder input tensors.
- Prior network input tensors.
- Any feature engineering that leaks $y$ into $(x, d, c)$ before the
  decoder.

---

## 7  Data splitting policy

All splits are **per-experiment (regime-level)**.  Within each $(d, c)$
regime:

```
First N·head_frac samples  →  training set
Last  N·(1-head_frac) samples  →  validation set
```

- **No global shuffle.** Temporal order within each experiment is preserved.
- **No cross-regime mixing** before the split.
- The `head_frac` parameter (default 0.8) is recorded in `state_run.json`.

This prevents both **temporal leakage** (future samples appearing in train)
and **regime leakage** (different-condition samples artificially balancing
the split).
