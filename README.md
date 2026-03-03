# VLC Channel Digital Twin — cVAE

A PhD-level research repository implementing a **data-driven digital twin** of a
Visible Light Communication (VLC) channel using a Conditional Variational
Autoencoder (cVAE) with heteroscedastic decoding and conditional prior.

## Objective

Learn the conditional distribution of the physical VLC channel from
synchronized experimental I/Q measurements:

$$p(y \mid x, d, c)$$

| Symbol | Meaning |
|--------|---------|
| $x \in \mathbb{R}^2$ | Transmitted I/Q sample (baseband) |
| $y \in \mathbb{R}^2$ | Received I/Q sample after the physical channel |
| $d \in \mathbb{R}$ | Distance between LED and photodetector (m) |
| $c \in \mathbb{R}$ | LED drive / bias current (mA) |

The goal is **distributional fidelity** — not only mean mapping — preserving
the nonlinearities and noise characteristics of the real channel.

## Modeling Philosophy

### Deterministic baseline

A deterministic regression model $\hat{y} = f(x, d, c)$ learns only the
conditional mean $\mathbb{E}[y \mid x, d, c]$.  Even with heteroscedastic
noise heads (predicting per-sample variance), such models underestimate
tail behavior, multi-modal structure, and correlated residual patterns that
are physically present in VLC channels (LED nonlinearity, shot noise,
multipath).

### Generative digital twin (cVAE)

The cVAE captures the **full conditional distribution**, not just the mean:

- **Encoder** $q_\phi(z \mid x, d, c, y)$ — compresses the residual
  information that $y$ carries *beyond* what $(x, d, c)$ can explain.
  Used **only during training**.
- **Conditional prior** $p_\psi(z \mid x, d, c)$ — learns to predict the
  latent code from observable conditions alone.  Used at **inference time**.
- **Decoder** $p_\theta(y \mid x, d, c, z)$ — generates channel output
  from the transmit signal, operating conditions, and the latent sample.

> **Critical constraint:** the decoder **never receives $y$**.  It sees only
> $(x, d, c, z)$.  This prevents label leakage and ensures the twin
> generalizes as a forward channel model.

The cVAE is **not** a supervised autoencoder.  The latent variable $z$
represents the residual stochasticity of the physical channel that cannot
be predicted from $(x, d, c)$ alone.

### Training discipline

- **Split by experiment** (regime-level): each $(d, c)$ regime is split
  temporally (head = train, tail = val) to avoid temporal leakage.
- **No global shuffle**: samples from different regimes are never mixed
  before splitting.
- **$\beta$-annealing + free-bits**: stabilize ELBO optimization and
  prevent posterior collapse where $z$ is ignored.

## Repository layout

```
data/                     Dataset (Git LFS)
  dataset_fullsquare_organized/
    dist_*/curr_*/IQ_data/
docker/                   Dockerfile
docs/                     Technical documentation
notebooks/                Exploratory notebooks
outputs/                  Run artifacts (run_YYYYMMDD_HHMMSS/)
scripts/
  train.sh                Training wrapper
  eval.sh                 Evaluation wrapper
src/
  config/                 Defaults, IO, schemas (stubs)
  data/
    channel_dataset.py    GNU Radio capture block
    loading.py            Dataset IO, discovery, experiment loading
    splits.py             (stub)
    normalization.py      (stub)
  models/                 Model definitions (stubs)
  training/
    cvae_TRAIN_documented.py   Main training + grid search script
    logging.py            Run bootstrap (RUN_DIR, _last_run.txt)
  evaluation/
    analise_cvae_reviewed.py   Evaluation + metrics + plots
    non_gaussianity_by_regime.py
```

## Quick start

```bash
# Train (grid search over ~48 configurations)
bash scripts/train.sh

# Evaluate the latest (or a specific) run
bash scripts/eval.sh
# or: RUN_ID=run_20260301_120000 bash scripts/eval.sh
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATASET_ROOT` | `data/dataset_fullsquare_organized` | Path to organized dataset |
| `OUTPUT_BASE` | `outputs/` | Root for run artifacts |
| `RUN_ID` | auto (`run_YYYYMMDD_HHMMSS`) | Explicit run identifier |

## Run artifacts

Each run produces:

```
outputs/run_YYYYMMDD_HHMMSS/
  state_run.json            Config snapshot + normalization + split info
  models/
    best_model_full.keras   Best cVAE (full encoder+prior+decoder)
    best_decoder.keras
    best_prior_net.keras
  logs/
    training_history.json
  plots/                    Constellation overlays, PSD, latent diagnostics
  tables/                   Grid results, dataset inventory, split tables
```

## Key metrics

- **EVM (%)** and **SNR (dB)** — physical signal quality.
- **Residual distribution**: mean/covariance distance, skewness, kurtosis,
  PSD of $\Delta = y - x$.
- **Latent diagnostics**: active dimensions, KL per dimension, decoder
  sensitivity to $z$.
- **score\_v2**: composite ranking metric for grid search.

## References

- Kingma & Welling, "Auto-Encoding Variational Bayes", 2014.
- Surveys on data-driven VLC channel modeling (2020–2025).
