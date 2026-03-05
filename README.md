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
configs/
  protocol_default.json       Default protocol definition (regimes × studies)
  protocol_default.yaml       YAML variant
conftest.py                   Pytest root config (makes `import src.*` work)
data/                         Dataset (Git LFS)
  dataset_fullsquare_organized/
    dist_*/curr_*/IQ_data/
docker/                       Dockerfile + container configs
docs/
  MODELING_ASSUMPTIONS.md     Core modeling decisions
  PROTOCOL.md                 Protocol runner reference
  SESSION_STATE.md            Session/run state schema
  smoke_b2_notes.txt          Single-regime full stat-fidelity smoke results
notebooks/                    Exploratory Jupyter notebooks
outputs/                      Run artifacts (exp_YYYYMMDD_HHMMSS/)
scripts/
  train.sh                    Training wrapper
  eval.sh                     Evaluation wrapper
  smoke_dist_metrics.sh       Quick smoke test for distribution metrics
tests/
  test_stat_tests.py          18 unit tests for MMD, Energy, PSD, FDR
  test_stat_plots.py          11 unit tests for stat fidelity plots
src/
  baselines/
    deterministic.py          Deterministic regression baseline
  config/
    defaults.py               Default values, key names
    io.py                     Load/save config (YAML/JSON)
    overrides.py              RunOverrides dataclass (CLI → per-regime)
    schema.py                 Dataclasses for run configuration
  data/
    channel_dataset.py        GNU Radio capture block
    loading.py                Dataset IO, discovery, experiment loading
    splits.py                 Train/val split by experiment (no global shuffle)
    normalization.py          Peak/power normalization, sync rules
  metrics/
    distribution.py           Distribution-fidelity metrics (moments, PSD, Gaussianity)
  models/
    cvae.py                   cVAE encoder/decoder/prior architecture
    cvae_components.py        Sub-network building blocks
    callbacks.py              Early stopping, ReduceLR, logging
    losses.py                 Reconstruction, KL, free-bits, β schedule
    sampling.py               Reparameterization trick, prior sampling
  training/
    train.py                  Training entrypoint CLI
    engine.py                 Training loop: build, compile, fit, save
    gridsearch.py             Grid-search combinatorics + execution
    logging.py                state_run.json writer, RUN_DIR creation
    cvae_TRAIN_documented.py  Legacy monolithic training script (reference)
  evaluation/
    evaluate.py               Evaluation entrypoint CLI
    metrics.py                EVM, SNR, KL diagnostics, active dims
    plots.py                  Constellation overlays, histograms, scatter
    report.py                 CSV/JSON summary tables
    analise_cvae_reviewed.py  Legacy monolithic evaluation script (reference)
    non_gaussianity_by_regime.py
    stat_tests/               Statistical Fidelity Suite
      __init__.py             Convenience imports (mmd_rbf, energy_test, etc.)
      mmd.py                  Two-sample MMD test (RBF kernel, BLAS-optimised permutations)
      energy.py               Energy distance test (BLAS-optimised permutations)
      psd.py                  Power Spectral Density distance + bootstrap CI
      fdr.py                  Benjamini–Hochberg FDR correction
      plots.py                Heatmaps + scatter plots for stat results
  protocol/
    run.py                    Protocol runner — orchestrates train+eval across regimes
    selector_engine.py        Regime × experiment selection logic
    split_strategies.py       Within-regime / cross-regime split strategies
```

## Quick start

### Protocol runner (recommended)

The protocol runner is the main entrypoint. It orchestrates training,
evaluation, baseline comparison, distribution metrics, and statistical
fidelity tests for one or more $(d, c)$ regimes in a single reproducible
run.

```bash
# Full 27-regime run (3 distances × 9 currents)
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --max_epochs 200

# Single-regime smoke test with stat fidelity
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol path/to/one_regime.json \
  --max_epochs 30 --max_grids 1 --max_experiments 1 \
  --stat_tests --stat_mode full --stat_max_n 5000

# Dry run (validate protocol + model summary, no training)
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --dry_run
```

### Legacy wrappers

```bash
# Train (grid search over ~42 configurations)
bash scripts/train.sh

# Evaluate the latest (or a specific) run
bash scripts/eval.sh
```

### CLI flags reference

| Flag | Default | Purpose |
|------|---------|---------|
| `--dataset_root` | *(required)* | Path to organized dataset |
| `--output_base` | *(required)* | Root for run artifacts |
| `--protocol` | `configs/protocol_default.json` | Protocol JSON defining regimes |
| `--max_epochs` | per-protocol | Maximum training epochs |
| `--max_grids` | all | Limit grid-search configs to N |
| `--max_experiments` | all | Limit experiments per regime |
| `--max_samples_per_exp` | all | Cap samples per experiment |
| `--skip_eval` | `false` | Train only, skip evaluation |
| `--dry_run` | `false` | Validate + model summary, no training |
| `--no_baseline` | `false` | Skip deterministic baseline |
| `--no_dist_metrics` | `false` | Skip distribution-fidelity metrics |
| `--stat_tests` | `false` | Run MMD/Energy/PSD stat tests per regime |
| `--stat_mode` | `quick` | `quick` (200 perms) or `full` (2000 perms) |
| `--stat_max_n` | `50000` | Max samples for stat tests |
| `--keras_verbose` | `2` | Keras fit verbosity (0/1/2) |

## Run artifacts

Each run produces:

```
outputs/exp_YYYYMMDD_HHMMSS/
  manifest.json                 Full run manifest (config, acceptance, timing)
  tables/
    summary_by_regime.csv       All metrics for every regime (one row each)
    stat_fidelity_by_regime.*   FDR-corrected stat test results (csv + xlsx)
  plots/
    stat_tests/                 Heatmaps + scatter plots (MMD, q-val, PSD)
  studies/
    within_regime/regimes/
      <regime_id>/
        state_run.json          Config snapshot + normalization + split info
        models/
          best_model_full.keras Best cVAE (encoder + prior + decoder)
          best_decoder.keras
          best_prior_net.keras
        logs/
          training_history.json
          metricas_globais_reanalysis.json
        plots/                  Constellation overlays, PSD, latent diagnostics
        tables/                 Grid results, dataset inventory, split tables
```

## Key metrics

### Signal quality
- **EVM (%)** and **SNR (dB)** — physical signal quality (real vs. predicted).
- **ΔEVM / ΔSNR** — gap between cVAE and baseline.

### Distribution fidelity
- **Δmean\_l2, Δcov\_fro, Δskew\_l2, Δkurt\_l2** — moment-level distance
  between real and synthetic residuals.
- **PSD\_L2** — spectral distance of residual $\Delta = y - x$.
- **Gaussianity rejection** — Jarque–Bera test on residuals.

### Statistical Fidelity Suite (`--stat_tests`)
- **MMD²** (Maximum Mean Discrepancy, RBF kernel) + permutation p-value.
- **Energy distance** (Székely & Rizzo) + permutation p-value.
- **PSD distance** + bootstrap 95% confidence interval.
- **FDR correction** (Benjamini–Hochberg) across all regimes → q-values.
- **Acceptance verdict**: PASS / PARTIAL / FAIL based on q-values and PSD
  ratio relative to baseline.

### Latent diagnostics
- Active dimensions, KL per dimension, decoder sensitivity to $z$.
- **score\_v2**: composite ranking metric for grid search.

## Testing

```bash
# Run all stat tests (29 tests, ~6 s)
pytest tests/ -v

# Specific test file
pytest tests/test_stat_tests.py -v    # 18 tests: MMD, Energy, PSD, FDR
pytest tests/test_stat_plots.py -v    # 11 tests: heatmaps, scatter, generate_all
```

## References

- Kingma & Welling, "Auto-Encoding Variational Bayes", 2014.
- Gretton et al., "A Kernel Two-Sample Test", JMLR 2012 (MMD).
- Székely & Rizzo, "Testing for Equal Distributions in High Dimension", 2004 (Energy distance).
- Benjamini & Hochberg, "Controlling the False Discovery Rate", 1995 (FDR).
- Surveys on data-driven VLC channel modeling (2020–2025).
