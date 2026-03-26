# VLC Channel Digital Twin — cVAE

A PhD-level research repository implementing a **data-driven digital twin** of a
Visible Light Communication (VLC) channel using a Conditional Variational
Autoencoder (cVAE) with heteroscedastic decoding and conditional prior.

## Folder Name

This folder is:

- `/workspace/2026/feat_seq_bigru_residual_cvae`

Meaning of the name:

- `feat_seq_bigru_residual_cvae` comes from the Git branch name `feat/seq-bigru-residual-cvae`
- the name is historical; this folder is now the main unified worktree
- this folder contains the current code for all active architecture families

Use this folder when:

- you want the main day-to-day repository
- you want to switch architecture by `arch_variant`
- you want to compare `seq_bigru_residual` and `delta_residual`

Historical note:

- the old local folder for the adversarial line was removed to save disk space
- the branch name `feat/delta-residual-adv` still exists only for traceability

## Current Docs

Use these documents in this order:

- [docs/ACTIVE_CONTEXT.md](docs/ACTIVE_CONTEXT.md) — shortest path for the unified branch context
- [docs/FLOW_DECODER_PLAN.md](docs/FLOW_DECODER_PLAN.md) — active plan for the conditional flow decoder branch
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — current architecture and repo state
- [TRAINING_PLAN.md](TRAINING_PLAN.md) — active scientific plan and gates
- [docs/FUTURE_ADVERSARIAL_STRATEGY.md](docs/FUTURE_ADVERSARIAL_STRATEGY.md) — single backlog note for a future adversarial comeback
- [docs/RUN_REANALYSIS_PLAYBOOK.md](docs/RUN_REANALYSIS_PLAYBOOK.md) — how to review new `exp_*` runs quickly
- [docs/DIAGNOSTIC_CHECKLIST.md](docs/DIAGNOSTIC_CHECKLIST.md) — executable diagnostic workflow
- [docs/PROTOCOL.md](docs/PROTOCOL.md) — protocol runner, artifacts, CLI
- [docs/MODELING_ASSUMPTIONS.md](docs/MODELING_ASSUMPTIONS.md) — modeling rationale
- [docs/GEMINI_BOOTSTRAP.md](docs/GEMINI_BOOTSTRAP.md) — concise handoff for a secondary AI
- [docs/GEMINI_PLAYBOOK.md](docs/GEMINI_PLAYBOOK.md) — operational playbook for known situations
- [docs/GEMINI_PROMPTS.md](docs/GEMINI_PROMPTS.md) — prompt templates for Gemini or other copilots

Historical refactor planning has been archived under:

- [docs/archive/REFACTOR_PLAN_legacy.md](docs/archive/REFACTOR_PLAN_legacy.md)

## Experimental Branch

Current experimental branch in this worktree:

- `feat/conditional-flow-decoder`

Purpose of this branch:

- keep the Gaussian seq reference and the best MDN quick line as baselines
- stop spending more budget on pure MDN sweeps
- implement a conditional flow decoder over the residual law
- test whether a more expressive conditional likelihood closes the remaining
  near-range distribution-shape gap

Current implementation status:

- Phase 1 plumbing is already in place
- first flow family implemented:
  - conditional `sinh-arcsinh` flow per residual axis
- first smoke preset available:
  - `seq_flow_smoke`
- first structural smoke run completed:
  - `outputs/exp_20260326_033237`

The worktree path remains:

- `/workspace/2026/feat_seq_bigru_residual_cvae`

The folder name is historical; the active Git branch may differ from the path.

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

## Residual Instrumentation

The branch keeps the residual instrumentation layer focused on the residual
`Δ = Y - X`, which now becomes the direct target of the planned flow decoder.

New outputs:

- `tables/residual_signature_by_regime.csv`
- `tables/residual_signature_by_amplitude_bin.csv`
- `tables/train_regime_diagnostics_history.csv`
- `plots/best_model/residual_signature_overview.png`

New runtime controls:

- `train_regime_diagnostics_enabled`
- `train_regime_diagnostics_every`
- `train_regime_diagnostics_mc_samples`
- `train_regime_diagnostics_max_samples_per_regime`
- `train_regime_diagnostics_amplitude_bins`
- `train_regime_diagnostics_focus_only_0p8m`

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

## Architecture Variants

This branch contains multiple cVAE families behind a single builder. The
active architecture is chosen by the grid/config field:

- `arch_variant="concat"`
  - original point-wise cVAE
  - predicts `Y` directly from `(x, d, c, z)`
- `arch_variant="channel_residual"`
  - point-wise residual decoder
  - predicts a residual internally and resolves `Y = X + Δ`
- `arch_variant="delta_residual"`
  - point-wise residual-target variant
  - trains directly on `Δ = Y - X`, then reconstructs `Ŷ = X + Δ̂`
- `arch_variant="seq_bigru_residual"`
  - sequence-aware residual model
  - uses a short input window (`window_size=7`) and a BiGRU prior/encoder
  - requires `--no_data_reduction` because balanced block reduction breaks temporal context
- `arch_variant="legacy_2025_zero_y"`
  - faithful port of the 2025 notebook-era model for controlled comparison

In practice:

- this branch is the active research branch for `seq_bigru_residual`
  and `delta_residual`
- prefer switching `arch_variant`, `grid_tag`, or `grid_preset` instead of
  switching branches for day-to-day experiments
- `delta_residual` is the strongest current point-wise research line
- `seq_bigru_residual` is the strongest temporal line and includes the only
  reference run that has passed all gates so far
- mixed-family comparisons are now run inside the same `src.protocol.run`
  experiment rather than through separate training-only flows
- the adversarial strategy was intentionally removed from this active code path;
  if we revisit it later, use
  [docs/FUTURE_ADVERSARIAL_STRATEGY.md](docs/FUTURE_ADVERSARIAL_STRATEGY.md)
  as the only implementation note

## Repository layout

```
configs/
  protocol_default.json       Default reduced protocol (12 regimes: 3 distances x 4 currents)
  protocol_default.yaml       YAML variant of the same reduced protocol
conftest.py                   Pytest root config (makes `import src.*` work)
data/                         Dataset (Git LFS)
  dataset_fullsquare_organized/
    dist_*/curr_*/IQ_data/
docker/                       Dockerfile + container configs
docs/
  DIAGNOSTIC_CHECKLIST.md    Executable diagnosis workflow
  MODELING_ASSUMPTIONS.md     Core modeling decisions
  PROTOCOL.md                 Protocol runner reference
  SESSION_STATE.md            Session/run state schema
  archive/                    Historical plans kept out of the active path
  smoke_b2_notes.txt          Historical smoke notes
PROJECT_STATUS.md             Current codebase / validation status
TRAINING_PLAN.md              Active scientific plan and acceptance gates
notebooks/                    Exploratory Jupyter notebooks
outputs/                      Run artifacts (exp_YYYYMMDD_HHMMSS/)
scripts/
  train.sh                    Training wrapper
  eval.sh                     Evaluation wrapper
  run_tf25_gpu.sh             Start persistent TF 2.5 GPU container
  enter_tf25_gpu.sh           Open an interactive shell inside the container
  stop_tf25_gpu.sh            Stop and remove the persistent container
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
    runtime.py                Runtime builders for train/eval engines
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
    train.py                  Deprecated shim (kept only for backward compatibility)
    engine.py                 Canonical training engine entrypoint
    pipeline.py               Canonical training pipeline orchestration
    grid_plan.py              Canonical cVAE grid definition + filters
    gridsearch.py             Grid-search combinatorics + execution
    logging.py                state_run.json writer, RUN_DIR creation
  evaluation/
    evaluate.py               Evaluation entrypoint CLI
    engine.py                 Canonical evaluation engine entrypoint
    metrics.py                EVM, SNR, KL diagnostics, active dims
    plots.py                  Constellation overlays, histograms, scatter
    report.py                 CSV/JSON summary tables
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

### New collaborator: first run

If you are joining this project for the first time, use this path first.
It downloads the full repository with dataset pointers, opens the standard GPU
container, and runs the safest possible smoke test before any real training.

On the host:

```bash
git lfs install
git clone -b feat/seq-bigru-residual-cvae https://github.com/rodrigofmiranda/cVAe_2026.git
cd cVAe_2026
git lfs pull

bash scripts/run_tf25_gpu.sh
bash scripts/enter_tf25_gpu.sh
```

Inside the container:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae

# safest first check: validates config + dataset discovery, no training
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --dry_run
```

If that works, the next simple example is the reduced 12-regime protocol
(`0.8/1.0/1.5 m × 100/300/500/700 mA`), which is now the minimum active study:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --max_epochs 1 \
  --max_grids 1 \
  --max_experiments 1 \
  --max_samples_per_exp 2000
```

Do not start with a full 27-regime run or a long `--train_once_eval_all`
experiment until the two commands above work on your machine.

### Persistent GPU container helpers

For long runs on a remote host, prefer a persistent `tmux` session on the host
that owns the interactive Docker container. The repo now ships helper scripts:

```bash
./scripts/run_tf25_gpu.sh
./scripts/enter_tf25_gpu.sh
./scripts/stop_tf25_gpu.sh
```

These start a host-side `tmux` session named `cvae_tf25_gpu`, launch
`docker run --rm -it ... bash` inside it, and let you reattach later without
killing the container when your local editor disconnects.

### Reduced multi-regime and model-reuse workflow

For the current 12-regime reduced protocol (`0.8/1.0/1.5 m × 100/300/500/700 mA`):

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset best_compare_large \
  --max_epochs 80 \
  --patience 8 \
  --reduce_lr_patience 4 \
  --stat_tests --stat_mode quick \
  --no_data_reduction \
  --no_baseline
```

### Current recommended grids

The strongest current protocol-level seq reference is now:
- `exp_20260324_023558`
- winner:
  - `S6seq_W7_h64_lat4_b0p003_lmmd1p75_fb0p10_lr0p0003_L128-256-512`

Current replay under the new axis-wise diagnostics confirmed the same family as
the strongest seq line. The new reference picture is:

- `10/12` regimes passed
- all `1.0 m` regimes pass
- all `1.5 m` regimes pass
- only `0.8 m / 100 mA` and `0.8 m / 300 mA` still fail

The most recent capacity/context comparison run (`exp_20260323_210309`) did
not overtake it. The broader overnight preset remains available for wide
searches, but it is no longer the preferred immediate next step:

- `seq_overnight_12h`

For replaying only the strongest seq candidates under the new axis-wise
residual diagnostics, the repo now also includes:

- `seq_replay_axis_diagnostics`

Selected replay set:

- `S4seq_W7_h64_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
- `S4seq_W7_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
- `S4seq_W9_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
- `S2seq_W7_h64_lat4_b0p001_lmmd1p0_fb0p10_lr0p0003_L128-256-512`

Canonical replay command:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset seq_replay_axis_diagnostics \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests --stat_mode full --stat_max_n 5000 \
  --no_data_reduction
```

For the current next step, a short finishing sweep focused only on the
remaining `0.8 m` failures is preferred:

- `seq_finish_0p8m`

Design:

- keep `W7_h64` fixed
- center on the new winner `lambda_mmd=1.75`
- probe slightly stronger `lambda_mmd=2.0`
- include only low-LR / higher-beta hedges that were already strong in the
  overnight training diagnostics
- total: `6` candidates

Canonical finishing command:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset seq_finish_0p8m \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests --stat_mode full --stat_max_n 5000 \
  --no_data_reduction
```

For wider searches, the larger overnight preset remains available:

Design:

- keep the winning `W7_h64` family as the main axis
- sweep lower `lr` values for stability
- sweep stronger `lambda_mmd` for the hard `0.8 m` regimes
- keep only a small low-LR `W9_h96` hedge block
- total: `28` candidates
- target runtime: about `10` to `12` hours on the recent A6000-class setup

Canonical command:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset seq_overnight_12h \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests --stat_mode full --stat_max_n 5000 \
  --no_data_reduction
```

To reuse a previously trained shared model and skip retraining:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --reuse_model_run_dir outputs/exp_YYYYMMDD_HHMMSS/train \
  --stat_tests --stat_mode quick \
  --no_baseline
```

### Protocol runner (recommended)

The protocol runner is the main entrypoint. It orchestrates training,
evaluation, baseline comparison, distribution metrics, and statistical
fidelity tests for one or more $(d, c)$ regimes in a single reproducible
run.

```bash
# Default reduced 12-regime run (3 distances × 4 currents)
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --train_once_eval_all \
  --max_epochs 80

# Full 27-regime run (3 distances × 9 currents)
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_full_dataset.json \
  --max_epochs 200

# Reduced 12-regime smoke test with stat fidelity
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --max_epochs 30 --max_grids 1 --max_experiments 1 \
  --stat_tests --stat_mode full --stat_max_n 5000

# Dry run (validate protocol + model summary, no training)
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --dry_run

# Train once globally, then evaluate that same shared model across all regimes
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --train_once_eval_all \
  --max_epochs 120 --max_grids 2
```

### Canonical smoke tests

Use these three deterministic checks before wider runs:

```bash
# 1) Auto-discovery + model/bootstrap validation only
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --dry_run

# 2) Reduced 12-regime protocol smoke
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --max_epochs 1 --max_grids 1 --max_experiments 1 --max_samples_per_exp 2000

# 3) Import/build sanity for the cVAE stack
python -c "from src.models.cvae import build_cvae; build_cvae({'layer_sizes':[128,256,512],'latent_dim':4,'beta':0.003,'lr':3e-4,'dropout':0.0,'free_bits':0.1,'kl_anneal_epochs':80,'batch_size':16384,'activation':'leaky_relu'})"
```

### Which entrypoint to use

There is one canonical experiment entrypoint:

- `python -m src.protocol.run`: full experimental protocol for both exploratory grids and final scientific validation.

Inside `src.protocol.run`, there are **two protocol modes**:

- default: `per_regime_retrain`
  - trains one cVAE per regime
  - use this for diagnosis, baseline-vs-cVAE comparisons, and local regime analysis
- `--train_once_eval_all`: `train_once_eval_all`
  - trains a single shared global cVAE once on the selected dataset
  - evaluates that same model across all regimes without retraining
  - use this for the final universal digital-twin objective

In practice:

- exploratory grids should still use `src.protocol.run`, typically with `--train_once_eval_all` and a reduced protocol/config subset
- final validation should also use `src.protocol.run`, on the target protocol and with the canonical gates/summary tables
- `src.training.train` is no longer a supported public entrypoint

### Comparative mixed-family preset

The current protocol-first comparison preset is:

- `best_compare_large`

It mixes the strongest current candidates from two families:

- point-wise `delta_residual`
- temporal `seq_bigru_residual` (including the `lambda_mmd` block)

This is the recommended preset when the goal is to compare the best residual
and sequential candidates under the same reduced scientific protocol.

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --grid_preset best_compare_large \
  --max_epochs 80 \
  --patience 8 \
  --reduce_lr_patience 4 \
  --stat_tests --stat_mode quick \
  --no_data_reduction
```

For the thesis end goal, the target artifact is the **shared global model**:

- train once on the full dataset
- condition on `x`, `d`, and `I`
- evaluate by regime without retraining
- keep per-regime retraining only as a diagnostic / ablation path

Example: full-data exploratory cVAE grid search, without post-split train reduction:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --train_once_eval_all \
  --no_data_reduction
```

Residual-channel architecture experiments use a dedicated preset instead of
changing the default grid. This keeps the canonical concat decoder intact
while enabling a structural ablation where the decoder predicts
`Δ = Y - X` internally and resolves the final mean as `Y = X + Δ`.

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --train_once_eval_all \
  --no_data_reduction \
  --grid_preset residual_small \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6
```

For a cheaper full-data exploratory run, prefer the compact preset plus explicit
training patience overrides:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --train_once_eval_all \
  --no_data_reduction \
  --grid_preset exploratory_small \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6
```

There are two patience knobs:

- `--patience`: early stopping patience after KL warmup
- `--reduce_lr_patience`: patience for learning-rate reduction

Note: reducing patience helps, but very long runs are still mostly driven by
`--max_epochs` and each grid's KL warmup (`kl_anneal_epochs`).

### Final universal-twin workflow

For the final differentiable digital twin that will later sit inside the
end-to-end learned communication system, use this sequence:

1. `src.protocol.run --train_once_eval_all` with a focused global grid to identify strong shared-model candidates.
2. `src.protocol.run --train_once_eval_all` on the target protocol to train the shared model once and evaluate it across all regimes.
3. Read `tables/summary_by_regime.csv`, `tables/protocol_leaderboard.csv`, and the consolidated heatmap in `plots/best_model/`.

The canonical visual summary is:

- `plots/best_model/heatmap_gate_metrics_by_regime.png`
  - matrix of regime-level fidelity/gate metrics for the champion model
  - intended to answer whether the trained twin remains valid across the active `(distance, current)` conditions

Example:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --train_once_eval_all \
  --grid_tag "G2_lat4_b0p001_fb0p0_lr0p0003_L128-256-512|G0_lat4_b0p003_fb0p10_lr0p0003_L128-256-512" \
  --max_grids 2 \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6 \
  --stat_tests --stat_mode quick
```

### Legacy wrappers

```bash
# Shared-global protocol run (wrapper over src.protocol.run --train_once_eval_all)
bash scripts/train.sh

# Evaluate the latest (or a specific) run
bash scripts/eval.sh
```

### CLI flags reference

| Flag | Default | Purpose |
|------|---------|---------|
| `--dataset_root` | *(required)* | Path to organized dataset |
| `--output_base` | *(required)* | Root for run artifacts |
| `--protocol` | bundled reduced 12-regime default | Protocol JSON defining regimes |
| `--reuse_model_run_dir` | unset | Reuse a previous shared-model `train/` directory and skip retraining |
| `--max_epochs` | per-protocol | Maximum training epochs |
| `--max_grids` | all | Limit grid-search configs to N |
| `--grid_preset` | all | Named grid subset, e.g. `exploratory_small` |
| `--max_regimes` | all | Limit executed regimes after protocol resolution |
| `--max_experiments` | all | Limit experiments per regime |
| `--max_samples_per_exp` | all | Cap samples per experiment |
| `--max_val_samples_per_exp` | all | Cap validation samples per experiment after split |
| `--patience` | training default | Early stopping patience after KL warmup |
| `--reduce_lr_patience` | training default | ReduceLROnPlateau patience |
| `--skip_eval` | `false` | Train only, skip evaluation |
| `--dry_run` | `false` | Validate + model summary, no training |
| `--no_baseline` | `false` | Skip deterministic baseline |
| `--no_cvae` | `false` | Skip cVAE training/evaluation and keep only baseline outputs |
| `--baseline_only` | `false` | Alias for `--no_cvae` |
| `--no_dist_metrics` | `false` | Skip distribution-fidelity metrics |
| `--stat_tests` | `false` | Run MMD/Energy/PSD stat tests per regime |
| `--stat_mode` | `quick` | `quick` (200 perms) or `full` (2000 perms) |
| `--stat_max_n` | `5000` in `quick`, `50000` in `full` | Max samples for stat tests |
| `--keras_verbose` | `2` | Keras fit verbosity (0/1/2) |

Quick mode note:

- `stat_mode=quick` alone does not cap Keras validation.
- For a real quick run, cap all three:
  - `--max_samples_per_exp`
  - `--max_val_samples_per_exp`
  - `--max_dist_samples`

## Run artifacts

Each run produces:

```
outputs/exp_YYYYMMDD_HHMMSS/
  manifest.json                 Full run manifest (config, timing, winner summary)
  logs/                         Shared logs root for the whole experiment
    protocol_input.json
    train/
    eval/
  tables/
    summary_by_regime.csv       Canonical per-regime validation table (source of truth)
    protocol_leaderboard.csv    Canonical ranking derived from protocol metrics/gates
    stat_fidelity_by_regime.csv Statistical-fidelity projection
  plots/
    best_model/
      heatmap_gate_metrics_by_regime.png
  train/                        Shared global model directory (only in --train_once_eval_all)
    models/
      best_model_full.keras
      best_decoder.keras
      best_prior_net.keras
    plots/
      champion/
        analysis_dashboard.png
      training/
        dashboard_analysis_complete.png
    tables/
      grid_training_diagnostics.csv
      gridsearch_results.csv
      gridsearch_results.xlsx
    state_run.json
  eval/
    <regime_id>/
      state_run.json
      plots/
      tables/
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

### Canonical summary fields
- `summary_by_regime.csv` is the primary table consumed by validation scripts and thesis analysis.
- It includes physical fidelity, residual-distribution metrics, baseline vs cVAE comparisons, and the formal `stat_*` test outputs in one row per regime.
- Derived columns include `var_ratio_pred_real`, `better_than_baseline_*`, `gate_g1`…`gate_g6`, and `validation_status`.

### Champion plots
- Only the current champion gets the full visual bundle.
- The training side writes:
  - `train/plots/champion/analysis_dashboard.png`
  - `train/plots/training/dashboard_analysis_complete.png`
- The protocol side writes:
  - `plots/best_model/heatmap_gate_metrics_by_regime.png`
- The dashboard is the main human-facing artifact for the winner.
- The training dashboard is the operational artifact for convergence and hyperparameter decisions.
- The heatmap is the scientific artifact for regime-by-regime twin validation.

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

## Dataset (Git LFS)

The experimental I/Q dataset is stored in this repository via **Git LFS**.
It contains synchronized baseband captures from a real VLC channel
(LED → free-space → photodetector) across 27 operating regimes.

### Requirements

```bash
# Install Git LFS (if not already available)
sudo apt install git-lfs   # Debian/Ubuntu
brew install git-lfs        # macOS

git lfs install
```

### Cloning with data

```bash
# Full clone (downloads all LFS objects — ~1.1 GB)
git clone https://github.com/rodrigofmiranda/cVAe_2026.git
cd cVAe_2026

# If you already cloned without LFS, pull the data:
git lfs pull
```

### Clone on another PC

If you want the current unified research branch, clone:

```bash
git clone -b feat/seq-bigru-residual-cvae https://github.com/rodrigofmiranda/cVAe_2026.git
cd cVAe_2026
git lfs install
git lfs pull
```

This is the recommended path for collaborators working on the current
protocol-first flow. This same branch contains the sequential line and the
non-adversarial point-wise residual line. Select between them with
`arch_variant` and the corresponding grid preset.

Recommended branches:

- `release/cvae-online` — recommended base for a functional online cVAE deployment
- `feat/channel-residual-architecture` — residual-architecture branch
- `feat/seq-bigru-residual-cvae` — active research branch for `seq_bigru_residual` and `delta_residual`
- `feat/delta-residual-adv` — archived adversarial branch kept only for traceability

Clone directly into the recommended online branch:

```bash
git clone -b release/cvae-online https://github.com/rodrigofmiranda/cVAe_2026.git
cd cVAe_2026
git lfs install
git lfs pull
```

If you already cloned the repository and want a different branch:

```bash
git fetch --all
git switch release/cvae-online
git lfs pull
```

Switch to the residual or unified research branch when needed:

```bash
git switch feat/channel-residual-architecture
git lfs pull

git switch feat/seq-bigru-residual-cvae
git lfs pull
```

### Dataset structure

```
data/dataset_fullsquare_organized/
  _report/                        Dataset-level quality report
    REPORT.md                     Summary with plots and tables
    summary_by_regime.csv         Per-regime statistics
    heatmap_*.png, line_*.png     Diagnostic plots
  dist_0.8m/                      Distance = 0.8 m
    curr_100mA/                   LED current = 100 mA
      full_square_0.8m_100mA_001_YYYYMMDD_HHMMSS/
        IQ_data/
          X.npy                   Transmitted I/Q (N×2, float32) — cVAE input
          Y.npy                   Received I/Q (N×2, float32)   — cVAE target
          x_sent.npy              Raw sent (intermediate, not used by cVAE)
          y_recv.npy              Raw received (intermediate)
          y_recv_sync.npy         Sync without normalization (intermediate)
          y_recv_norm.npy         Normalized without phase (intermediate)
        metadata.json             Experiment metadata (conversion params)
        report.json               Per-experiment quality metrics
    curr_200mA/
    …
    curr_900mA/
  dist_1.0m/                      Distance = 1.0 m
    curr_100mA/ … curr_900mA/
  dist_1.5m/                      Distance = 1.5 m
    curr_100mA/ … curr_900mA/
```

> **Note:** The cVAE uses **only** `X.npy` (input) and `Y.npy` (target).
> The other `.npy` files are intermediate pipeline outputs — ignore them.

| Dimension | Value |
|-----------|-------|
| Distances | 0.8 m, 1.0 m, 1.5 m |
| Currents | 100, 200, 300, 400, 500, 600, 700, 800, 900 mA |
| Regimes | 3 × 9 = **27** |
| Samples per regime | ~900,000 I/Q pairs |
| Array shape | `(N, 2)` — columns are `[I, Q]` (in-phase, quadrature) |
| Dtype | `float32` |
| Total files | 309 (162 `.npy` + 91 `.png` + 54 `.json` + 1 `.md` + 1 `.csv`) |
| LFS-tracked | `.npy` and `.png` files (~1.1 GB) |
| Git-tracked | `.json`, `.md`, `.csv` files (small, human-readable) |

### Loading data in Python

```python
import numpy as np, json

base = "data/dataset_fullsquare_organized/dist_0.8m/curr_200mA"
exp  = "full_square_0.8m_200mA_001_20260211_153502"

# cVAE arrays
X = np.load(f"{base}/{exp}/IQ_data/X.npy")   # (N, 2) — transmitted
Y = np.load(f"{base}/{exp}/IQ_data/Y.npy")   # (N, 2) — received (sync+phase+norm)

# Metadata & quality metrics
meta   = json.load(open(f"{base}/{exp}/metadata.json"))
report = json.load(open(f"{base}/{exp}/report.json"))

print(f"Sent: {X.shape}  Received: {Y.shape}")
print(f"EVM: {report['evm_pct']:.2f}%  SNR: {report['snr_dB']:.2f} dB")
```

### metadata.json keys

| Key | Description |
|-----|-------------|
| `dist_m` | Distance in meters |
| `curr_mA` | LED forward current in milliamps |
| `regime_id` | Unique regime identifier (e.g. `dist_0p8m__curr_100mA`) |
| `conversion.factor_ref` | Amplitude normalization factor (anchor-based) |
| `conversion.estimated_lag_samples` | Sync delay in samples |
| `conversion.estimated_phase_rad` | Estimated carrier phase offset (rad) |
| `conversion.anchor` | Reference regime used for normalization |

### report.json keys

| Key | Description |
|-----|-------------|
| `evm_pct` | Error Vector Magnitude (%) |
| `snr_dB` | Signal-to-Noise Ratio (dB) |
| `n_samples` | Number of I/Q samples |
| `lag_samples` | Synchronization lag (samples) |
| `phase_rad` | Estimated phase (radians) |
| `factor_ref` | Normalization factor |
| `var_I`, `var_Q` | Noise variance per quadrature |
| `log_var_I`, `log_var_Q` | Log-variance per quadrature |
| `skew_I`, `kurt_excess_I` | Higher-order statistics (I channel) |

## References

- Kingma & Welling, "Auto-Encoding Variational Bayes", 2014.
- Gretton et al., "A Kernel Two-Sample Test", JMLR 2012 (MMD).
- Székely & Rizzo, "Testing for Equal Distributions in High Dimension", 2004 (Energy distance).
- Benjamini & Hochberg, "Controlling the False Discovery Rate", 1995 (FDR).
- Surveys on data-driven VLC channel modeling (2020–2025).
