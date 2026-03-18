# Claude Code Plan — seq-BiGRU Residual cVAE

## 0. Scope

This document is the starting plan for a **new architecture project** on branch:

```text
feat/seq-bigru-residual-cvae
```

The goal is to replace the current **point-wise** cVAE decoder family with a
**sequence-aware** conditional variational model that uses local temporal
context from the transmitted waveform.

This plan is written for Claude Code so it can start the work without needing
to inspect the external PDF tooling or the local paper index.

Recommended Claude Code integration:

- load this plan through the project memory file [CLAUDE.md](/workspace/2026/CLAUDE.md)
- keep review-only criteria in [REVIEW.md](/workspace/2026/REVIEW.md)
- use shared project settings in [.claude/settings.json](/workspace/2026/.claude/settings.json)
  so Claude starts in Plan Mode for this branch by default
- use path-scoped rules in [.claude/rules/seq-invariants.md](/workspace/2026/.claude/rules/seq-invariants.md)
  so the key invariants load when Claude works in `src/`, `tests/`, and related files
- use reusable project skills:
  - `/seq-bigru-kickoff [phase-or-focus]`
  - `/seq-review [scope]`
- do not rely on manual copy/paste of this file when Claude Code can load the
  shorter shared memory and then read this plan intentionally

Important:

- The `knowledge/` area and any local PDF / Docling / Chroma tooling are **not**
  part of the deliverable for this repo.
- Literature findings that matter for this architectural change are summarized
  in this document already.
- Do not spend time extending the paper-ingestion tooling in this branch.

This file is intentionally detailed.
It is the implementation handoff, not the always-loaded memory file.
Keep the always-loaded project memory concise in `CLAUDE.md`.

## 0.1 Recommended Claude Code Workflow

Use Claude Code with this project in this order:

1. launch from `/workspace/2026`
2. confirm loaded memory with `/memory`
3. keep the session in Plan Mode while doing Phase 0 and planning the next code change
4. use `/rename` early so the session can be resumed later
5. switch to normal editing only when the next change is narrow and explicit
6. use `/compact` after long investigations to preserve decisions and reduce context noise
7. use `/rewind` or checkpoints if the implementation drifts
8. for parallel investigations or alternate approaches, prefer separate worktrees
   over mixing multiple architectural experiments in one session

Why this workflow:

- this project is a multi-file architecture change with scientific constraints
- the risk of solving the wrong problem is higher than the cost of an explicit plan
- the repo already contains unrelated local changes, so session hygiene matters

## 1. Starting Point

Current base commit of the active project line:

```text
48e1ca9  docs(ai): add Gemini handoff, playbook, and prompt templates
```

Current branch for this project:

```text
feat/seq-bigru-residual-cvae
```

The current cVAE architecture state before this new project is:

- default architecture: `concat`
- experimental architecture already tested: `channel_residual`
- both are **point-wise MLP** models
- both failed the final scientific protocol in `train_once_eval_all`

## 2. Read Before Coding

Read these files in this exact order before making changes:

1. [PROJECT_STATUS.md](/workspace/2026/PROJECT_STATUS.md)
2. [TRAINING_PLAN.md](/workspace/2026/TRAINING_PLAN.md)
3. [docs/PROTOCOL.md](/workspace/2026/docs/PROTOCOL.md)
4. [src/models/cvae.py](/workspace/2026/src/models/cvae.py)
5. [src/data/loading.py](/workspace/2026/src/data/loading.py)
6. [src/protocol/split_strategies.py](/workspace/2026/src/protocol/split_strategies.py)
7. [src/training/pipeline.py](/workspace/2026/src/training/pipeline.py)
8. [src/training/grid_plan.py](/workspace/2026/src/training/grid_plan.py)
9. [src/config/defaults.py](/workspace/2026/src/config/defaults.py)
10. [src/config/schema.py](/workspace/2026/src/config/schema.py)
11. [src/protocol/run.py](/workspace/2026/src/protocol/run.py)
12. [src/evaluation/engine.py](/workspace/2026/src/evaluation/engine.py)

After that, inspect the two final validation experiments:

13. [outputs/exp_20260313_153655/manifest.json](/workspace/2026/outputs/exp_20260313_153655/manifest.json)
14. [outputs/exp_20260313_153655/tables/summary_by_regime.csv](/workspace/2026/outputs/exp_20260313_153655/tables/summary_by_regime.csv)
15. [outputs/exp_20260317_202653/manifest.json](/workspace/2026/outputs/exp_20260317_202653/manifest.json)
16. [outputs/exp_20260317_202653/tables/summary_by_regime.csv](/workspace/2026/outputs/exp_20260317_202653/tables/summary_by_regime.csv)

If running inside Claude Code interactively, confirm loaded memory with `/memory`
before starting significant work.

## 3. Why This New Project Exists

The current experimental result is:

- `concat` failed scientific validation
- `channel_residual` also failed scientific validation
- both fail on:
  - `0/27` MMD q-value passes
  - `0/27` Energy q-value passes
  - `0/27` full statistical passes
  - only `2/27` PSD-ratio passes

Residual vs concat:

- residual improved some operational and second-order metrics
- residual was better than concat in many regimes for:
  - `|delta_evm|` (`17/27`)
  - `|delta_snr|` (`17/27`)
  - covariance (`23/27`)
  - PSD (`16/27`)
- residual was worse than concat in:
  - kurtosis (`27/27`)
- residual still failed `27/27` regimes overall

Conclusion:

- the main bottleneck is probably **not** just mean prediction parameterization
- the main bottleneck is likely the **point-wise modeling assumption**

## 4. Literature-Based Rationale

The external literature consulted outside the repo strongly points to
**time-domain sequence modeling** for VLC channel modeling:

- A 2021 VLC paper models the channel with **BiLSTM** in the time domain,
  using fixed-length waveform windows and real experimental signals.
- A 2022 digital-twin VLC paper states explicitly that `Din` and `Dout` are
  **time series** and compares **BiLSTM** and **BiGRU** for channel modeling.
- The same 2022 paper reports that **BiGRU** achieves similar modeling quality
  with lower computational cost than BiLSTM.
- A 2020 conditional-GAN paper is relevant as an alternative generative
  family, but switching from cVAE to GAN is a larger research jump and should
  not be the first move.

Therefore the recommended next architecture is:

```text
seq_bigru_residual
```

Meaning:

- sequence-aware
- BiGRU-based
- residual on the channel output
- still within the cVAE family

## 5. Core Hypothesis

Train a cVAE that predicts:

```text
p(y_t | x_{t-k:t+k}, d, I)
```

instead of the current point-wise:

```text
p(y_t | x_t, d, I)
```

The model should use a **local waveform window** around each sample, not just
the current I/Q point.

## 6. Non-Negotiable Pipeline Invariants

Do not violate these rules:

1. Split must remain **per experiment**, temporal `head=train`, `tail=val`
2. Windowing must happen **after split**
3. A window must never cross:
   - experiment boundaries
   - train/val boundaries
4. Reduction/cap still applies to **train only**
5. Evaluation remains based on:
   - deterministic inference for `EVM/SNR`
   - `mc_concat` for distribution metrics
6. Final scientific decision still comes from:
   - `manifest.json`
   - `summary_by_regime.csv`
   - `stat_fidelity_by_regime.csv`

## 7. Repository Map For This Project

This is the subset of the repo that matters for this architecture project:

```text
/workspace/2026
├── src/
│   ├── config/
│   │   ├── defaults.py
│   │   └── schema.py
│   ├── data/
│   │   └── loading.py
│   ├── evaluation/
│   │   └── engine.py
│   ├── models/
│   │   └── cvae.py
│   ├── protocol/
│   │   ├── run.py
│   │   └── split_strategies.py
│   └── training/
│       ├── grid_plan.py
│       └── pipeline.py
├── docs/
│   ├── PROTOCOL.md
│   └── CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md
├── outputs/
│   ├── exp_20260313_153655/
│   └── exp_20260317_202653/
├── PROJECT_STATUS.md
└── TRAINING_PLAN.md
```

## 8. Files To Touch

Expected write scope for this project:

- [src/config/defaults.py](/workspace/2026/src/config/defaults.py)
- [src/config/schema.py](/workspace/2026/src/config/schema.py)
- [src/models/cvae.py](/workspace/2026/src/models/cvae.py)
- new file recommended: [src/models/cvae_sequence.py](/workspace/2026/src/models/cvae_sequence.py)
- new file recommended: [src/data/windowing.py](/workspace/2026/src/data/windowing.py)
- [src/training/pipeline.py](/workspace/2026/src/training/pipeline.py)
- [src/training/grid_plan.py](/workspace/2026/src/training/grid_plan.py)
- [src/evaluation/engine.py](/workspace/2026/src/evaluation/engine.py)
- [src/protocol/run.py](/workspace/2026/src/protocol/run.py)
- tests to add under `/workspace/2026/tests/`

Files to read but preferably avoid editing unless necessary:

- [src/data/loading.py](/workspace/2026/src/data/loading.py)
- [src/protocol/split_strategies.py](/workspace/2026/src/protocol/split_strategies.py)

Files not in scope:

- `knowledge/`
- `outputs/`
- `data/`
- external paper tooling

These out-of-scope areas are repeated in `CLAUDE.md` and `REVIEW.md` on purpose,
so both implementation sessions and review sessions stay aligned.

## 9. Recommended Technical Design

### 9.1 Architecture name

Add a new variant:

```text
arch_variant = "seq_bigru_residual"
```

### 9.2 Data representation

Use fixed-size centered windows over `X`.

Recommended first representation:

- `X_seq`: shape `(N, W, 2)`
- `Y_target`: shape `(N, 2)` for the **center sample only**
- `D`: shape `(N, 1)`
- `C`: shape `(N, 1)`

Do **not** start with full seq-to-seq prediction.

First objective:

```text
predict y_center from x_window
```

This keeps the output interface close to the current protocol and reduces
surface area.

### 9.3 Windowing policy

Recommended first implementation:

- odd `window_size`
- centered context
- `pad_mode="edge"` to preserve one prediction per raw sample
- `stride=1`

Why:

- keeps output length aligned with the current protocol
- avoids dropping edge samples
- makes protocol metrics easier to keep compatible

### 9.4 Sequence backbone

Use **BiGRU** as the default sequence summarizer.

Reason:

- literature support in VLC channel modeling
- lower cost than BiLSTM
- enough capacity for the first iteration

### 9.5 Suggested probabilistic structure

Keep the cVAE structure, but sequence-aware:

- prior:
  - input: `x_window`, `d`, `I`
  - output: `z_mean_p`, `z_log_var_p`
- encoder:
  - input: `x_window`, `d`, `I`, `y_center`
  - output: `z_mean_q`, `z_log_var_q`
- decoder:
  - input: `z`, `x_window`, `d`, `I`
  - output: `mean_I`, `mean_Q`, `logvar_I`, `logvar_Q`

Keep the residual formulation:

```text
delta_mean = f(z, x_window, d, I)
y_mean = x_center + delta_mean
```

### 9.6 Conditioning strategy

Recommended first implementation:

- repeat normalized `(d, I)` across the time axis
- concatenate repeated condition features to `x_window`
- let BiGRU process `[x_window, d_rep, I_rep]`

This is simpler and less fragile than adding FiLM or attention first.

## 10. Detailed Execution Plan

### Phase 0 — Orientation

Goal:

- understand current point-wise assumptions
- confirm exact files and interfaces

Actions:

1. read the files listed in Section 2
2. inspect the two protocol experiments in `outputs/`
3. confirm the current inference helpers expect point-wise `(X, D, C)`

Deliverable:

- no code yet
- a short internal note of the exact interfaces that must change

### Phase 1 — Config surface

Goal:

- introduce sequence-model parameters cleanly

Recommended new config keys:

- `arch_variant = "seq_bigru_residual"`
- `window_size`
- `window_stride`
- `window_pad_mode`
- `seq_hidden_size`
- `seq_num_layers`
- `seq_bidirectional`

Files:

- `src/config/defaults.py`
- `src/config/schema.py`

Rules:

- keep current defaults fully backward compatible
- the default project behavior must remain `concat`

### Phase 2 — Window builder

Goal:

- convert split experiment arrays into sequence windows

Create:

- `src/data/windowing.py`

Core functions to add:

- `build_windows_single_experiment(...)`
- `build_windows_from_split_arrays(...)`
- optionally `windowize_experiments_before_concat(...)`

Important rules:

- no crossing experiment boundaries
- no crossing train/val boundaries
- windowing happens after split
- support edge padding

Expected output:

- `X_seq_train`, `Y_train`, `D_train`, `C_train`
- `X_seq_val`, `Y_val`, `D_val`, `C_val`

### Phase 3 — Sequence model module

Goal:

- implement a clean sequence-aware cVAE path

Recommended new file:

- `src/models/cvae_sequence.py`

Why:

- keep `src/models/cvae.py` readable
- avoid mixing dense and sequence logic into one large file

Implement:

- sequence context encoder based on BiGRU
- sequence-aware prior
- sequence-aware encoder
- sequence-aware decoder

Keep:

- same KL/loss layer
- same log-variance clamp behavior
- same train/inference semantics

### Phase 4 — Thin dispatch from existing cVAE entrypoint

Goal:

- preserve existing public API

Modify:

- `src/models/cvae.py`

Strategy:

- keep `build_cvae()` as the canonical entrypoint
- dispatch to the point-wise path for:
  - `concat`
  - `channel_residual`
- dispatch to the sequence path for:
  - `seq_bigru_residual`

Do not break:

- model saving
- layer names `encoder`, `prior_net`, `decoder`

These names matter because evaluation and protocol code look them up.

### Phase 5 — Training integration

Goal:

- make the training pipeline feed sequence inputs when needed

Modify:

- `src/training/pipeline.py`

Behavior:

- if `arch_variant` is point-wise:
  - keep existing arrays
- if `arch_variant` is sequence-aware:
  - build windowed train/val arrays after split and after train-only cap/reduction

Critical:

- normalization for `d` and `I` stays conceptually the same
- training summary and state writing should record sequence config

### Phase 6 — Inference and evaluation integration

Goal:

- keep protocol metrics working for the new sequence model

Modify:

- `src/evaluation/engine.py`
- `src/protocol/run.py`

Current issue:

- `_quick_cvae_predict()` assumes raw `X_va` can be fed directly as `(N,2)`
- sequence model will need windows `(N,W,2)`

Required adaptation:

- detect sequence-aware `arch_variant`
- build inference windows from raw `X_va` before prediction
- preserve one output per original sample

Important:

- deterministic path must still support `EVM/SNR`
- stochastic `mc_concat` path must still support distribution metrics

### Phase 7 — Grid integration

Goal:

- expose minimal sequential grids without exploding the search space

Modify:

- `src/training/grid_plan.py`

Add:

- `_seq_bigru_residual_candidates()`
- preset:
  - `seq_residual_smoke`
  - `seq_residual_small`

Initial recommendation:

- do not start with a large grid
- start with `latent_dim=4`
- use 2 to 4 candidate configs only

### Phase 8 — Tests

Goal:

- catch shape/interface regressions early

Add tests for:

1. window builder shape correctness
2. no cross-boundary leakage
3. sequence cVAE build succeeds
4. inference model for sequence variant builds
5. protocol helper can run prediction with sequence variant on toy arrays

Suggested new tests:

- `tests/test_windowing.py`
- `tests/test_seq_cvae_build.py`
- `tests/test_seq_inference.py`

### Phase 9 — Smoke experiments

Goal:

- verify plumbing before spending hours in protocol

Recommended order:

1. unit tests
2. 1-grid, 1-experiment, 2-epoch smoke in `src.training.train`
3. 1-regime protocol smoke
4. small global exploratory run
5. full protocol only after the above pass

### Phase 10 — Scientific validation

Goal:

- compare the sequence model against the current concat/residual baselines

Reference runs:

- concat reference:
  - `outputs/exp_20260313_153655`
- residual reference:
  - `outputs/exp_20260317_202653`

Primary acceptance question:

- does `seq_bigru_residual` improve **scientific protocol outcomes**, not just train loss?

Main comparison targets:

- `summary_by_regime.csv`
- `stat_fidelity_by_regime.csv`
- counts of:
  - `pass_mmd_qval`
  - `pass_energy_qval`
  - `pass_both_qval`
  - `pass_psd_ratio`
- regime-wise improvements against baseline

## 11. Recommended First Experimental Settings

Do not start with the literature-sized window immediately if it is too heavy.

Recommended sequence settings for the first implementation:

- `window_size = 33` for plumbing smoke
- `window_size = 65` for first serious exploratory run
- optional later candidate: `129`

Reason:

- the literature uses larger time windows, but this repo already runs large
  batches and a full global protocol is expensive
- first prove the software path and training stability

Recommended first model settings:

- `latent_dim = 4`
- `seq_hidden_size = 64` or `128`
- `seq_num_layers = 1`
- `seq_bidirectional = True`
- `beta in {0.001, 0.002, 0.003}`
- `free_bits in {0.0, 0.10}`

## 12. Commands Claude Should Expect To Use

### Quick tests

```bash
cd /workspace/2026
python -m pytest tests -q
```

### Targeted smoke train

```bash
cd /workspace/2026
python -m src.training.train \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --run_id seq_bigru_residual_smoke \
  --grid_preset seq_residual_smoke \
  --max_grids 1 \
  --max_experiments 1 \
  --max_samples_per_exp 200000 \
  --max_epochs 2 \
  --keras_verbose 2
```

### One-regime protocol smoke

```bash
cd /workspace/2026
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA.json \
  --train_once_eval_all \
  --grid_preset seq_residual_smoke \
  --max_grids 1 \
  --max_epochs 2 \
  --stat_tests \
  --stat_mode quick
```

### Small global exploratory run

```bash
cd /workspace/2026
python -m src.training.train \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --run_id seq_bigru_residual_small \
  --grid_preset seq_residual_small \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6
```

## 13. Stop Criteria

Stop and reassess if any of these happen:

1. the sequence path breaks backward compatibility for `concat`
2. inference cannot preserve one output per original sample
3. windowing leaks across split boundaries
4. smoke train produces NaNs or unusable KL explosion across all configs
5. protocol cannot consume the saved sequence model via the existing layer-name conventions

## 14. Fallback If The Full Sequential cVAE Is Too Invasive

Only if the full plan becomes blocked:

Fallback A:

- implement a smaller `gated_residual` point-wise decoder with stronger
  conditioning on `(distance, current)`

Fallback B:

- implement sequence conditioning only in the decoder first

Do **not** jump to conditional GAN before completing the sequential cVAE attempt.

## 15. Definition of Success

Success is **not**:

- lower train loss
- lower validation loss alone
- prettier reconstructions alone

Success is:

- a stable sequence-aware cVAE integrated into the repo
- smoke tests passing
- protocol runs completing end-to-end
- scientific improvement over both:
  - `concat`
  - `channel_residual`

Minimum scientific sign of progress:

- more than `0/27` MMD or Energy q-value passes
- better `validation_status` distribution than the current baselines
- reduced `0.8 m` failure severity without regressing `1.5 m` as hard as the residual point-wise model did
