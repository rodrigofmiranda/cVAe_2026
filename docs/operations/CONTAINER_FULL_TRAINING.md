# Container Full Training Runbook

Canonical end-to-end flow for running a full protocol training job inside the
project GPU container.

Use this runbook when the goal is:

- start or enter the persistent container session
- launch a full `train_once_eval_all` protocol run
- monitor logs without losing the session
- inspect the final artifacts under `outputs/exp_*`

This runbook assumes:

- repo root: `/home/eduardo/cVAe_2026`
- mounted container workdir: `/workspace/2026/feat_seq_bigru_residual_cvae`
- default image: `vlc/tf25-gpu-ready:1`
- canonical wrapper: `scripts/ops/train.sh`

## 1. Pre-flight on the host

From the host clone:

```bash
cd /home/eduardo/cVAe_2026
git status -sb
ls data/dataset_fullsquare_organized
```

Optional quick checks before consuming GPU time:

```bash
test -x scripts/ops/run_tf25_gpu.sh
test -x scripts/ops/train.sh
```

If you need the current image name or tmux session defaults:

```bash
sed -n '1,120p' scripts/ops/run_tf25_gpu.sh
```

## 2. Start the persistent GPU container

Preferred launcher:

```bash
cd /home/eduardo/cVAe_2026
bash scripts/ops/run_tf25_gpu.sh
```

What this does:

- creates or reuses tmux session `cvae_tf25_gpu`
- starts container `cvae_tf25_gpu`
- mounts the repo into `/workspace/2026/feat_seq_bigru_residual_cvae`
- sources `scripts/ops/container_bootstrap_python.sh`
- bootstraps persistent repo-local Python deps into `.pydeps` when needed

Enter the running session:

```bash
bash scripts/ops/enter_tf25_gpu.sh
```

If you prefer raw tmux:

```bash
tmux attach -t cvae_tf25_gpu
```

## 3. Validate the container session

Inside the container:

```bash
pwd
python3 -V
python3 -c "import tensorflow as tf; print(tf.__version__)"
python3 -c "import numpy, pandas, matplotlib, openpyxl, pytest; print('python deps ok')"
```

Optional GPU visibility check:

```bash
nvidia-smi
```

## 4. Launch the full training run

Inside the container, from the repo root:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
bash scripts/ops/train.sh
```

This wrapper runs the canonical protocol:

```bash
python -u -m src.protocol.run \
  --dataset_root "$REPO_ROOT/data/dataset_fullsquare_organized" \
  --output_base "$REPO_ROOT/outputs" \
  --train_once_eval_all
```

Important behavior:

- training artifacts go under `outputs/exp_YYYYMMDD_HHMMSS/train/`
- per-regime evaluation artifacts go under `outputs/exp_YYYYMMDD_HHMMSS/eval/`
- top-level protocol tables go under `outputs/exp_YYYYMMDD_HHMMSS/tables/`
- the latest completed experiment record is written to `outputs/_latest_completed_experiment.json`

## 5. Recommended full-run variants

Full reduced protocol with the sequence family:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -u -m src.protocol.run \
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

Full 27-regime universal-twin run:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python -u -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_full_dataset.json \
  --train_once_eval_all \
  --max_epochs 120 \
  --max_grids 2
```

Rule for `seq_bigru_residual`:

- always add `--no_data_reduction`

## 6. Monitor without losing the run

Detach from tmux and keep the job alive:

```text
Ctrl-b d
```

Re-enter later:

```bash
cd /home/eduardo/cVAe_2026
bash scripts/ops/enter_tf25_gpu.sh
```

Useful checks from another host shell:

```bash
tmux ls
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

If you want the live protocol process list inside the container:

```bash
pgrep -af "src.protocol.run|src.training.train"
```

## 7. Review outputs after completion

The canonical finished run looks like:

```text
outputs/exp_YYYYMMDD_HHMMSS/
├── train/
├── eval/
├── logs/
├── tables/
└── manifest.json
```

First review pass:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
python scripts/analysis/summarize_experiment.py
```

Or target a specific run:

```bash
python scripts/analysis/summarize_experiment.py outputs/exp_YYYYMMDD_HHMMSS
```

Core files to inspect:

- `outputs/exp_YYYYMMDD_HHMMSS/manifest.json`
- `outputs/exp_YYYYMMDD_HHMMSS/tables/protocol_leaderboard.csv`
- `outputs/exp_YYYYMMDD_HHMMSS/tables/summary_by_regime.csv`
- `outputs/exp_YYYYMMDD_HHMMSS/train/tables/gridsearch_results.csv`
- `outputs/_latest_completed_experiment.json`

## 8. Common failure points

Missing Python plotting/runtime modules:

- re-enter via `scripts/ops/run_tf25_gpu.sh` or source `scripts/ops/container_bootstrap_python.sh`

Dataset not found:

- confirm `data/dataset_fullsquare_organized` exists in the host clone
- confirm the repo is mounted into the container workdir expected by `run_tf25_gpu.sh`

Sequence model behaving incorrectly on reduced samples:

- for `seq_bigru_residual`, do not omit `--no_data_reduction`

No finished experiment selected by tools:

- inspect `outputs/exp_*`
- confirm `manifest.json` and `tables/summary_by_regime.csv` exist

## 9. Stop the container session

From the host:

```bash
cd /home/eduardo/cVAe_2026
bash scripts/ops/stop_tf25_gpu.sh
```

If the training is still running, stop it deliberately from inside tmux first.
