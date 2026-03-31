# VLC Channel Digital Twin - cVAE

Public research repository for a data-driven digital twin of a Visible Light
Communication (VLC) channel using Conditional Variational Autoencoders (cVAE)
and related generative baselines.

The central task is to learn the conditional channel distribution

$$p(y \mid x, d, c)$$

from synchronized I/Q measurements, where:

- `x` is the transmitted baseband sample
- `y` is the received sample after the physical channel
- `d` is LED-to-photodetector distance
- `c` is LED drive current

The goal is distributional fidelity, not only mean prediction.

## Start Here

If you landed on this repository through GitHub and are not sure where to
start, use this order:

1. [docs/BRANCH_GUIDE.md](docs/BRANCH_GUIDE.md) - what each public branch is for
2. [PROJECT_STATUS.md](PROJECT_STATUS.md) - current scientific and codebase status
3. [docs/README.md](docs/README.md) - documentation map
4. [docs/PROTOCOL.md](docs/PROTOCOL.md) - canonical experiment runner
5. [docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt](docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt) - end-to-end acquisition-to-training data path

Internal team members using the shared server should also read:

- [docs/active/INFRA_GUIDE.md](docs/active/INFRA_GUIDE.md) - SSH, Unix users, Docker, tmux, Git LFS

Important GitHub note:

- the README shown on the GitHub website changes with the selected branch
- the default public landing branch is currently `main`
- if you want the latest coordinated research line, check the branch guide first

## Which Branch Should I Use?

For most people, there are only three meaningful starting points:

| Branch | Use it when | Notes |
|---|---|---|
| `main` | you want the public landing page and a stable overview | default branch on GitHub |
| `feat/mdn-g5-recovery-explore-remote` | you want the currently coordinated remote research line | best default choice for active collaboration right now |
| `release/cvae-online` | you want a release-style snapshot instead of live experimentation | more conservative than the active research branch |

The full map of public branches is in [docs/BRANCH_GUIDE.md](docs/BRANCH_GUIDE.md).

## Clone The Current Active Work

To copy the current active remote work into a fresh local clone:

```bash
git clone https://github.com/rodrigofmiranda/cVAe_2026.git
cd cVAe_2026
git fetch --all --prune
git switch feat/mdn-g5-recovery-explore-remote
git lfs install --local
git lfs pull
```

Without `git lfs pull`, the dataset tree may exist but the large files will not
actually be present.

## Quick Run

The canonical entrypoint is:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/protocol_default.json
```

For the sequential family `seq_bigru_residual`, also use:

```bash
--no_data_reduction
```

because the default balanced-block reduction breaks temporal context.

## Modeling Summary

This repository currently keeps multiple model families behind a shared
protocol/training path:

- `concat` - original point-wise cVAE
- `channel_residual` - point-wise residual decoder
- `delta_residual` - point-wise residual-target family
- `seq_bigru_residual` - sequence-aware residual family
- `legacy_2025_zero_y` - historical notebook-era reference line

In day-to-day research, architecture selection is usually done by
`arch_variant`, `grid_tag`, or `grid_preset`, not by opening a separate repo.

## Documentation

- [docs/BRANCH_GUIDE.md](docs/BRANCH_GUIDE.md) - public branch map
- [docs/README.md](docs/README.md) - documentation index
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - current codebase and science status
- [docs/PROTOCOL.md](docs/PROTOCOL.md) - protocol runner and artifacts
- [docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt](docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt) - acquisition-to-training data path
- [docs/MODELING_ASSUMPTIONS.md](docs/MODELING_ASSUMPTIONS.md) - modeling rationale
- [docs/DIAGNOSTIC_CHECKLIST.md](docs/DIAGNOSTIC_CHECKLIST.md) - diagnostic workflow
- [docs/active/INFRA_GUIDE.md](docs/active/INFRA_GUIDE.md) - internal server onboarding and isolation guide

## Repository Layout

```text
configs/        Protocol and grid configuration
data/           Dataset stored with Git LFS
docker/         Container build and runtime assets
docs/           Public docs, internal guides, archive, references
notebooks/      Exploratory notebooks
outputs/        Experiment artifacts
scripts/        Operational and analysis helpers
src/            Core training, evaluation, protocol, and model code
tests/          Unit and integration tests
```

## For New Collaborators

If your goal is to understand the program and copy the current work:

1. Read [docs/BRANCH_GUIDE.md](docs/BRANCH_GUIDE.md)
2. Clone the repository and switch to `feat/mdn-g5-recovery-explore-remote`
3. Run `git lfs pull`
4. Read [PROJECT_STATUS.md](PROJECT_STATUS.md)
5. Read [docs/PROTOCOL.md](docs/PROTOCOL.md)
6. Read [docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt](docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt)
7. If you are using the shared lab server, follow [docs/active/INFRA_GUIDE.md](docs/active/INFRA_GUIDE.md)
