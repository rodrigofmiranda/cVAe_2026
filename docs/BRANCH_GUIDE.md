# Public Branch Guide

This repository keeps multiple public branches because the research evolved
through several architecture families and experimental lines. If you are new to
the project, do not start by guessing from the branch names alone.

## Recommended Starting Points

| Branch | Audience | Recommendation |
|---|---|---|
| `main` | visitors arriving from GitHub | start here for the public landing page |
| `feat/mdn-g5-recovery-explore-remote` | active collaborators | best branch to copy if you want the latest coordinated remote work |
| `release/cvae-online` | readers who want a release-style snapshot | use when you want something more conservative than the active research branch |

## How To Copy The Current Active Work

```bash
git clone https://github.com/rodrigofmiranda/cVAe_2026.git
cd cVAe_2026
git fetch --all --prune
git switch feat/mdn-g5-recovery-explore-remote
git lfs install --local
git lfs pull
```

If you are using the shared lab server, continue with
[active/INFRA_GUIDE.md](active/INFRA_GUIDE.md).

## How To Read This Branch List

- `main` is the public default branch, not necessarily the newest experiment line
- `feat/...` branches usually represent a research direction or architecture family
- `release/...` branches are snapshot-style branches
- `exp/...` branches are exploratory and should not be treated as the default entrypoint
- this guide covers the public remote branches visible on GitHub
- local-only branches may still exist on the server and are not guaranteed to appear here

## Historical Outputs On The Shared Server

If you are auditing old runs on the shared lab server, branch history and output
history do not line up perfectly in the current `outputs/` tree.

- active post-migration outputs live in the current clone under `outputs/`
- recovered pre-migration historical outputs were copied into the local-only directory `outputs/_recovered_vlc_backup_20260402`
- the recovered inventory is indexed in `outputs/_recovered_vlc_backup_20260402/inventory.csv`
- that recovered set currently preserves historical evidence for `feat/sample-aware-mmd`, `feat/sample-aware-mmd-gpu`, and `feat/seq-bigru-residual-cvae`
- `feat/delta-residual-adv` still exists mainly as a historical documentation trail; docs reference its former `/workspace/2026/feat_delta_residual_adv/...` runs, but that original worktree is not currently preserved on disk
- older manifests often store `git_commit` without `git_branch`, so branch attribution for legacy runs may still require commit ancestry or the recovered inventory
- newer protocol manifests now record both `git_commit` and `git_branch`, which should make future audits direct

## Branch Map

| Branch | Role | When to use it |
|---|---|---|
| `main` | public landing branch | use for the repository overview and stable public entrypoint |
| `release/cvae-online` | release-style snapshot | use when you want a more packaged reference branch |
| `exp/refactor_architecture` | refactor exploration | use only if you are working on structural code refactoring |
| `feat/channel-residual-architecture` | point-wise residual decoder line | use if you are studying the `channel_residual` family specifically |
| `feat/conditional-flow-decoder` | conditional flow decoder exploration | use only if you are reproducing or continuing the flow-decoder line |
| `feat/delta-residual-adv` | residual-target and adversarial-era experiments | use mainly for historical comparison or targeted recovery of that line |
| `feat/imdd-graybox-channel` | gray-box IM/DD channel line | use if your work is about the IM/DD gray-box formulation |
| `feat/mdn-g5-recovery` | core MDN G5 recovery line | use to inspect the main recovery direction that led into the current remote branch |
| `feat/mdn-g5-recovery-explore-remote` | current coordinated remote line | best default branch for active collaboration right now |
| `feat/mdn-g5-recovery-run` | run-oriented companion branch for the MDN recovery line | use only if you specifically need that run snapshot/history |
| `feat/sample-aware-mmd` | sample-aware MMD research line | use only if you are revisiting that loss family |
| `feat/sample-aware-mmd-gpu` | GPU-oriented continuation of sample-aware MMD | use only for reproducing that GPU-specific branch |
| `feat/sample-aware-mmd-gpu-local` | local/GPU sample-aware MMD companion line | use only if you specifically need the local GPU variant published online |
| `feat/seq-bigru-residual-cvae` | first major sequential residual cVAE line | use if you need the early seq baseline/history |
| `feat/seq-bigru-residual-diffusion` | sequential diffusion exploration | use only for the diffusion variant |
| `feat/seq-bigru-residual-diffusion-v2` | follow-up diffusion exploration | use only for the second diffusion iteration |
| `feat/seq-bigru-residual-mdn-route` | sequential MDN routing line | use if you are tracing the path that introduced the MDN family |
| `feat/seq-bigru-residual-spline-flow` | sequential spline-flow exploration | use only for reproducing the spline-flow branch |
| `feat/seq-bigru-residual-spline-flow-v2` | second spline-flow exploration | use only for the follow-up spline-flow branch |
| `feat/seq-imdd-graybox-mdn` | sequential IM/DD plus MDN line | use if your work is specifically on the seq IM/DD branch |

## Practical Guidance For New Users

If you want to understand the project first:

1. stay on `main`
2. read [../README.md](../README.md)
3. read [../PROJECT_STATUS.md](../PROJECT_STATUS.md)
4. read [PROTOCOL.md](PROTOCOL.md)

If you want to copy the work that the team is actively extending:

1. clone the repository
2. switch to `feat/mdn-g5-recovery-explore-remote`
3. run `git lfs pull`
4. if you are on the shared server, follow [active/INFRA_GUIDE.md](active/INFRA_GUIDE.md)

If you want to reproduce an older scientific line:

1. locate the matching family branch in the table above
2. read its branch-specific `README.md` and `PROJECT_STATUS.md` if present
3. expect historical assumptions, older docs, and different operational constraints

## Why Not Use A Separate "Guide Branch"?

A separate guide branch would still force users to discover and switch branches
before they could understand the repository. The better pattern is:

- keep `main` as the public doorway
- keep this file in `main`
- keep branch-specific details inside the branches themselves
- keep infrastructure/onboarding in [active/INFRA_GUIDE.md](active/INFRA_GUIDE.md)
