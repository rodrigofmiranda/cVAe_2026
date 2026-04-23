# Public Branch Guide

The repository is now governed by one integration branch, two long-lived line
branches, and architecture branches under each line.

## Recommended Starting Points

| Branch | Use it when | Role |
|---|---|---|
| `main` | you want the canonical documentation and presentation layer | integration and public overview |
| `research/full-square` | you want the active `full_square` line | main branch for the square-support research family |
| `research/full-circle` | you want the active `full_circle` line | main branch for the circular-support research family |

## Canonical Taxonomy

- `main`
  - integration, documentation, `Tese`, manifests, branch map
- `research/full-square`
  - long-lived branch of the `full_square` line
- `research/full-circle`
  - long-lived branch of the `full_circle` line
- `arch/full-square/<arquitetura>`
  - architecture-specific work under `full_square`
- `arch/full-circle/<arquitetura>`
  - architecture-specific work under `full_circle`
- `exp/...`
  - optional short-lived experiment branches

## Initial Architecture Branches

### full_square

- `arch/full-square/probabilistic-shaping`
- `arch/full-square/mdn-return`
- `arch/full-square/pointwise-revival`
- `arch/full-square/seq-bigru-residual`

### full_circle

- `arch/full-circle/clean-baseline`
- `arch/full-circle/soft-local`
- `arch/full-circle/soft-rinf-local`
- `arch/full-circle/disk-geom3`

## Practical Guidance

If you want the canonical presentation for advisors or onboarding:

1. stay on `main`
2. read `README.md`
3. read `PROJECT_STATUS.md`
4. read `docs/operations/*.md`
5. read `Tese/README.md`

If you want to work on one main scientific line:

1. open the matching worktree
2. stay on `research/full-square` or `research/full-circle`
3. branch from the corresponding `arch/...` namespace for focused work

If you want to evaluate external modulations:

1. stay in the canonical repo
2. use the benchmark layer under `benchmarks/modulations/...`
3. compare `full_square` and `full_circle` using the same modulation protocol
