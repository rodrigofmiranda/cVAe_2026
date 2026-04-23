# VLC Channel Digital Twin - Canonical Research Layout

Canonical repository for the VLC channel digital-twin program.

The project is now organized around two primary research lines:

- `full_square`
- `full_circle`

External modulations such as `16QAM`, `4QAM`, and future variants are treated
as a transversal benchmark layer used to compare the two primary lines under a
shared validation framework.

## Canonical Git Model

Official local worktrees:

- `/home/rodrigo/cVAe_2026`
  - integration/documentation base
  - reference branch target: `main`
- `/home/rodrigo/cVAe_2026_full_square`
  - main worktree for the `full_square` line
  - long-lived branch: `research/full-square`
- `/home/rodrigo/cVAe_2026_full_circle`
  - main worktree for the `full_circle` line
  - long-lived branch: `research/full-circle`

Architecture work happens on `arch/...` branches. Benchmarking work happens in
`benchmarks/modulations/...`.

## Start Here

1. [PROJECT_STATUS.md](PROJECT_STATUS.md)
2. [docs/README.md](docs/README.md)
3. [docs/operations/REPO_GOVERNANCE.md](docs/operations/REPO_GOVERNANCE.md)
4. [docs/operations/WORKTREE_BRANCH_POLICY.md](docs/operations/WORKTREE_BRANCH_POLICY.md)
5. [docs/operations/ADVISOR_REORGANIZATION_BRIEF_2026-04-22.md](docs/operations/ADVISOR_REORGANIZATION_BRIEF_2026-04-22.md)
6. [Tese/README.md](Tese/README.md)

## Repository Layout

```text
Tese/            Curated thesis-ready material
configs/         Common, line-specific, and benchmark configuration layers
data/            Runtime datasets and benchmark data roots
knowledge/       Notes and syntheses by line and by benchmark
docs/            Operations, lineage, references, benchmark guides
outputs/         Runtime artifacts split by line and by modulation benchmark
scripts/         Common wrappers plus line-specific and benchmark entrypoints
src/             Shared code plus line/benchmark logical namespaces
tests/           Automated validation
```

## Scientific Model

- `full_square` and `full_circle` are the two principal scientific approaches.
- `mdn_return` is now treated as a historical lineage inside `full_square`.
- `16QAM` is the first canonical external benchmark.
- `4QAM` and later modulations enter the same benchmark layer without changing
  the repository shape.
- the current staging branch for the canonical reorganization is
  `wip/canonical-reorg-base-20260422`

## Migration Notes

- active protocol configs now live canonically in `configs/full_square/`
- legacy root config paths remain available for compatibility
- canonical `16QAM` benchmark scripts now live in
  `scripts/benchmarks/modulations/16qam/`
- `data/` and `outputs/` now expose tracked README placeholders for the new
  line/benchmark split without versioning runtime artifacts

## Operational Note

Legacy local clones such as `cVAe_2026_shape`, `cVAe_2026_shape_fullcircle`,
and `cVAe_2026_mdn_return` remain available as migration sources, but they are
no longer the intended canonical presentation layout.
