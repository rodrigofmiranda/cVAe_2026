# Mapa de Worktrees e Branches

## Repositorio canonico

- `/home/rodrigo/cVAe_2026`
  - papel: integracao, documentacao e camada `Tese`
  - branch-alvo de integracao: `main`
  - branch local atual de staging: `wip/canonical-reorg-base-20260422`

## Worktrees ativos de linha

- `/home/rodrigo/cVAe_2026_full_square`
  - branch: `research/full-square`
- `/home/rodrigo/cVAe_2026_full_circle`
  - branch: `research/full-circle`

## Taxonomia oficial de branches

- `main`
- `research/full-square`
- `research/full-circle`
- `arch/full-square/<arquitetura>`
- `arch/full-circle/<arquitetura>`
- `exp/...`

## Observacao operacional

No momento, o worktree-base em `/home/rodrigo/cVAe_2026` preserva um estado
local preexistente e, por isso, a integracao limpa em `main` deve ser tratada
com cuidado em rodada propria de consolidacao.
