# Worktree and Branch Policy

## Worktrees oficiais

- `/home/rodrigo/cVAe_2026`
  - integracao/documentacao
- `/home/rodrigo/cVAe_2026_full_square`
  - linha `full_square`
- `/home/rodrigo/cVAe_2026_full_circle`
  - linha `full_circle`

## Taxonomia de branches

- `main`
- `research/full-square`
- `research/full-circle`
- `arch/full-square/<arquitetura>`
- `arch/full-circle/<arquitetura>`
- `exp/full-square/<arquitetura>/<hipotese>`
- `exp/full-circle/<arquitetura>/<hipotese>`

## Regra operacional

- commits de trabalho diario acontecem em `arch/...`
- consolidacao parcial acontece em `research/...`
- consolidacao para apresentacao/documentacao acontece em `main`

## Branches iniciais registradas

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

## Observacao local importante

O worktree `/home/rodrigo/cVAe_2026` ainda possui modificacoes locais
preexistentes e, neste momento, esta staged localmente em
`wip/canonical-reorg-base-20260422`, nao em `main`. A reorganizacao canonica
foi preparada sem sobrescrever esse estado local.
