# PROJECT_STATUS

> Atualizado em 2026-04-22.
> Este arquivo descreve a organizacao canonica atual do programa de pesquisa.

## Estrutura canonica oficial

O programa passa a ser apresentado como um repositorio canonico unico com duas
linhas principais de pesquisa e uma camada transversal de benchmarks.

### Linhas principais

- `full_square`
- `full_circle`

### Camada transversal

- `benchmarks/modulations/16qam`
- `benchmarks/modulations/4qam`
- `benchmarks/modulations/future`

## Worktrees oficiais

1. `/home/rodrigo/cVAe_2026`
   - papel: integracao, documentacao, manifests e camada `Tese`
   - branch-alvo: `main`
2. `/home/rodrigo/cVAe_2026_full_square`
   - papel: linha principal `full_square`
   - branch longa: `research/full-square`
3. `/home/rodrigo/cVAe_2026_full_circle`
   - papel: linha principal `full_circle`
   - branch longa: `research/full-circle`

## Regra de branches

- `main`
  - integracao e apresentacao
- `research/full-square`
  - consolidacao da linha `full_square`
- `research/full-circle`
  - consolidacao da linha `full_circle`
- `arch/full-square/<arquitetura>`
  - trabalho por arquitetura dentro de `full_square`
- `arch/full-circle/<arquitetura>`
  - trabalho por arquitetura dentro de `full_circle`
- `exp/...`
  - screening curto e descartavel quando necessario

## Arquiteturas iniciais registradas

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

## Status cientifico da reorganizacao

A reorganizacao atual e nao-destrutiva.

- clones legados continuam disponiveis como fonte historica
- outputs historicos nao foram movidos em massa
- a camada `Tese` foi centralizada no repositorio canonico
- a documentacao passou a separar explicitamente `full_square`, `full_circle`
  e benchmark de modulacoes

## Clones legados agora tratados como apoio de migracao

- `/home/rodrigo/cVAe_2026_shape`
- `/home/rodrigo/cVAe_2026_shape_fullcircle`
- `/home/rodrigo/cVAe_2026_mdn_return`

Eles continuam uteis para rastreabilidade, mas nao devem mais ser apresentados
como a organizacao canonica final do programa.

## Benchmark transversal de modulacoes

Papel oficial:

- validacao externa
- comparacao entre linhas
- camada reutilizavel para `16QAM`, `4QAM` e futuras modulacoes

`16QAM` e o primeiro benchmark oficial dessa camada.
`4QAM` ja possui a estrutura reservada para continuidade futura.

## Documentos de governanca

- `docs/operations/REPO_GOVERNANCE.md`
- `docs/operations/WORKTREE_BRANCH_POLICY.md`
- `docs/operations/BENCHMARK_MODULATIONS_POLICY.md`
- `docs/reference/CANONICAL_REPO_LAYOUT.md`
- `Tese/README.md`
