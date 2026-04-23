# Repo Governance

## Objetivo

Definir a governanca do repositorio canonico da pesquisa VLC cVAE.

## Regra principal

O programa de pesquisa e apresentado com duas linhas principais:

- `full_square`
- `full_circle`

Nenhuma terceira arquitetura ou modulacao deve ser apresentada como linha
principal paralela a essas duas.

## Repositorio canonico

Repositorio base:

- `/home/rodrigo/cVAe_2026`

Worktrees oficiais de linha:

- `/home/rodrigo/cVAe_2026_full_square`
- `/home/rodrigo/cVAe_2026_full_circle`

## Politica de integracao

- `main` concentra documentacao, `Tese`, manifests e estado consolidado
- `research/full-square` concentra a linha `full_square`
- `research/full-circle` concentra a linha `full_circle`
- `arch/...` concentra trabalho por arquitetura
- `exp/...` so deve ser usado para screening curto

## Benchmark de modulacoes

`16QAM`, `4QAM` e futuras modulacoes pertencem a uma camada transversal de
benchmark. Elas nao definem uma linha principal separada.

## Clones legados

Os clones abaixo sao mantidos como fontes de migracao e rastreabilidade:

- `/home/rodrigo/cVAe_2026_shape`
- `/home/rodrigo/cVAe_2026_shape_fullcircle`
- `/home/rodrigo/cVAe_2026_mdn_return`

Esses clones nao devem ser usados como forma principal de apresentar a
organizacao do projeto para novos colaboradores ou orientadores.
