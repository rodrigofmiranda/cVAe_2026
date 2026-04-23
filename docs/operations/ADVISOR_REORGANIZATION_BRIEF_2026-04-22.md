# Resumo Executivo da Reorganizacao Canonica

## Objetivo

Apresentar uma organizacao unica, coerente e rastreavel do programa de
pesquisa VLC cVAE para orientacao, execucao tecnica e futura redacao da tese.

## Diagnostico do estado anterior

Antes desta reorganizacao, o trabalho estava distribuido em clones/worktrees
que refletiam momentos diferentes da pesquisa:

- `cVAe_2026_shape`
- `cVAe_2026_shape_fullcircle`
- `cVAe_2026_mdn_return`

Isso era util operacionalmente, mas ruim para apresentacao academica, porque:

- a separacao entre linhas principais e linhagens historicas nao ficava clara
- `16QAM` podia parecer uma terceira linha, quando na verdade e benchmark
- a organizacao de branches e worktrees nao refletia explicitamente a logica
  cientifica do projeto

## Modelo canonico adotado

O programa passa a ser apresentado com:

- **duas abordagens principais**
  - `full_square`
  - `full_circle`
- **uma camada transversal de validacao externa**
  - `16QAM`
  - `4QAM`
  - futuras modulacoes

## Regra cientifica

- `full_square` e `full_circle` sao as duas linhas principais
- `mdn_return` deixa de ser tratado como eixo principal e passa a ser uma
  linhagem historica de `full_square`
- `16QAM`, `4QAM` e futuras modulacoes nao sao novas linhas principais
  independentes; elas formam a camada de benchmark transversal

## Regra de Git e worktrees

Worktrees oficiais:

- `/home/rodrigo/cVAe_2026`
  - integracao, documentacao, manifests e camada `Tese`
- `/home/rodrigo/cVAe_2026_full_square`
  - linha principal `full_square`
- `/home/rodrigo/cVAe_2026_full_circle`
  - linha principal `full_circle`

Taxonomia oficial de branches:

- `main`
- `research/full-square`
- `research/full-circle`
- `arch/full-square/<arquitetura>`
- `arch/full-circle/<arquitetura>`
- `exp/...`

## Estrutura do repositorio

A organizacao canonica passa a separar:

- linha `full_square`
- linha `full_circle`
- benchmark de modulacoes
- camada curada `Tese`

Exemplos:

- `docs/lineage/full_square/`
- `docs/lineage/full_circle/`
- `docs/benchmarks/modulations/16qam/`
- `knowledge/syntheses/full_square/`
- `knowledge/syntheses/full_circle/`
- `knowledge/syntheses/benchmarks/modulations/16qam/`

Regra corrigida para outputs:

- resultados de benchmark ficam junto da arquitetura/candidato avaliado
- exemplo:
  `outputs/architectures/<arquitetura>/<candidato>/benchmarks/16qam/`
- o worktree `cVAe_2026` nao deve ser usado como novo deposito de resultados

## O que ja foi implementado

- criacao dos worktrees canonicos `cVAe_2026_full_square` e
  `cVAe_2026_full_circle`
- criacao das branches `research/...` e `arch/...` iniciais
- centralizacao da pasta `Tese/` no repositorio canonico
- criacao da documentacao de governanca
- criacao da documentacao e dos wrappers de benchmark transversal para
  modulacoes
- importacao dos documentos de linhagem principais de `full_square` e
  `full_circle`

## O que ainda permanece como transicao controlada

- clones legados continuam acessiveis para rastreabilidade
- outputs historicos nao foram movidos em massa
- o worktree-base ainda esta numa branch de staging local
  `wip/canonical-reorg-base-20260422`, preservando alteracoes preexistentes
  antes da integracao limpa em `main`
- a separacao fisica completa de codigo em `src/full_square` e
  `src/full_circle` foi iniciada, mas o refactor pesado do codigo compartilhado
  ainda nao foi feito

## Mensagem principal para orientacao

O projeto agora pode ser apresentado de forma simples:

1. existem **duas abordagens principais**: `full_square` e `full_circle`
2. existe uma **camada transversal de validacao externa** por modulacao
3. cada arquitetura tem sua **branch propria**
4. a pasta `Tese/` concentra a narrativa curada e rastreavel da pesquisa

## Documentos de apoio

- `docs/operations/REPO_GOVERNANCE.md`
- `docs/operations/WORKTREE_BRANCH_POLICY.md`
- `docs/operations/BENCHMARK_MODULATIONS_POLICY.md`
- `docs/reference/CANONICAL_REPO_LAYOUT.md`
- `Tese/README.md`
