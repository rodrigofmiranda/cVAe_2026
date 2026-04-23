# Canonical Repository Layout

Este documento descreve a organizacao alvo do repositorio canonico.

## Linhas principais

As duas abordagens principais sao:

- `full_square`
- `full_circle`

## Camada transversal

Modulacoes externas entram como benchmark transversal:

- `16QAM`
- `4QAM`
- futuras modulacoes

## Estrutura logica

```text
Tese/
docs/
knowledge/
configs/
  common/
  full_square/
  full_circle/
  benchmarks/modulations/
scripts/
  common/
  full_square/
  full_circle/
  benchmarks/modulations/
src/
  common/
  full_square/
  full_circle/
  benchmarks/modulations/
data/
  README.md
outputs/
  README.md
```

## Regra de separacao

- hipoteses, execucao e sintese especificas de uma linha ficam em
  `full_square/` ou `full_circle/`
- a metodologia das comparacoes externas por modulacao fica em
  `benchmarks/modulations/...`
- os artefatos de avaliacao por modulacao ficam junto da arquitetura/candidato
  que foi testado, por exemplo
  `outputs/architectures/<arquitetura>/<candidato>/benchmarks/16qam/`
- infraestrutura compartilhada fica em `common/`
- `mdn_return` passa a ser documentado como linhagem do `full_square`

## Compatibilidade de migracao

- protocolos ativos da linha `full_square` vivem em `configs/full_square/`
- caminhos legados em `configs/*.json` e `configs/*.yaml` permanecem como
  compatibilidade
- scripts de benchmark `16QAM` vivem em
  `scripts/benchmarks/modulations/16qam/`
- caminhos antigos em `scripts/analysis/` permanecem como wrappers finos
- `cVAe_2026` nao deve ser usado como novo destino de outputs de benchmark
