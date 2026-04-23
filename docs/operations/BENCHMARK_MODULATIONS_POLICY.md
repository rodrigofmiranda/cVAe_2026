# Benchmark Modulations Policy

## Papel cientifico

As modulacoes externas servem para:

- validacao externa do gemeo digital
- comparacao entre `full_square` e `full_circle`
- extensao reutilizavel para novos cenarios

## Regra institucional

`16QAM`, `4QAM` e futuras modulacoes nao sao tratadas como uma terceira linha
principal do projeto.

Elas entram na camada:

- `benchmarks/modulations/<modulacao>/`

## Estrutura canonica

Para cada modulacao:

- dados: `data/benchmarks/modulations/<mod>/`
- scripts: `scripts/benchmarks/modulations/<mod>/`
- docs: `docs/benchmarks/modulations/<mod>/`
- syntheses: `knowledge/syntheses/benchmarks/modulations/<mod>/`
- outputs: `outputs/benchmarks/modulations/<mod>/`
- tese: `Tese/06_validacao_do_gemeo/<mod>/`

## Benchmark inicial

Primeira modulacao oficial:

- `16QAM`

Estrutura reservada para continuidade:

- `4QAM`
- `future`
