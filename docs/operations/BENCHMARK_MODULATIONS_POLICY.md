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

Essa camada e conceitual e metodologica. Ela nao deve virar um deposito
central de resultados.

## Estrutura canonica

Para cada modulacao:

- dados: no worktree da linha que executa a avaliacao, tipicamente `data/16qam/`
  ou caminho explicitamente passado por `--dataset_root`
- scripts: wrappers compartilhados em `scripts/benchmarks/modulations/<mod>/`
- docs: `docs/benchmarks/modulations/<mod>/`
- syntheses: `knowledge/syntheses/benchmarks/modulations/<mod>/`
- outputs: sempre junto da arquitetura/candidato testado, nunca em um
  acumulador central no repo-base
- tese: `Tese/06_validacao_do_gemeo/<mod>/`

Padrao fisico dos outputs:

```text
/home/rodrigo/cVAe_2026_full_square/outputs/architectures/<arquitetura>/<candidato>/benchmarks/<mod>/
/home/rodrigo/cVAe_2026_full_circle/outputs/architectures/<arquitetura>/<candidato>/benchmarks/<mod>/
```

Quando houver comparacao crossline, o sumario pode existir como indice
derivado, mas deve apontar para os artefatos locais de cada candidato. O
sumario nao substitui os manifests locais.

## Benchmark inicial

Primeira modulacao oficial:

- `16QAM`

Estrutura reservada para continuidade:

- `4QAM`
- `future`
