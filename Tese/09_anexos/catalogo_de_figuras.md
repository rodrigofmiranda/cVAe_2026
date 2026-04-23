# Catálogo de Figuras

## Propósito

Criar um índice inicial de figuras canônicas ou planejadas para a tese.

## Escopo

Nesta primeira versão, o catálogo é curado e orientado à redação. Ele não
varre automaticamente todos os plots existentes em `outputs/`.

## Fontes canônicas usadas

- [figuras_e_tabelas_planejadas](../08_planejamento_da_redacao/figuras_e_tabelas_planejadas.md)
- [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md)
- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)

## Status do texto

`rascunho`

## Síntese

| Figura ou família visual | Origem canônica | Papel na tese |
| --- | --- | --- |
| dashboard de treinamento por grid | artefatos `train/plots/training/` descritos no protocolo | mostrar convergência e estabilidade de treino |
| tabela-resumo de protocolo | `protocol_leaderboard.csv`, `summary.csv` e derivados | sintetizar comparação entre candidatos |
| heatmaps de validação por regime | artefatos de protocolo e plots auxiliares | mostrar onde os gargalos aparecem |
| comparação clean vs geometry-biased | outputs `full_circle` selecionados | tornar visível a diferença entre baseline e máximo operacional |
| plots auxiliares de `MI/GMI/NGMI` | `summary_plots.py` e saídas associadas | apoiar a leitura de validação externa |

## Implicações

Este catálogo ajuda a evitar que a tese seja montada no fim a partir de uma
busca desordenada por imagens já existentes.
