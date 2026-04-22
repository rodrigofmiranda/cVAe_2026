# Resultados por Linha

## Propósito

Concentrar, em uma só página, a leitura estabilizada dos resultados das linhas
principais.

## Escopo

Este documento compara, em nível de tese, `full_square`, `full_circle` e
validação do twin.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)
- [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
- [support_scientific_screen_master_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/support_scientific_screen_master_table_2026-04-10.md)

## Status do texto

`curado`

## Síntese

| Linha | Resultado estabilizado | Leitura científica |
| --- | --- | --- |
| `full_square` | melhores referências chegaram a `10/12`, com gargalo remanescente localizado | linha principal que entendeu o problema |
| `shape` sobre `full_square` | suporte-aware ajudou como triagem, mas o controle `E2` permaneceu como referência mais robusta | hipótese geométrica plausível, ainda não fechada |
| `full_circle` geometry-biased | melhor resultado operacional observado: `8/12` | mostra ganho operacional sob viés geométrico |
| `full_circle` clean | melhor resultado clean observado: `5/12` | baseline científica ainda fraca |
| validação do twin | `G1..G5` consolidados como validação principal; `G6` reposicionado como tela auxiliar | maturação metodológica importante |
| validação externa | `16QAM` reconhecido como checagem forte de generalização | papel já estabelecido, consolidação numérica ainda a completar |

## Leitura integrada

A tese hoje já pode sustentar três afirmações fortes:

1. o problema foi entendido e localizado em `full_square`
2. a hipótese geométrica é plausível, mas ainda não foi confirmada de forma
   limpa pela baseline `full_circle`
3. a metodologia de validação amadureceu a ponto de separar aceitação principal
   do twin, screen estatístico e checagem externa

## Implicações

Este quadro deve orientar a ordem dos capítulos de resultados: primeiro
`full_square`, depois `full_circle`, depois validação e discussão.
