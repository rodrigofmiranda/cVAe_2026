# Visão Geral da Linha Full-Circle

## Propósito

Apresentar a linha `full_circle` como trilha científica separada da linha
`full_square`.

## Escopo

Este documento introduz o papel de `full_circle` e a razão pela qual ele não
deve ser tratado como simples continuação automática de `shape`.

## Fontes canônicas usadas

- [FULL_CIRCLE_NEXT_STEP](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_NEXT_STEP.md)
- [FULL_CIRCLE_CLEAN_RUN_CHECKLIST](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md)
- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)

## Status do texto

`curado`

## Pergunta científica

O que exatamente a linha `full_circle` tentou responder que a linha
`full_square` não podia responder sozinha?

## Síntese

`full_circle` foi introduzido para testar a hipótese de geometria de suporte em
dados reais. A motivação era simples:

- `shape` em `full_square` sugeria que bordas e cantos do suporte quadrado
  influenciavam o gargalo
- isso não bastava para concluir que uma nova geometria de aquisição resolveria
  o problema

Portanto, `full_circle` foi tratado como linha separada, com duas leituras que
devem permanecer distintas:

- trilha geometry-biased, útil para exploração operacional
- trilha clean, necessária para baseline científica

## Leitura correta

- `full_circle` não confirma automaticamente as conclusões de `shape`
- melhor número operacional não equivale a melhor comparação científica
- o papel desta linha é testar a hipótese geométrica com disciplina
  metodológica

## Evidências

- A necessidade de restart limpo e separado já estava explicitada em
  [FULL_CIRCLE_NEXT_STEP](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_NEXT_STEP.md).
- A execução prática dessa separação foi documentada em
  [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md).

## Implicações

Na tese, `full_circle` deve aparecer como experimento de validação da hipótese
geométrica, não como simples extensão do melhor proxy anterior.
