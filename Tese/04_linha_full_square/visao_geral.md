# Visão Geral da Linha Full-Square

## Propósito

Apresentar uma leitura condensada da linha `full_square`, que foi a principal
trilha de desenvolvimento e entendimento do problema.

## Escopo

Este documento introduz a linha como um todo. Os subdocumentos desta pasta
detalham etapas específicas da evolução.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [WORKING_STATE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/WORKING_STATE.md)
- [PROJECT_STATUS](/home/rodrigo/cVAe_2026_mdn_return/PROJECT_STATUS.md)

## Status do texto

`curado`

## Pergunta científica

Como a linha `full_square` evoluiu de um problema amplo de modelagem para um
gargalo residual localizado e interpretável?

## Síntese

A linha `full_square` foi onde o problema foi refinado. Ela começou como
modelagem gerativa condicionada ponto a ponto e evoluiu até uma leitura muito
mais específica:

- o problema não era apenas média condicional
- o gargalo não era global
- a maior dificuldade ficou concentrada em regimes curtos e de baixa corrente
- parte da dificuldade parecia ligada ao shape residual e possivelmente à
  geometria do suporte

## Etapas principais

1. baseline point-wise tradicional
2. famílias residuais
3. família sequencial `seq_bigru_residual`
4. recuperação via `MDN`
5. point-wise revival local
6. reinterpretação `shape`
7. screening científico A/B/C/D

## Resultado histórico mais importante

O legado principal de `full_square` não é um único número. É a capacidade de
localizar o problema remanescente e transformar a busca de modelos em um
programa científico mais focalizado.

## Evidências

- A narrativa histórica consolidada está em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).

## Implicações

Na tese, `full_square` deve aparecer como a linha que gerou o entendimento do
problema, mesmo quando os melhores números finais forem discutidos em conjunto
com outras linhas.
