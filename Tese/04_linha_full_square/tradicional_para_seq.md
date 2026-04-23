# Tradicional para Seq

## Propósito

Explicar a primeira grande transição metodológica da linha `full_square`: da
leitura point-wise tradicional para a família sequencial residual.

## Escopo

Este texto cobre a mudança conceitual, não um catálogo completo dos grids
iniciais.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [WORKING_STATE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/WORKING_STATE.md)

## Status do texto

`curado`

## Pergunta científica

Por que foi necessário sair da modelagem point-wise simples e adotar uma
arquitetura sequencial residual?

## Síntese

O deslocamento para `seq_bigru_residual` ocorreu quando a linha percebeu que:

- o canal não estava sendo descrito adequadamente por uma lei puramente local
- parte da estrutura relevante exigia contexto temporal curto
- a representação residual organizava melhor o problema do que a saída direta

Essa mudança foi importante porque transformou o problema de “mais capacidade”
em problema de “qual estrutura de dependência o canal exige”.

## O que essa fase resolveu

- tirou o projeto de uma leitura excessivamente estática
- elevou a família sequencial a baseline forte
- preparou o terreno para a etapa de recuperação via `MDN`

## O que essa fase não resolveu

- não eliminou o gargalo de shape nos regimes mais difíceis
- não decidiu, sozinha, entre lei local e lei sequencial como explicação final

## Evidências

- A importância dessa transição está sintetizada em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md),
  especialmente na passagem “Traditional cVAE” para “Sequential Residual cVAE”.

## Implicações

Na tese, esta fase deve aparecer como mudança de hipótese sobre a estrutura do
canal, não apenas como mudança de arquitetura.
