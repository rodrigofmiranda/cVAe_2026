# Pointwise Revival

## Propósito

Explicar por que a retomada controlada de famílias point-wise ainda foi
cientificamente importante mesmo depois do sucesso da linha sequencial com MDN.

## Escopo

Este documento cobre a lógica do `pointwise revival`, não o histórico completo
de todas as variantes ponto a ponto.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [WORKING_STATE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/WORKING_STATE.md)

## Status do texto

`curado`

## Pergunta científica

Se a linha sequencial com `MDN` era dominante, por que ainda valia testar uma
retomada point-wise?

## Síntese

Valia testar porque o sucesso sequencial não fechava a hipótese de que parte do
canal pudesse ser descrita por uma lei local estreita e bem escolhida.

O `pointwise revival` mostrou um resultado importante:

- uma família point-wise ampla, no estilo antigo, não retornou como solução
- uma família point-wise local e estreita continuou cientificamente viva

O melhor resultado local registrado foi:

- run: `outputs/exp_20260404_155322`
- campeão: `S38bB_pw25_local_lat4_b0p01_fb0p0_lr0p0003_bs2048_anneal3_L32-64`
- resultado: `10/12`

## Leitura correta

Esse bloco não prova que point-wise seja superior ao sequencial. Ele prova algo
mais interessante:

- a representação do gargalo ainda não estava totalmente fechada
- uma lei local continuava plausível
- a tese não pode tratar a linha sequencial como resposta ontologicamente única

## Evidências

- A leitura acima está explicitamente sintetizada em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).

## Implicações

Esse resultado justifica a postura metodológica posterior de não importar para
`full_circle` apenas a baseline mais engenheirada, mas também reabrir espaço
para baselines simples e locais.
