# MDN Recovery

## Propósito

Registrar por que a linha `MDN` foi um marco na interpretação do problema em
`full_square`.

## Escopo

Este documento destaca o que a recuperação via `MDN` resolveu e o que ela
deixou em aberto.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [WORKING_STATE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/WORKING_STATE.md)
- [PROJECT_STATUS](/home/rodrigo/cVAe_2026_mdn_return/PROJECT_STATUS.md)

## Status do texto

`curado`

## Pergunta científica

O decoder `MDN` apenas melhora métricas médias ou ele realmente muda a leitura
do gargalo residual em `full_square`?

## Síntese

A resposta sustentada pelos artefatos do projeto é que o `MDN` mudou a leitura
do problema. Ele não apenas melhorou o score agregado; ele mostrou que a maior
parte da linha já era capaz de reproduzir o canal com alta fidelidade, ficando
o gargalo concentrado em poucos regimes.

O marco mais importante registrado foi:

- run de referência: `outputs/exp_20260328_153611`
- campeão: `S27cov_lc0p25_tail95_t0p03`
- resultado histórico: `10/12`

## Leitura científica do marco

Esse resultado não diz que o problema acabou. Ele diz que:

- o gargalo não é colapso generalizado
- o regime difícil está localizado
- `G5` passou a concentrar a maior parte da falha remanescente

## O que o MDN ajudou a descartar

- a ideia de que bastaria um decoder gaussiano simples melhor ajustado
- a ideia de que o problema restante era puramente de covariância
- a ideia de que mais componentes ou mais perdas antigas resolveriam tudo

## Evidências

- Essa leitura está consolidada na seção “Second Major Shift: MDN Recovery On
  Full-Square” de
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).

## Implicações

Na tese, o `MDN` deve aparecer como o ponto em que o problema deixa de ser
difuso e passa a ser uma falha residual localizada, o que muda o tipo de
hipótese seguinte a ser testada.
