# Baseline Clean vs Geometry-Biased

## Propósito

Fixar a separação conceitual e metodológica entre os runs `full_circle`
geometry-biased e os runs `full_circle` clean.

## Escopo

Este documento traduz em linguagem de tese a decisão central tomada nessa
linha: melhores números operacionais não substituem baseline limpa.

## Fontes canônicas usadas

- [FULL_CIRCLE_NEXT_STEP](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_NEXT_STEP.md)
- [FULL_CIRCLE_CLEAN_RUN_CHECKLIST](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md)
- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)

## Status do texto

`curado`

## Pergunta científica

Por que a tese precisa manter duas leituras distintas dentro da mesma linha
`full_circle`?

## Síntese

A separação é necessária porque os resultados `full_circle` passaram a ocupar
três classes distintas, cada uma respondendo uma pergunta diferente.

| Classe | Pergunta que responde | Leitura correta |
| --- | --- | --- |
| geometry-biased | “intervenções geométricas fortes ajudam operacionalmente?” | útil para exploração e engenharia |
| soft-radial | “um prior radial suave recupera desempenho sem voltar a `geom3` e `disk_l2`?” | evidência intermediária, geometry-light |
| clean | “a nova geometria real melhora o problema sem ajuda extra?” | baseline científica da hipótese geométrica |

## Resultado prático atual

- geometry-biased atingiu até `8/12`
- soft-radial atingiu `6/12` com
  `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0`
- clean ficou na faixa `1/12` a `5/12`

Esses números não autorizam dizer que `full_circle` resolveu o problema. Eles
autorizam dizer que:

- o viés geométrico forte ajuda operacionalmente
- um prior radial suave pode recuperar parte do desempenho perdido sem recorrer
  a `geom3` ou `disk_l2`
- a hipótese geométrica limpa ainda não foi confirmada no mesmo nível

## Evidências

- A necessidade dessa separação já aparece antes da execução em
  [FULL_CIRCLE_NEXT_STEP](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_NEXT_STEP.md).
- A confirmação prática da fraqueza relativa da baseline clean aparece em
  [FULL_CIRCLE_CLEAN_RUN_CHECKLIST](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md).
- A nova evidência intermediária de prior radial suave aparece em
  [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md).

## Implicações

Esta é uma peça crítica para a honestidade metodológica da tese. Sem ela,
ficaria fácil misturar:

- teto operacional sob viés geométrico forte
- recuperação parcial com prior radial suave
- baseline científica limpa

Essas três leituras precisam permanecer separadas na redação final.
