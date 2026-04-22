# Execução Real Full-Circle

## Propósito

Registrar a cronologia mínima dos runs `full_circle` que já entraram na camada
de evidência científica.

## Escopo

Este documento não substitui os diretórios de output nem o plano de execução.
Ele resume quais blocos foram executados e qual leitura estabilizada eles
permitem.

## Fontes canônicas usadas

- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)
- [FULL_CIRCLE_CLEAN_RUN_CHECKLIST](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md)
- [leaderboard e2 shortlist](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/e2_finalists_shortlist_100k/exp_20260416_014727/protocol_leaderboard.csv)
- [leaderboard g2 shortlist](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/g2_shortlist_100k/exp_20260416_120619/protocol_leaderboard.csv)

## Status do texto

`curado`

## Pergunta científica

Quais execuções `full_circle` realmente ocorreram e o que elas mostram quando
separa-se exploração geometry-biased de baseline clean?

## Síntese

Até o momento, a execução real `full_circle` já produziu quatro blocos
interpretáveis e um bloco de resolução já encerrado:

1. shortlist inicial herdada da referência `E2`
2. shortlist geometry-biased com viés de disco
3. reruns clean em `full_circle` real
4. follow-up soft-radial sem `geom3` e sem `disk_l2`
5. reruns diretos para fechar os candidatos não resolvidos do follow-up

## Blocos executados e leitura atual

| Bloco | Artefato canônico | Melhor assinatura | Leitura resumida |
| --- | --- | --- | --- |
| shortlist `E2` | `e2_finalists_shortlist_100k/exp_20260416_014727` | `S27cov_sciv1_lr0p00015` | referência inicial de transferência, `4/12` |
| shortlist `g2` | `g2_shortlist_100k/exp_20260416_120619` | `...disk_geom3` | geometry-biased melhora shortlist, `7/12` |
| geometry-biased split `A` | `disk_bs8192_lat10_100k_split_a/exp_20260416_165643` | `...disk_geom3_bs8192` | melhor resultado operacional observado, `8/12` |
| geometry-biased split `B` | `disk_bs8192_lat10_100k_split_b/exp_20260416_165644` | `...disk_geom3_lat10` | resultado fraco e instável, `3/12` |
| clean split `A` primeira rodada | `20260416_182317_clean...split_a/exp_20260416_182319` | `...fc_clean_lc0p25_t0p03` | baseline clean ainda fraca, `1/12` |
| clean split `B` primeira rodada | `20260416_182317_clean...split_b/exp_20260416_182319` | `...fc_clean_lc0p25_t0p03_lat10` | melhora parcial, `4/12` |
| clean split `A` segunda rodada | `20260417_115140_clean...split_a/exp_20260417_115142` | `...fc_clean_lc0p25_t0p03_bs8192` | leve recuperação, `2/12` |
| clean split `B` segunda rodada | `20260417_115140_clean...split_b/exp_20260417_115142` | `...fc_clean_lc0p25_t0p03_lat10` | melhor clean observado até aqui, `5/12` |
| soft-radial bloco `A` | `20260420_233254_soft_radial_block_a_100k/exp_20260420_233256` | `...soft_rinf_local...` | melhor geometry-light observado, `6/12` |
| soft-radial bloco `B` | `20260420_233254_soft_radial_block_b_100k/exp_20260420_233257` | `...covsoft...` | campeão interno do bloco falhou no protocolo, `0/12`; `bs8192` e `tail98` ficaram sem resposta direta |
| rerun direto `soft_rinf_local_bs8192` | `20260421_234722_soft_radial_resolve_bs8192_100k/exp_20260421_234723` | `...soft_rinf_local_bs8192` | empate com o melhor geometry-light, `6/12` |
| rerun direto `soft_rinf_local_tail98` | `20260421_234722_soft_radial_resolve_tail98_100k/exp_20260421_234724` | `...soft_rinf_local_tail98` | continuação negativa, `1/12` |

## Leitura estabilizada

- a trilha geometry-biased foi operacionalmente mais forte
- a trilha soft-radial abriu uma classe intermediária promissora
- a trilha clean permaneceu claramente mais fraca
- `soft_rinf_local` e `soft_rinf_local_bs8192` formam o melhor patamar
  geometry-light observado até aqui, ambos com `6/12`
- `tail98` e `covsoft` ficaram descartados como continuações locais
- isso reforça a importância de não usar o melhor número geometry-biased como
  baseline científica final
- também mostra que blocos multi-candidato precisam de reruns diretos quando o
  protocolo avalia apenas o campeão interno do bloco

## Evidências

- Os runs clean e geometry-biased estão catalogados no diretório
  `/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle`.
- Os follow-ups soft-radial também estão catalogados nesse diretório, com os
  prefixos `20260420_233254_soft_radial_*` e `20260421_234722_soft_radial_*`.
- A interpretação metodológica de cada bloco está em
  [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)
  e
  [FULL_CIRCLE_CLEAN_RUN_CHECKLIST](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md).

## Implicações

Para a tese, o ponto principal não é “qual run ganhou”. O ponto principal é que
os resultados atuais distinguem claramente:

- ganho operacional sob viés geométrico forte
- recuperação parcial com prior radial suave
- baseline científica limpa
