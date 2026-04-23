# Status e Limitações

## Propósito

Registrar o estado atual da linha `full_circle` com uma redação que já possa
ser reaproveitada na tese.

## Escopo

Este texto resume:

- o que já pode ser afirmado
- o que ainda não pode ser afirmado
- quais limites metodológicos permanecem

## Fontes canônicas usadas

- [FULL_CIRCLE_CLEAN_RUN_CHECKLIST](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)
- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)

## Status do texto

`curado`

## Síntese

### O que já pode ser afirmado

- `full_circle` foi executado como trilha real e separada.
- geometry-biased e clean foram comparados de maneira suficientemente clara.
- uma classe intermediária `soft-radial` foi identificada sem recorrer a
  `geom3` nem a `disk_l2`.
- o melhor candidato dessa classe até agora,
  `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0`, atingiu `6/12`.
- `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_bs8192` empatou esse
  resultado em `6/12`, sem superá-lo.
- a baseline clean ainda não alcançou o mesmo nível dos melhores runs
  geometry-biased.

### O que ainda não pode ser afirmado

- que a geometria circular real resolve o gargalo residual
- que `full_circle` já superou cientificamente a melhor referência
  `full_square`
- que a melhor solução final de tese deve ser geometry-biased
- que a linha geometry-light já está completamente exaurida, pois ainda falta
  decidir se vale ou não um retune extremamente local em torno do melhor ponto

### Limitações principais

- a baseline clean ainda está abaixo do desejado
- blocos multi-candidato em protocolo compartilhado podem deixar candidatos
  promissores sem resposta direta, exigindo reruns com `--grid_tag`
- `tail98` e `covsoft` se mostraram más continuações locais, o que estreita o
  espaço plausível de busca geometry-light
- resultados de `G6` devem continuar sendo interpretados como auxiliares
- a história numérica fina da validação externa `16QAM` ainda precisa ser
  consolidada nesta camada

## Evidências

- A própria checklist ativa recomenda separar `G1..G5` de `G6` e não
  superinterpretar o screen estatístico, em
  [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md).
- A execução recente do follow-up soft-radial e dos reruns diretos de
  confirmação está registrada em
  [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md).

## Implicações

Na tese, a linha `full_circle` deve entrar como resultado importante e honesto:
ela reforça a hipótese geométrica como trilha relevante, mostra uma recuperação
intermediária com prior radial suave, mas ainda não fecha a questão em favor da
baseline clean.
