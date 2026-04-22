# Stat-Screen G6

## Propósito

Registrar o papel correto de `G6` na metodologia atual.

## Escopo

Este documento trata de `G6` como tela estatística auxiliar e explicita o que
ele significa e o que ele não significa.

## Fontes canônicas usadas

- [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
- [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md)
- [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)

## Status do texto

`curado`

## Pergunta científica

Como a tese deve reportar `G6` sem superestimar seu significado?

## Síntese

`G6` deve ser descrito como:

- tela estatística auxiliar
- teste de ausência de mismatch detectado sob o budget configurado
- informação complementar à validação principal

`G6` não deve ser descrito como:

- prova de equivalência do canal
- prova de indistinguibilidade absoluta
- critério único de aceitação do twin

## Semântica operacional

Hoje, `G6` depende de:

- `stat_mmd_qval > 0.05`
- `stat_energy_qval > 0.05`

Mas esse `0.05` deve ser lido como nível estatístico convencional, não como
threshold de engenharia calibrado.

## Evidências

- A auditoria de significado de `G6` está em
  [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md).
- A separação explícita entre `validation_status_twin` e `stat_screen_pass`
  aparece no
  [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md)
  e na
  [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md).

## Implicações

Na tese, `G6` pode enriquecer a discussão e ajudar na comparação de candidatos,
mas nunca deve substituir a validação principal.
