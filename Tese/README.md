# Tese — Camada Centralizada da Linha VLC cVAE

## Propósito

Centralizar, em formato editorial e científico, a história experimental da
linha VLC cVAE sem substituir as fontes operacionais do projeto.

## Escopo

Esta pasta organiza:

- o problema científico
- a fundamentação
- a metodologia
- as linhas `full_square` e `full_circle`
- a validação do gêmeo digital
- as decisões tomadas com base em dados

Ela não substitui `docs/`, `knowledge/` nem `outputs/`. Ela funciona como
camada curada que sintetiza, interpreta e aponta para as fontes canônicas.

## Fontes canônicas usadas

- [README do worktree mdn_return](/home/rodrigo/cVAe_2026_mdn_return/README.md)
- [PROJECT_STATUS](/home/rodrigo/cVAe_2026_mdn_return/PROJECT_STATUS.md)
- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [FULL_CIRCLE_NEXT_STEP](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_NEXT_STEP.md)
- [FULL_CIRCLE_CLEAN_RUN_CHECKLIST](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md)
- [WORKING_STATE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/WORKING_STATE.md)
- [support_ablation_e0_e3_comparison](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/support_ablation_e0_e3_comparison_2026-04-07.md)
- [support_ablation_e3b_e4_followup](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/support_ablation_e3b_e4_followup_2026-04-07.md)
- [support_scientific_screen_master_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/support_scientific_screen_master_table_2026-04-10.md)
- [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
- [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md)
- [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)
- [vlc_probabilistic_shaping_strategy](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/vlc_probabilistic_shaping_strategy_2026-04-03.md)
- [vlc_shaping_experimental_methodology](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/vlc_shaping_experimental_methodology_2026-04-03.md)
- [askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance](/home/rodrigo/cVAe_2026_shape/knowledge/notes/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance.md)
- [shu_2020_probabilistic_shaping_optical](/home/rodrigo/cVAe_2026_shape/knowledge/notes/shu_2020_probabilistic_shaping_optical.md)
- [Askari e Lampe 2025 parsed](/home/rodrigo/cVAe_2026_shape/knowledge/papers/parsed/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance/document.md)
- [Shu e Zhang 2020 parsed](/home/rodrigo/cVAe_2026_shape/knowledge/papers/parsed/shu_2020_probabilistic_shaping_optical/document.md)
- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)

## Status do texto

`curado`

## Como ler esta pasta

Ordem sugerida para um novo colaborador:

1. `00_metadados/escopo_e_convencoes.md`
2. `01_introducao_e_problema/`
3. `03_metodologia/`
4. `04_linha_full_square/`
5. `05_linha_full_circle/`
6. `06_validacao_do_gemeo/`
7. `07_resultados_e_decisoes/`
8. `09_anexos/`

## Convenções desta camada

- Cada arquivo da `Tese/` é uma síntese curada, não um log operacional.
- Toda afirmação importante deve apontar para uma fonte canônica.
- Resultados numéricos entram como leitura estabilizada e sempre com rastreio.
- `G1..G5` definem a validação principal do gêmeo digital.
- `G6` aparece sempre como `stat_screen`, separado da validação principal.
- Métricas como `MI`, `GMI`, `NGMI` e `AIR` são tratadas como auxiliares.

Quando houver base técnica disponível em `knowledge/notes` ou
`knowledge/papers/parsed`, a preferência editorial é citar também essa camada,
e não apenas as sínteses internas.

## Classes de documento

| Classe | Papel | Onde vive |
| --- | --- | --- |
| Texto de tese | síntese científica, narrativa e decisão | `Tese/` |
| Fonte canônica | histórico operacional e notas ativas | `docs/`, `knowledge/` |
| Evidência | runs, tabelas, plots e artefatos de protocolo | `outputs/` |

## Estrutura

- [00_metadados](00_metadados/escopo_e_convencoes.md): convenções, glossário e mapa de worktrees.
- [01_introducao_e_problema](01_introducao_e_problema/problema_de_pesquisa.md): problema, objetivos e hipóteses.
- [02_fundamentacao](02_fundamentacao/fundamentos_vlc_imdd.md): base conceitual de VLC IM/DD, digital twin e geometria de suporte.
- [03_metodologia](03_metodologia/pipeline_experimental.md): pipeline, protocolos, datasets e famílias de modelo.
- [04_linha_full_square](04_linha_full_square/visao_geral.md): evolução histórica da linha principal no `full_square`.
- [05_linha_full_circle](05_linha_full_circle/visao_geral.md): execução e leitura científica da linha `full_circle`.
- [06_validacao_do_gemeo](06_validacao_do_gemeo/validacao_principal_g1_g5.md): metodologia validada do twin.
- [07_resultados_e_decisoes](07_resultados_e_decisoes/resultados_por_linha.md): síntese dos resultados e da matriz de decisão.
- [08_planejamento_da_redacao](08_planejamento_da_redacao/esqueleto_capitulos.md): preparação para a escrita da tese.
- [09_anexos](09_anexos/catalogo_de_runs.md): catálogos de runs, documentos e figuras.

## Síntese

A função desta pasta é reduzir a distância entre o laboratório e a tese. O
objetivo não é repetir tudo o que existe; o objetivo é deixar explícito:

- o que foi testado
- por que foi testado
- o que os dados sustentam
- o que foi descartado
- o que permanece em aberto

## Evidências

- O histórico `full_square` já está condensado em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).
- O `full_circle` real foi tratado em worktree separado e consolidado em
  [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)
  e
  [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md).
- A metodologia de validação foi auditada em
  [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
  e calibrada em
  [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md).

## Implicações

Com esta camada, a redação futura da tese passa a ter um ponto de entrada
único, em português técnico, com distinção explícita entre:

- baseline científica
- linha operacional
- evidência auxiliar
