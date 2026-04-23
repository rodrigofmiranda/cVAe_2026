# Comparação Crossline Full Square vs Full Circle (2026-04-23)

## Propósito

Registrar, em formato de tese, a leitura consolidada mais recente da comparação
entre `full_square` e `full_circle`, incorporando também o rerun limpo da
família `legacy2025` como referência histórica do lado `full_square`.

## Escopo

Este documento não substitui a seção geral de validação externa em `16QAM`.
Ele registra uma rodada específica de consolidação, já estabilizada, útil para:

- comparação entre linhas científicas concorrentes
- distinção entre protocolo principal e validação externa
- posicionamento final do `legacy2025` após correção do bug de loader

## Fontes canônicas usadas

- [full_square vs full_circle master table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/full_square_vs_full_circle_master_table_2026-04-23.md)
- [16QAM crossline with legacy clean](/home/rodrigo/cVAe_2026_full_square/outputs/analysis/eval_16qam_crossline_20260423_with_legacy_clean/README.md)
- [best_compare_large protocol leaderboard](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/comparative/best_compare_large/full_data_sel4_overnight_20260423_040529/exp_20260423_040722/tables/protocol_leaderboard.csv)
- [best_compare_large 16QAM manifest](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/comparative/best_compare_large/full_data_sel4_overnight_20260423_040529/benchmarks/16qam/eval_16qam_sel4_stats_20260423_040529/manifest_all_regimes_eval.csv)
- [legacy2025 clean rerun protocol leaderboard](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751/exp_20260423_114759/tables/protocol_leaderboard.csv)
- [legacy2025 clean rerun 16QAM manifest](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751/benchmarks/16qam/eval_16qam_sel4_stats_20260423_114751/manifest_all_regimes_eval.csv)
- [full_circle soft-radial master table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/full_circle_soft_radial_master_table_2026-04-22.md)
- [full_circle 16QAM crossline summary](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260422_plus_soft_radial/crossline_summary/README.md)

## Status do texto

`curado`

## Pergunta científica

Quando `full_square` e `full_circle` são comparados de forma honesta, com o
protocolo principal do twin separado da validação externa em `16QAM`, qual é a
leitura estabilizada que a tese pode sustentar?

## Síntese

A leitura consolidada desta rodada é a seguinte:

- `full_circle` ainda preserva vantagem no protocolo interno quando a comparação
  é feita contra o melhor comparador canônico atual de `full_square`
- `full_square`, por outro lado, segue sendo claramente superior na validação
  externa em `16QAM`
- `disk_geom3` continua sendo teto operacional do `full_circle`, mas não deve
  ser apresentado como baseline científica neutra
- `soft_rinf_local` é hoje o melhor compromisso metodológico dentro de
  `full_circle`
- `legacy2025`, depois da correção do loader, voltou a ser comparável de forma
  justa, mas seu desempenho real ficou fraco tanto no protocolo quanto em
  `16QAM`

## Quadro estabilizado da comparação

| linha | candidato | papel | protocolo principal | validação externa `16QAM` | leitura científica |
| --- | --- | --- | --- | --- | --- |
| `full_square` | `best_compare_large` | melhor comparador crossline atual da linha | `5/12` pass, `7/12` fail | melhor média global entre todos os braços comparados | hoje é o melhor árbitro externo do lado `full_square` |
| `full_circle` | `clean baseline` | baseline científica honesta | `5/12` pass | pior que `best_compare_large` nas médias globais | importante como referência limpa, mas ainda sem força de liderança |
| `full_circle` | `soft_rinf_local` | geometry-light | `6/12` pass | melhora a baseline clean, mas segue atrás de `best_compare_large` externamente | melhor compromisso científico atual dentro de `full_circle` |
| `full_circle` | `disk_geom3` | teto operacional | `8/12` pass | não lidera a leitura externa global | manter como teto operacional, não como baseline |
| `full_square` | `legacy2025_large` rerun limpo | referência histórica da linha | `1/12` pass, `11/12` fail | `12/12` completos, porém com médias fracas | válido como leitura histórica, descartável como candidato atual |

## Leitura do protocolo principal

Quando o critério é a validação principal do twin, centrada em `G1..G5`, a
leitura atual não autoriza declarar vitória simples de `full_square` sobre
`full_circle`.

Os pontos centrais são:

- `soft_rinf_local` atingiu `6/12` regimes aprovados
- `best_compare_large` atingiu `5/12`
- `disk_geom3` chegou a `8/12`, mas sob uma formulação operacional com viés
  geométrico explícito
- `legacy2025` limpo ficou em `1/12`, mostrando que a família é hoje apenas
  referência histórica

Portanto, dentro do protocolo principal, `full_circle` ainda conserva interesse
científico real. A linha não pode ser descartada com base apenas na rodada mais
recente do `full_square`.

## Leitura da validação externa em 16QAM

Na validação externa, a hierarquia fica bem mais clara.

Resumo médio da comparação mais recente:

| candidato | regimes | `|ΔEVM|` mean | `|ΔSNR|` mean | `ΔPSD` mean | `MI_pred` mean | `GMI_pred` mean | `NGMI_pred` mean | `AIR_pred` mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_square_best_compare_large` | `12` | `0.725` | `0.258` | `0.060` | `2.270` | `2.195` | `0.549` | `2.270` |
| `full_circle_clean` | `12` | `4.037` | `0.843` | `0.103` | `2.083` | `2.000` | `0.500` | `2.083` |
| `full_circle_disk` | `12` | `3.814` | `0.923` | `0.107` | `2.104` | `2.033` | `0.508` | `2.104` |
| `full_circle_soft_rinf_local` | `12` | `3.956` | `0.871` | `0.103` | `2.099` | `2.015` | `0.504` | `2.099` |
| `full_square_legacy2025_clean` | `12` | `5.426` | `1.300` | `0.161` | `2.015` | `1.980` | `0.495` | `2.015` |

Por contagem de vitórias regime a regime:

- `best_compare_large` venceu `8/12` ou mais regimes em todos os indicadores
  principais da comparação externa
- nenhum braço de `full_circle` venceu o `best_compare_large` na leitura média
  global
- o `legacy2025` limpo não ficou apenas atrás do `best_compare_large`; ele
  também ficou atrás dos principais braços de `full_circle`

A conclusão estável é:

- `full_square` segue sendo o melhor balizador externo de generalização
- `soft_rinf_local` é o melhor representante metodológico do lado
  `full_circle`
- `legacy2025` não deve voltar para a frente competitiva desta linha

## Papel do legacy2025 após a correção do loader

O overnight original do `legacy2025` ficou inutilizável como evidência
científica porque a etapa de avaliação não conseguia reabrir o artefato salvo.
O erro foi fechado com um fallback de reconstrução da arquitetura campeã a
partir do `gridsearch_results.csv` e carregamento via `load_weights`.

Depois da correção, o rerun limpo confirmou duas coisas distintas:

- o problema anterior era de infraestrutura de carregamento, não de ausência de
  modelo
- uma vez que o modelo é avaliado corretamente, ele não se mostra competitivo

Assim, a leitura final da família é metodologicamente importante: ela não deve
ser descartada por bug, mas sim por desempenho efetivo após correção.

## Evidências

- A síntese canônica de comparação entre linhas está em
  [full_square_vs_full_circle_master_table_2026-04-23.md](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/full_square_vs_full_circle_master_table_2026-04-23.md).
- O comparativo externo consolidado com cinco braços está em
  [README.md](/home/rodrigo/cVAe_2026_full_square/outputs/analysis/eval_16qam_crossline_20260423_with_legacy_clean/README.md).
- O rerun limpo do `legacy2025` fechou `1/12` no protocolo em
  [protocol_leaderboard.csv](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751/exp_20260423_114759/tables/protocol_leaderboard.csv).
- A validação externa limpa do `legacy2025` fechou `12/12` regimes em
  [manifest_all_regimes_eval.csv](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751/benchmarks/16qam/eval_16qam_sel4_stats_20260423_114751/manifest_all_regimes_eval.csv).

## Implicações

A tese já pode sustentar, com esta rodada, uma distinção metodológica madura:

- `G1..G5` continuam sendo o critério principal de aceitação do twin
- `16QAM` funciona como árbitro externo entre linhas concorrentes
- a leitura `full_square` versus `full_circle` não é unidimensional
- a discussão final deve separar claramente:
  - liderança no protocolo interno
  - liderança em generalização externa
  - linhas históricas descartadas por desempenho real
