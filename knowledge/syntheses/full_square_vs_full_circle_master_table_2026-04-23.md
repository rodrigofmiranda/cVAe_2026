# Full Square vs Full Circle Master Table (2026-04-23)

Esta nota consolida a comparação crossline mais útil no estado atual do
projeto, separando claramente:

- protocolo principal do gêmeo digital (`G1..G5` + leitura auxiliar `G6`)
- validação externa em `16QAM`
- situação operacional do `legacy2025_large` após o bug de loader

O objetivo é produzir uma leitura estável que possa ser citada tanto em docs
operacionais quanto na camada `Tese`, sem exigir releitura integral do histórico
recente.

## Fontes canônicas usadas

- [best_compare_large protocol leaderboard](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/comparative/best_compare_large/full_data_sel4_overnight_20260423_040529/exp_20260423_040722/tables/protocol_leaderboard.csv)
- [best_compare_large summary_by_regime](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/comparative/best_compare_large/full_data_sel4_overnight_20260423_040529/exp_20260423_040722/tables/summary_by_regime.csv)
- [best_compare_large 16QAM manifest](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/comparative/best_compare_large/full_data_sel4_overnight_20260423_040529/benchmarks/16qam/eval_16qam_sel4_stats_20260423_040529/manifest_all_regimes_eval.csv)
- [16QAM updated crossline summary](/home/rodrigo/cVAe_2026_full_square/outputs/analysis/eval_16qam_crossline_20260423_best_compare_vs_fullcircle/README.md)
- [16QAM updated crossline summary with legacy clean rerun](/home/rodrigo/cVAe_2026_full_square/outputs/analysis/eval_16qam_crossline_20260423_with_legacy_clean/README.md)
- [full circle soft-radial master table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/full_circle_soft_radial_master_table_2026-04-22.md)
- [full circle PROJECT_STATUS 16QAM section](/home/rodrigo/cVAe_2026_shape_fullcircle/PROJECT_STATUS.md)
- [legacy2025 original protocol leaderboard with failed loader path](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_overnight_20260423_035804/exp_20260423_040105/tables/protocol_leaderboard.csv)
- [legacy2025 clean rerun launch README](/home/rodrigo/cVAe_2026_full_square/outputs/_launch_logs/legacy2025_clean_rerun_20260423_114751/README.txt)
- [legacy2025 clean rerun protocol leaderboard](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751/exp_20260423_114759/tables/protocol_leaderboard.csv)
- [legacy2025 clean rerun 16QAM manifest](/home/rodrigo/cVAe_2026_full_square/outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751/benchmarks/16qam/eval_16qam_sel4_stats_20260423_114751/manifest_all_regimes_eval.csv)

## Reading Rule

- `Protocol` = validação principal do twin, ainda centrada em `G1..G5`
- `16QAM` = validação externa de generalização, não substitui o protocolo
- `Baseline` = referência científica honesta de uma linha
- `Geometry-light` = compromisso metodológico entre baseline limpa e viés geométrico forte
- `Ceiling` = melhor teto operacional, não baseline neutra

## Master Table

| Linha | Candidato / papel | Status | Protocolo principal | 16QAM externo | Leitura estabilizada |
| --- | --- | --- | --- | --- | --- |
| `full_square` | `best_compare_large` champion `S2seq_W7_h64_lat4_b0p003_lmmd0p5_fb0p10_lr0p0003_L128-256-512` | resolvido | `5/12` pass, `7/12` fail; `G1=8`, `G2=9`, `G3=8`, `G4=12`, `G5=5`, `G6=7` | `12/12` regimes completos; melhor média global entre os quatro braços comparados | hoje é o melhor comparador crossline do lado `full_square`: não é o melhor teto histórico interno da linha, mas é fortíssimo na validação externa |
| `full_circle` | `clean baseline` | resolvido | `5/12` pass | `12/12`; pior que `best_compare_large` nas médias globais, mas ainda vence `3/12` regimes em `EVM/SNR/PSD` na comparação histórica | continua sendo a baseline científica honesta de `full_circle`, sem força suficiente para deslocar `full_square` |
| `full_circle` | `soft_rinf_local` geometry-light | resolvido | `6/12` pass | `12/12`; melhora a baseline clean dentro da família `full_circle`, mas segue atrás de `best_compare_large` na leitura global | melhor compromisso científico atual dentro de `full_circle`; referência geometry-light legítima |
| `full_circle` | `disk_geom3` ceiling | resolvido | `8/12` pass | `12/12`; não vence em `EVM/SNR/PSD` contra `best_compare_large`, embora retenha algumas vitórias em métricas de informação | deve ser mantido como teto operacional, não como baseline neutra |
| `full_square` | `legacy2025_large` | resolvido | rerun limpo confirmou `1/12` pass, `11/12` fail; `G1=7`, `G2=5`, `G3=3`, `G4=10`, `G5=4`, `G6=3` | `12/12` regimes completos; leitura externa média inferior aos braços canônicos de `full_square` e também inferior aos principais braços `full_circle` | o artefato antigo continua inválido por bug, mas o rerun limpo fecha o diagnóstico: a família voltou a ser comparável e, neste estado, é cientificamente fraca |

## Resultado Principal Da Comparação Crossline

O quadro atual mostra uma tensão metodologicamente importante:

- no protocolo interno do twin, o melhor representante canônico de
  `full_circle` (`soft_rinf_local`) ainda é melhor que o `best_compare_large`
  em contagem bruta de regimes aprovados (`6/12` vs `5/12`)
- no entanto, na validação externa em `16QAM`, o `best_compare_large` domina os
  representantes canônicos de `full_circle`

Essa tensão não é ruído. Ela sugere que, no estado atual:

1. `full_circle` ainda preserva valor científico como linha de investigação
   interna do twin
2. `full_square` segue sendo a linha mais robusta quando o critério é
   generalização externa sob um alfabeto discreto tradicional
3. a decisão final da tese não deve ser feita olhando apenas para um dos dois
   eixos

## Leitura 16QAM Atualizada

A comparação agregada mais recente entre:

- `full_square_best_compare_large`
- `full_circle_clean`
- `full_circle_disk`
- `full_circle_soft_rinf_local`

produziu o seguinte resumo médio:

| candidato | regimes | `|ΔEVM|` mean | `|ΔSNR|` mean | `ΔPSD` mean | `MI_pred` mean | `GMI_pred` mean | `NGMI_pred` mean | `AIR_pred` mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_square_best_compare_large` | `12` | `0.725` | `0.258` | `0.060` | `2.270` | `2.195` | `0.549` | `2.270` |
| `full_circle_clean` | `12` | `4.037` | `0.843` | `0.103` | `2.083` | `2.000` | `0.500` | `2.083` |
| `full_circle_disk` | `12` | `3.814` | `0.923` | `0.107` | `2.104` | `2.033` | `0.508` | `2.104` |
| `full_circle_soft_rinf_local` | `12` | `3.956` | `0.871` | `0.103` | `2.099` | `2.015` | `0.504` | `2.099` |

Por contagem de vitórias regime a regime:

- `full_square_best_compare_large` venceu `9/12` regimes em `|ΔEVM|`, `|ΔSNR|`
  e `ΔPSD`
- ele também venceu `8/12` regimes em `MI`, `GMI`, `NGMI` e `AIR`
- nenhum dos três braços `full_circle` venceu o `best_compare_large` na leitura
  média global

Portanto, a leitura externa estabilizada continua sendo:

- `full_square` é hoje o melhor balizador externo global
- `soft_rinf_local` é o melhor compromisso dentro de `full_circle`
- `disk_geom3` continua como teto operacional do `full_circle`, mas isso não se
  converte em liderança externa global

## Situação Do legacy2025_large

O resultado original do overnight `legacy2025_large` não deve ser tratado como
negativo científico da família. O que falhou foi a reabertura do artefato salvo
na etapa de avaliação. Esse problema foi fechado no rerun limpo.

Diagnóstico fechado:

- `tf.keras.load_model(...)` falhava no artefato `.keras` com a mensagem
  `legacy_core_dense_in expected 2 variables, received 0`
- o modelo pôde ser recuperado ao reconstruir a arquitetura campeã a partir de
  `gridsearch_results.csv` e aplicar `model.load_weights(...)`
- após esse patch, um smoke com `--reuse_model_run_dir` voltou a produzir
  artefatos de avaliação reais para o `legacy_2025`
- foi necessário manter `TF_ENABLE_ONEDNN_OPTS=0`, porque o `LayerNormalization`
  da topologia legacy acionava `_MklLayerNorm` incompatível com GPU quando esse
  toggle não estava desligado

Resultado final do rerun limpo:

- protocolo principal: `1/12` pass, `11/12` fail
- leitura completa com `G6`: `1/12` pass, `stat_screen=3/12`
- campeão confirmado: `L3legacy_lat8_b0p1_fb0p0_lr0p0001_bs8192_anneal50_L64-128-256-512`
- benchmark externo `16QAM`: `ok=12/12`, mas com médias globais fracas
  (`|ΔEVM|=5.426`, `|ΔSNR|=1.300`, `ΔPSD=0.161`, `MI_pred=2.015`,
  `GMI_pred=1.980`, `NGMI_pred=0.495`, `AIR_pred=2.015`)

Decisão operacional:

- o diretório antigo de avaliação `16QAM` foi reclassificado com o sufixo
  `_failed_loader_prepatch`
- o rerun canônico foi concluído em uma nova base limpa e substitui
  definitivamente a leitura anterior
- a família `legacy2025` volta a ser utilizável como referência histórica da
  linha `full_square`, mas não como candidata competitiva atual

## Decisão Provisória Atual

1. usar `best_compare_large` como comparador `full_square` mais útil no estado
   atual da comparação crossline
2. manter `soft_rinf_local` como referência geometry-light do `full_circle`
3. manter `disk_geom3` como teto operacional, sem vendê-lo como baseline
4. manter `legacy2025_large` original fora da evidência científica e usar apenas
   o rerun limpo como leitura válida da família
5. tratar a comparação `protocolo interno` vs `16QAM externo` como um eixo de
   discussão central da tese, não como contradição a ser escondida
