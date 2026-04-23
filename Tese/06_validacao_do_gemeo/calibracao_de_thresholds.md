# Calibração de Thresholds

## Propósito

Sintetizar a calibração empírica dos thresholds de validação, transformando-a
em texto de tese.

## Escopo

Este arquivo resume a calibração retrospectiva de 2026-04-11 e o papel dela na
metodologia final.

## Fontes canônicas usadas

- [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md)
- [tabela de calibração](/home/rodrigo/cVAe_2026_shape/outputs/analysis/gate_threshold_calibration_20260411/gate_threshold_calibration_table.csv)
- [relatório de calibração](/home/rodrigo/cVAe_2026_shape/outputs/analysis/gate_threshold_calibration_20260411/gate_threshold_calibration_report.md)

## Status do texto

`curado`

## Pergunta científica

Os thresholds antigos refletiam a fronteira empírica dos melhores twins do
projeto?

## Síntese

A resposta foi não. A calibração mostrou que vários thresholds usados
operacionalmente estavam mais frouxos do que os melhores twins historicamente
atingiam.

## Leitura principal

- `G1`, `G2`, `G4` e parte de `G5` precisaram de aperto relevante
- `JBrel = 0.20` já estava próximo do papel desejado
- `G6` não precisava de novo alpha; precisava de nova interpretação

## Thresholds recomendados

| Métrica | Threshold proposto |
| --- | ---: |
| `cvae_rel_evm_error` | `0.04` |
| `cvae_rel_snr_error` | `0.03` |
| `cvae_mean_rel_sigma` | `0.05` |
| `cvae_cov_rel_var` | `0.15` |
| `cvae_psd_l2` | `0.18` |
| `cvae_delta_skew_l2` | `0.12` |
| `cvae_delta_kurt_l2` | `0.17` |
| `delta_jb_stat_rel` | `0.20` |

## Evidências

- Os artefatos formais da calibração estão em
  [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md)
  e nos arquivos de análise associados.

## Implicações

Na tese, essa calibração deve aparecer como passo de maturação metodológica:
o projeto deixou de usar gates apenas por conveniência histórica e passou a
ancorá-los na fronteira empírica dos melhores resultados.
