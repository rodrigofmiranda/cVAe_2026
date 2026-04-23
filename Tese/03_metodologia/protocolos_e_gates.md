# Protocolos e Gates

## Propósito

Consolidar a semântica atual do protocolo de avaliação e o papel de cada gate
na leitura dos resultados.

## Escopo

Este documento resume:

- a função dos gates `G1..G6`
- a separação entre validação principal e tela estatística
- o estado atual de calibração dos thresholds

## Fontes canônicas usadas

- [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md)
- [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
- [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)

## Status do texto

`curado`

## Pergunta científica

Como interpretar, de forma consistente, os indicadores de aceitação do twin?

## Síntese

O protocolo atual distingue três níveis de decisão:

1. `validation_status_twin`
   - usa `G1..G5`
2. `stat_screen_pass`
   - usa `G6`
3. `validation_status_full`
   - combinação conservadora de `G1..G6`

Essa separação é central para a tese e não deve ser colapsada.

## Leitura dos gates

| Gate | Papel resumido | Métricas centrais |
| --- | --- | --- |
| `G1` | erro relativo de EVM | `cvae_rel_evm_error` |
| `G2` | erro relativo de SNR | `cvae_rel_snr_error` |
| `G3` | consistência de escala residual e covariância | `cvae_mean_rel_sigma`, `cvae_cov_rel_var` |
| `G4` | fidelidade espectral | `cvae_psd_l2` |
| `G5` | shape residual e gaussianidade relativa | `delta_skew_l2`, `delta_kurt_l2`, `delta_jb_stat_rel` |
| `G6` | ausência de mismatch detectado no budget estatístico configurado | `stat_mmd_qval`, `stat_energy_qval` |

## Calibração atual

Após a calibração empírica de 2026-04-11, a leitura recomendada é:

- apertar `G1..G5` para refletir melhor a fronteira empírica dos bons twins
- manter `G6` em `q > 0.05`, mas como nível estatístico convencional

### Thresholds propostos para a validação principal

| Gate | Métrica | Threshold proposto |
| --- | --- | ---: |
| `G1` | `cvae_rel_evm_error` | `0.04` |
| `G2` | `cvae_rel_snr_error` | `0.03` |
| `G3` | `cvae_mean_rel_sigma` | `0.05` |
| `G3` | `cvae_cov_rel_var` | `0.15` |
| `G4` | `cvae_psd_l2` | `0.18` |
| `G5` | `cvae_delta_skew_l2` | `0.12` |
| `G5` | `cvae_delta_kurt_l2` | `0.17` |
| `G5` | `delta_jb_stat_rel` | `0.20` |

## Leituras que a tese deve evitar

- `G6` como prova de equivalência
- `G6` como prova de indistinguibilidade sem qualificação
- comparação de dois `G6` sem citar budget estatístico

## Evidências

- A separação metodológica aparece literalmente em
  [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md)
  e
  [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md).
- A auditoria crítica do significado de `G6` está em
  [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md).

## Implicações

O capítulo metodológico deve assumir explicitamente que o twin é aceito
principalmente por `G1..G5`, enquanto `G6` fornece uma leitura estatística
auxiliar e conservadora.
