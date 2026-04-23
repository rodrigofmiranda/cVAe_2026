# Validação Principal G1-G5

## Propósito

Definir a validação principal do gêmeo digital da forma que hoje está melhor
fundamentada no projeto.

## Escopo

Este documento trata apenas da aceitação principal do twin, sem incluir o
screen estatístico `G6`.

## Fontes canônicas usadas

- [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)
- [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
- [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md)
- [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md)

## Status do texto

`curado`

## Pergunta científica

O que a tese deve considerar como evidência principal de que um modelo é um
gêmeo digital útil?

## Síntese

A formulação mais sólida hoje é:

- `validation_status_twin = G1..G5`

Isso significa que a aceitação principal deve refletir:

- erro relativo de EVM
- erro relativo de SNR
- consistência de escala e covariância residual
- fidelidade espectral
- shape residual e gaussianidade relativa

## Por que essa escada é forte

- cobre erro médio e estrutura residual
- é multi-regime
- tem interpretação de engenharia mais direta do que um único NHST
- foi auditada e calibrada com referências históricas do projeto

## Evidências

- A recomendação de usar `G1..G5` como núcleo do twin está em
  [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)
  e em
  [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md).

## Implicações

Esta página deve servir como base direta do capítulo metodológico da tese.
Sempre que a tese falar em “validação do twin”, a leitura padrão deve ser
`G1..G5`, salvo menção explícita em contrário.
