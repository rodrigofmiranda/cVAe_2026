# Digital Twin e Validação

## Propósito

Definir o que significa, nesta tese, chamar um modelo de gêmeo digital válido.

## Escopo

Este documento estabelece o significado metodológico de:

- gêmeo digital
- validação principal
- screen estatístico
- validação externa

## Fontes canônicas usadas

- [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)
- [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
- [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md)
- [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md)

## Status do texto

`curado`

## Pergunta científica

Qual é a definição operacionalmente útil e cientificamente defensável de
gêmeo digital nesta linha?

## Síntese

Nesta tese, um gêmeo digital não é apenas um modelo com boa perda de treino.
Ele é um modelo que, por regime, reproduz suficientemente bem o comportamento
medido do canal segundo uma escada explícita de validação.

A leitura atualmente sustentada pelo projeto é:

- `G1..G5` formam a validação principal do twin
- `G6` é uma tela estatística auxiliar
- checagens externas como `16QAM` ajudam a testar generalização
- métricas como `MI/GMI/NGMI/AIR` enriquecem a análise, mas não definem
  aceitação principal

## Por que a separação importa

Sem essa separação, o projeto mistura três coisas distintas:

- utilidade de engenharia
- conservadorismo estatístico
- capacidade de generalização externa

A auditoria metodológica mostrou que isso torna as conclusões mais frágeis.

## Evidências

- A recomendação de separar `validation_status_twin` de `stat_screen_pass`
  aparece explicitamente em
  [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
  e
  [gate_threshold_calibration](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md).
- A tabela de fundamentos da validação já tratava `16QAM` e screens auxiliares
  como peças fortes, mas não equivalentes ao protocolo principal, em
  [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md).

## Implicações

O capítulo metodológico da tese deve tratar validação de twin como sistema em
camadas:

1. aceitação principal do modelo
2. tela estatística auxiliar
3. validação externa e métricas auxiliares
