# Escopo e Convenções

## Propósito

Definir o papel editorial da pasta `Tese/` e padronizar a forma como os
documentos são escritos, lidos e rastreados.

## Escopo

Este arquivo estabelece:

- o que entra e o que não entra na camada `Tese`
- a taxonomia de status dos textos
- a diferença entre síntese, fonte canônica e evidência
- o template mínimo dos documentos

## Fontes canônicas usadas

- [Tese/README](../README.md)
- [docs/README](/home/rodrigo/cVAe_2026_mdn_return/docs/README.md)
- [PROJECT_STATUS](/home/rodrigo/cVAe_2026_mdn_return/PROJECT_STATUS.md)
- [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)

## Status do texto

`curado`

## Pergunta científica

Como transformar documentação espalhada de laboratório em uma base textual de
tese que seja legível, rastreável e metodologicamente honesta?

## Síntese

A pasta `Tese/` segue quatro princípios:

1. síntese antes de acúmulo
2. rastreabilidade antes de memória de chat
3. separação entre operação e argumentação científica
4. distinção explícita entre validação principal e evidência auxiliar

## Regras editoriais

- Cada documento deve começar com:
  - `Propósito`
  - `Escopo`
  - `Fontes canônicas usadas`
  - `Status do texto`
- Sempre que útil, o corpo deve seguir:
  - `Pergunta científica`
  - `Síntese`
  - `Evidências`
  - `Implicações`
- Afirmações fortes devem apontar para um artefato ou documento específico.
- Valores numéricos devem ser apresentados como leitura estabilizada, não como
  transcrição bruta de terminal.
- A redação deve ser em português técnico, mais próxima de tese do que de log.

## Taxonomia de status

| Status | Uso esperado |
| --- | --- |
| `rascunho` | texto ainda incompleto, mas já útil para organizar o raciocínio |
| `curado` | síntese consistente, com fontes canônicas claras |
| `quase pronto para tese` | texto já próximo da redação final, exigindo apenas acabamento e normalização bibliográfica |

## Semântica das evidências

| Tipo | Significado |
| --- | --- |
| baseline científica | referência usada para uma comparação metodologicamente limpa |
| linha operacional | sequência de decisões práticas que levou aos melhores runs ou às melhores triagens |
| evidência auxiliar | diagnóstico, screen estatístico, métrica complementar ou checagem externa |

## Template mínimo

Todo novo documento desta pasta deve responder, no mínimo:

- qual pergunta científica ele aborda
- qual a leitura estabilizada atual
- quais fontes canônicas sustentam essa leitura
- quais implicações isso traz para a tese

## Leituras que devem ser evitadas

- tratar `Tese/` como substituto de `outputs/`
- usar `G6` como prova de equivalência do canal
- confundir proxy de geometria em `full_square` com validação real em `full_circle`
- tratar resultado operacional máximo como baseline científica limpa

## Evidências

- A separação `validation_status_twin` versus `stat_screen_pass` foi consolidada
  nas sínteses de validação em
  [gate_validation_audit](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/gate_validation_audit_2026-04-11.md)
  e
  [gate_threshold_calibration](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md).

## Implicações

Esta convenção permite que a tese seja escrita sem depender de reconstruções
orais do tipo “o que tínhamos decidido naquela época”, porque a camada já
deixa explícitos escopo, fonte e status de cada peça textual.
