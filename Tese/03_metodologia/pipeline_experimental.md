# Pipeline Experimental

## Propósito

Descrever o fluxo experimental do projeto, desde aquisição e organização dos
dados até treino, avaliação e curadoria dos resultados.

## Escopo

Este documento cobre o pipeline em nível metodológico, não o detalhe de cada
grid ou script operacional.

## Fontes canônicas usadas

- [VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING](/home/rodrigo/cVAe_2026_mdn_return/docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt)
- [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md)
- [vlc_shaping_experimental_methodology](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/vlc_shaping_experimental_methodology_2026-04-03.md)
- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)

## Status do texto

`curado`

## Pergunta científica

Como o projeto garante que as comparações entre modelos, geometrias e regimes
sejam feitas dentro de um fluxo coerente de aquisição, treino e avaliação?

## Síntese

O pipeline pode ser lido em seis estágios:

1. aquisição e organização do dataset
2. definição do recorte experimental e dos regimes
3. treino do modelo compartilhado ou por regime
4. avaliação multi-regime sob protocolo comum
5. geração de artefatos tabulares e visuais
6. curadoria científica das decisões

## Etapas do pipeline

### 1. Aquisição e organização

- sinais enviados e recebidos são organizados por distância, corrente e
  experimento
- o dataset é tratado como fonte física de identificação do canal

### 2. Seleção e split

- o projeto trabalha com seleção explícita de regimes
- em vários momentos a leitura mais importante é por regime, não pelo agregado

### 3. Treino

- há famílias ponto a ponto e sequenciais
- há linhas com treino global compartilhado e avaliação em todos os regimes
- a linha `shape` e a linha `full_circle` reutilizam essa estrutura de
  protocolo, mudando hipóteses e dados

### 4. Avaliação

- a avaliação computa métricas físicas, residuais, estatísticas e auxiliares
- os resultados são resumidos em tabelas e artefatos de protocolo

### 5. Curadoria

- runs fortes viram referências
- screenings comparativos viram sínteses
- decisões de promoção, descarte ou continuidade são registradas em documentos
  próprios

## Evidências

- A organização do fluxo de dados aparece em
  [VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING](/home/rodrigo/cVAe_2026_mdn_return/docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt).
- O protocolo de avaliação e seus campos principais aparecem em
  [PROTOCOL](/home/rodrigo/cVAe_2026_shape/docs/reference/PROTOCOL.md).

## Implicações

Na tese, este pipeline permite apresentar os resultados não como uma coleção
de treinos isolados, mas como um sistema experimental reproduzível.
