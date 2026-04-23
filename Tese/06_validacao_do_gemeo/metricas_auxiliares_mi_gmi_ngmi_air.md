# Métricas Auxiliares MI, GMI, NGMI e AIR

## Propósito

Explicar o papel das métricas de informação auxiliares na metodologia atual e
delimitar onde elas são informativas.

## Escopo

Este documento cobre:

- semântica metodológica
- disponibilidade no protocolo
- limites de interpretação

## Fontes canônicas usadas

- [PROTOCOL](/home/rodrigo/cVAe_2026_full_square/docs/reference/PROTOCOL.md)
- [information_metrics.py](/home/rodrigo/cVAe_2026_full_square/src/evaluation/information_metrics.py)
- [summary_plots.py](/home/rodrigo/cVAe_2026_full_square/src/evaluation/summary_plots.py)
- [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)
- [shu_2020_probabilistic_shaping_optical](/home/rodrigo/cVAe_2026_full_square/knowledge/notes/shu_2020_probabilistic_shaping_optical.md)
- [askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance](/home/rodrigo/cVAe_2026_full_square/knowledge/notes/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance.md)

## Status do texto

`curado`

## Pergunta científica

Como usar `MI`, `GMI`, `NGMI` e `AIR` de forma útil sem transformá-las em
gates indevidos?

## Síntese

O protocolo atual já prevê métricas auxiliares de informação quando o alfabeto
de entrada é discreto e inferível:

- `MI`
- `GMI`
- `NGMI`
- `AIR`

A leitura correta é:

- são diagnósticos auxiliares
- não participam de `G1..G6`
- são mais informativas em datasets discretos e repetitivos, como `16QAM`
- tendem a ficar indisponíveis ou pouco significativas em datasets densos como
  `full_square`

Essa interpretação está alinhada com a base técnica disponível em `knowledge`:
`NGMI`, `AIR` e métricas aparentadas são naturais quando existe alfabeto
discreto e leitura de comunicação digital, mas não devem ser forçadas como
critério principal em excitações densas contínuas usadas para identificação de
canal.

## Papel na tese

Essas métricas ajudam a responder perguntas como:

- o modelo preserva informação útil sob uma leitura de comunicação digital?
- a generalização em dados discretos acompanha a fidelidade residual?

Elas não respondem sozinhas:

- se o twin é válido pelo protocolo principal
- se o modelo reproduz adequadamente toda a lei do canal

## Evidências

- O protocolo declara explicitamente que essas métricas são auxiliares e não
  entram em `G1..G6`, em
  [PROTOCOL](/home/rodrigo/cVAe_2026_full_square/docs/reference/PROTOCOL.md).
- A implementação operacional está em
  [information_metrics.py](/home/rodrigo/cVAe_2026_full_square/src/evaluation/information_metrics.py).
- A conexão técnica dessas métricas com shaping óptico e regimes discretos pode
  ser apoiada por
  [shu_2020_probabilistic_shaping_optical](/home/rodrigo/cVAe_2026_full_square/knowledge/notes/shu_2020_probabilistic_shaping_optical.md)
  e, de forma mais ampla, por
  [askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance](/home/rodrigo/cVAe_2026_full_square/knowledge/notes/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance.md).

## Implicações

Na tese, essas métricas devem ser apresentadas como camada complementar de
análise, especialmente útil para validação externa em alfabetos discretos.
