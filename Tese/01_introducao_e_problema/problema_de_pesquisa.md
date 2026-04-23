# Problema de Pesquisa

## Propósito

Definir o problema científico central que organiza a linha VLC cVAE nesta
etapa da tese.

## Escopo

Este arquivo formula o problema em nível de tese, sem entrar ainda nos detalhes
de família de modelo ou de cronologia experimental.

## Fontes canônicas usadas

- [README do projeto](/home/rodrigo/cVAe_2026_mdn_return/README.md)
- [PROJECT_STATUS](/home/rodrigo/cVAe_2026_mdn_return/PROJECT_STATUS.md)
- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)

## Status do texto

`curado`

## Pergunta científica

Como construir e validar um gêmeo digital gerativo para um canal VLC IM/DD que
reproduza, de forma condicionada e multi-regime, não apenas erros médios, mas
também estrutura residual, cobertura, caudas e fidelidade estatística útil?

## Síntese

O problema não é apenas regressão de canal. O problema é aproximar a lei
condicional do canal medido sob diferentes distâncias e correntes, mantendo
interpretação física e protocolo de validação suficientemente fortes para
sustentar a noção de gêmeo digital.

Ao longo do projeto, três dificuldades se tornaram centrais:

- a fidelidade não pode ser julgada só por erro médio
- o gargalo residual ficou concentrado em regimes curtos e de baixa corrente
- a geometria do suporte dos sinais enviados pode influenciar esse gargalo

## Por que isso importa

- Um modelo que acerta apenas média ou variância não basta para uso como twin.
- Um protocolo sem distinção entre validação principal e screen estatístico
  confunde engenharia com teste formal.
- Um resultado forte em `full_square` não prova automaticamente a melhor
  geometria de excitação.

## Evidências

- A linha `full_square` foi progressivamente refinada até localizar o gargalo
  em regimes `0.8 m` de baixa corrente, conforme
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).
- A auditoria metodológica mostrou que o twin deve ser lido sobretudo por
  `G1..G5`, com `G6` separado como tela estatística, conforme
  [gate_validation_audit](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/gate_validation_audit_2026-04-11.md).

## Implicações

O problema de pesquisa da tese não pode ser escrito como “treinar um cVAE para
prever o canal”. A formulação correta exige:

- modelagem gerativa condicionada
- validação multi-regime
- leitura de shape residual
- comparação entre geometrias de excitação
