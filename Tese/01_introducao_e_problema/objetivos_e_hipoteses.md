# Objetivos e Hipóteses

## Propósito

Explicitar os objetivos da tese e as hipóteses que motivaram as principais
linhas experimentais.

## Escopo

Este documento conecta os objetivos gerais aos blocos experimentais que
aparecem depois em `full_square`, `full_circle` e validação.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [vlc_probabilistic_shaping_strategy](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/vlc_probabilistic_shaping_strategy_2026-04-03.md)
- [support_hyperparameter_scientific_screening](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/support_hyperparameter_scientific_screening_2026-04-08.md)
- [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)

## Status do texto

`curado`

## Síntese

### Objetivos

1. Construir um gêmeo digital gerativo condicionado para o canal VLC IM/DD.
2. Validar esse twin por uma escada metodológica mais forte do que erro médio.
3. Entender se o gargalo residual remanescente vem de capacidade do modelo,
   temporalidade, lei local ou geometria do suporte.
4. Separar claramente baseline científica, heurística operacional e evidência
   auxiliar.
5. Preparar uma narrativa de tese baseada em decisões rastreáveis.

### Hipóteses de trabalho

| Hipótese | Formulação de trabalho | Situação atual |
| --- | --- | --- |
| `H1` | Dependência temporal local importa; famílias sequenciais devem superar point-wise ingênuo. | fortalecida |
| `H2` | Decoder MDN melhora a lei condicional e recupera boa parte da falha que o decoder gaussiano não resolve. | fortalecida |
| `H3` | O gargalo principal em `full_square` não é global; ele fica concentrado em `0.8 m` e baixa corrente. | fortalecida |
| `H4` | Parte da falha remanescente depende da geometria do suporte do sinal enviado. | plausível, ainda não fechada |
| `H5` | Proxy geométrico em `full_square` ajuda a triar hipóteses, mas não substitui `full_circle` real. | fortalecida |
| `H6` | `G1..G5` são base mais adequada para aceitação do twin do que `G6` sozinho. | fortalecida |
| `H7` | Validação externa em `16QAM` ajuda a testar generalização, mas não substitui o protocolo principal. | fortalecida |

## Evidências

- `H1`, `H2` e `H3` aparecem de forma convergente na reconstrução de linhagem
  em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).
- `H4` e `H5` sustentam a linha `shape` e o posterior restart limpo em
  `full_circle`, como discutido em
  [FULL_CIRCLE_NEXT_STEP](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_NEXT_STEP.md)
  e
  [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md).
- `H6` e `H7` aparecem explicitamente na auditoria de validação em
  [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md).

## Implicações

As hipóteses acima organizam a tese não como um catálogo de grids, mas como um
programa científico: cada família de experimentos existe para enfraquecer,
confirmar ou refinar uma dessas hipóteses.
