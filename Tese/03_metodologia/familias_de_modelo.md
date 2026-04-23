# Famílias de Modelo

## Propósito

Resumir as principais famílias de modelo exploradas e o que cada uma ensinou ao
projeto.

## Escopo

Este documento não entra em todos os grids. Ele organiza a hierarquia de
famílias que depois reaparece nas linhas `full_square` e `full_circle`.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [WORKING_STATE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/WORKING_STATE.md)
- [support_hyperparameter_scientific_screening](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/support_hyperparameter_scientific_screening_2026-04-08.md)

## Status do texto

`curado`

## Pergunta científica

Que tipos de família foram necessários para isolar o gargalo do canal e o que
cada um mostrou?

## Síntese

| Família | Papel na história | Leitura atual |
| --- | --- | --- |
| point-wise tradicional | ponto de partida da modelagem gerativa condicionada | útil como baseline conceitual, mas insuficiente como solução completa |
| residual point-wise | reparametrização da saída como lei residual | reforçou que o alvo certo não era apenas a saída absoluta |
| `seq_bigru_residual` | inclusão de contexto temporal curto | tornou-se base forte para a linha principal |
| decoder gaussiano | referência simples e importante para comparação | perdeu espaço frente ao MDN nos regimes mais difíceis |
| decoder `MDN` | melhora da lei condicional | peça central para recuperar fidelidade em `full_square` |
| point-wise revival local | revisão crítica da hipótese de lei local | mostrou que um point-wise estreito continua cientificamente vivo |
| `shape/support-aware` | família de ablações sobre suporte e ponderação | útil para screening de hipóteses, não como verdade final automática |

## Lição metodológica

O projeto não avançou por substituição linear de uma família pela outra. O que
ocorreu foi um refinamento sucessivo do problema:

- de erro médio para lei residual
- de lei local para dependência temporal
- de capacidade do decoder para shape residual
- de shape residual para hipótese de geometria

## Evidências

- A síntese histórica completa está em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).
- A etapa de screening científico das variantes `shape` está em
  [support_hyperparameter_scientific_screening](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/support_hyperparameter_scientific_screening_2026-04-08.md).

## Implicações

Na tese, as famílias de modelo devem ser apresentadas como respostas a
perguntas específicas do programa científico, não como simples troca de
arquiteturas por conveniência.
