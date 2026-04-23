# Datasets e Aquisição

## Propósito

Explicitar quais conjuntos de dados sustentam esta tese e qual papel
metodológico cada um desempenha.

## Escopo

Este documento cobre:

- `full_square`
- `full_circle`
- `16QAM`

## Fontes canônicas usadas

- [VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING](/home/rodrigo/cVAe_2026_mdn_return/docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt)
- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)
- [DATASET_LFS_UPLOAD](/home/rodrigo/cVAe_2026_mdn_return/docs/operations/DATASET_LFS_UPLOAD.md)

## Status do texto

`curado`

## Pergunta científica

Quais datasets existem, por que eles não são equivalentes entre si e como cada
um deve entrar na argumentação da tese?

## Síntese

| Dataset | Papel principal na tese | Leitura correta |
| --- | --- | --- |
| `full_square` | linha principal de identificação e desenvolvimento de famílias de modelo | dataset denso de excitação quadrada, não equivalente a uma modulação digital tradicional |
| `full_circle` | teste experimental separado da hipótese geométrica | nova geometria de suporte, não simples recorte circular do mesmo dado |
| `16QAM` | validação externa e teste de generalização | conjunto externo e discreto, útil para comparar métricas de informação e robustez fora do dataset principal |

## Observações metodológicas

- `full_square` foi o ambiente em que o problema foi entendido e refinado.
- `full_circle` foi usado para verificar a hipótese geométrica em base real.
- `16QAM` não entra para substituir o protocolo principal; ele entra como
  checagem externa de generalização.

## Evidências

- A distinção entre `full_square` e `full_circle` está consolidada em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
  e
  [FULL_CIRCLE_EXECUTION_PLAN](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_EXECUTION_PLAN.md).
- O papel de `16QAM` como checagem externa aparece em
  [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)
  e em
  [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md).

## Implicações

Na tese, não se deve comparar esses datasets como se fossem três versões do
mesmo problema. Eles cumprem papéis diferentes dentro do programa científico.
