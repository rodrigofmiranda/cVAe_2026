# Decisões Baseadas em Dados

## Propósito

Transformar a história `full_square` em uma lista explícita de decisões que
foram sustentadas por evidência e não apenas por preferência de implementação.

## Escopo

Este arquivo cobre decisões já suficientemente maduras para entrar na narrativa
principal da tese.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [support_scientific_screen_master_table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/support_scientific_screen_master_table_2026-04-10.md)
- [support_hyperparameter_scientific_screening](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/support_hyperparameter_scientific_screening_2026-04-08.md)

## Status do texto

`curado`

## Síntese

| Decisão | Base empírica | Leitura adotada |
| --- | --- | --- |
| manter leitura residual, não apenas output direto | linhas point-wise antigas não fechavam o problema | decisão consolidada |
| promover família sequencial como baseline forte | contexto temporal curto melhorou a leitura do canal | decisão consolidada |
| usar `MDN` como linha principal sobre decoder gaussiano | `MDN` recuperou o problema até `10/12` e isolou o gargalo remanescente | decisão consolidada |
| não declarar a hipótese sequencial como única | `pointwise revival` local também chegou a `10/12` | decisão consolidada |
| reinterpretar o gargalo como parcialmente geométrico | falha ficou concentrada em regimes e regiões sensíveis do suporte | decisão consolidada |
| não promover variantes `shape` mais agressivas como baseline principal | screening A/B/C/D recolocou o controle `E2` no topo | decisão consolidada |
| manter `lr0p00015` e `tail98` como secundários científicos | tiveram leitura melhor do que outras expansões de complexidade | decisão consolidada |

## O que não foi decidido

- que `full_circle` é superior por definição
- que hard priors geométricos são a resposta final
- que mais capacidade latente, sozinha, fecha o gargalo

## Evidências

- O ordenamento final das decisões aparece em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).
- O screening científico que sustentou a reordenação do bloco `shape` está em
  [support_scientific_screen_master_table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/support_scientific_screen_master_table_2026-04-10.md).

## Implicações

Este quadro vira a ponte natural entre o capítulo de resultados `full_square` e
o capítulo de discussão da tese, porque ele transforma experimentos em
decisões.
