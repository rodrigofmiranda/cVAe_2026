# Glossário

## Propósito

Fixar um vocabulário comum para leitura da tese e reduzir ambiguidades entre
termos operacionais, estatísticos e de modelagem.

## Escopo

Este glossário cobre os termos centrais das linhas `full_square`,
`full_circle`, validação do twin e métricas auxiliares.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)
- [PROTOCOL](/home/rodrigo/cVAe_2026_full_square/docs/reference/PROTOCOL.md)
- [gate_validation_audit](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/gate_validation_audit_2026-04-11.md)

## Status do texto

`curado`

## Síntese

| Termo | Leitura adotada nesta tese |
| --- | --- |
| `VLC` | comunicação por luz visível |
| `IM/DD` | modulação por intensidade e detecção direta; o canal opera com restrições físicas de intensidade, não com sinal complexo arbitrário |
| `digital twin` | modelo gerativo condicionado que busca reproduzir, por regime, o comportamento do canal medido |
| `full_square` | dataset de excitação densa em suporte quadrado no plano IQ, usado como linha principal de identificação |
| `full_circle` | dataset de suporte circular real, usado como trilha separada para testar a hipótese de geometria |
| `shape` | linha que reinterpretou o gargalo de `full_square` como problema parcialmente ligado à geometria do suporte |
| `proxy geométrico` | intervenção feita ainda em `full_square` para simular efeitos de outra geometria de suporte |
| `baseline científica` | comparação metodologicamente limpa, sem vieses extras desnecessários |
| `linha operacional` | sequência prática de tentativas e refinamentos que levou aos runs mais fortes |
| `MDN` | mistura de distribuições no decoder, usada para melhorar a lei condicional em relação ao decoder gaussiano simples |
| `seq_bigru_residual` | família sequencial residual com janela curta, que modela dependência temporal local |
| `pointwise revival` | retomada controlada de famílias ponto a ponto locais, para reavaliar a hipótese de lei local |
| `G1..G5` | escada principal de validação do twin, usada em `validation_status_twin` |
| `G6` | tela estatística auxiliar baseada em MMD/Energy; deve ser reportada separadamente |
| `validation_status_twin` | decisão principal do twin baseada em `G1..G5` |
| `stat_screen_pass` | resultado da tela estatística auxiliar baseada em `G6` |
| `validation_status_full` | status conservador que combina `G1..G6` |
| `JBrel` | medida relativa ligada ao teste de Jarque-Bera usada dentro de `G5` como shape gate do projeto |
| `MI` | mutual information auxiliar |
| `GMI` | generalized mutual information auxiliar |
| `NGMI` | versão normalizada de GMI, auxiliar |
| `AIR` | achievable information rate auxiliar |
| `16QAM` | conjunto externo de validação e generalização, não equivalente ao `full_square` |

## Implicações

O glossário reduz dois riscos frequentes:

- usar a mesma palavra para coisas metodologicamente diferentes
- transportar para a tese leituras operacionais mais fortes do que os dados
  realmente sustentam
