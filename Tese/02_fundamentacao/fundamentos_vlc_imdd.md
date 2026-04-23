# Fundamentos VLC IM/DD

## Propósito

Consolidar a base conceitual mínima de VLC IM/DD necessária para interpretar os
experimentos desta tese.

## Escopo

Este texto cobre:

- o tipo de canal modelado
- a consequência da restrição IM/DD para o problema
- por que a geometria do sinal de entrada importa

## Fontes canônicas usadas

- [vlc_shaping_experimental_methodology](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/vlc_shaping_experimental_methodology_2026-04-03.md)
- [vlc_probabilistic_shaping_strategy](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/vlc_probabilistic_shaping_strategy_2026-04-03.md)
- [askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance](/home/rodrigo/cVAe_2026_shape/knowledge/notes/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance.md)
- [shu_2020_probabilistic_shaping_optical](/home/rodrigo/cVAe_2026_shape/knowledge/notes/shu_2020_probabilistic_shaping_optical.md)
- [Askari e Lampe 2025 parsed](/home/rodrigo/cVAe_2026_shape/knowledge/papers/parsed/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance/document.md)
- [Shu e Zhang 2020 parsed](/home/rodrigo/cVAe_2026_shape/knowledge/papers/parsed/shu_2020_probabilistic_shaping_optical/document.md)
- [fundamentos do fluxo de dados](/home/rodrigo/cVAe_2026_mdn_return/docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt)
- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)

## Status do texto

`curado`

## Pergunta científica

Quais propriedades do canal VLC IM/DD tornam inadequada uma leitura puramente
regressiva e justificam o uso de modelagem gerativa condicionada?

## Síntese

O canal estudado é um canal físico de intensidade óptica observada por
detecção direta. Isso impõe duas consequências importantes para a tese:

- a estrutura do sinal útil não é livre de restrições físicas
- a distribuição de saída não pode ser resumida adequadamente por uma média
  condicional e um erro quadrático

Na prática, isso significa que:

- a lei residual importa
- a cobertura e as caudas importam
- a fidelidade espectral e estatística importa
- a forma como o espaço de entrada é excitado pode alterar o tipo de região do
  canal que é observada e aprendida

## Papel do suporte de entrada

O projeto não trata `full_square` e `full_circle` como meras escolhas visuais.
Eles representam geometrias diferentes de excitação do canal e, portanto,
hipóteses diferentes sobre:

- quais regiões do espaço de entrada são mais estressantes
- quanto das falhas vem do modelo
- quanto das falhas vem do próprio desenho experimental

Essa leitura é compatível com a base técnica já disponível em `knowledge`. As
notas de Askari e Lampe (2025) e de Shu e Zhang (2020) sustentam a ideia de
que a melhor estratégia de sinalização em canal óptico real depende não apenas
da marginal do sinal, mas também de momentos, estrutura temporal, restrições
práticas e geometria efetiva de transmissão.

## Evidências

- A linha `shape` passou a reinterpretar o gargalo remanescente como problema
  parcialmente ligado à geometria de suporte, conforme
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).
- As sínteses de `probabilistic shaping` reforçam que sinais não uniformes e
  restrições de suporte alteram a física observada e a leitura de métricas, em
  [vlc_probabilistic_shaping_strategy](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/vlc_probabilistic_shaping_strategy_2026-04-03.md).
- A base técnica mais direta para essa leitura, já curada em `knowledge`,
  aparece em
  [askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance](/home/rodrigo/cVAe_2026_shape/knowledge/notes/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance.md)
  e
  [shu_2020_probabilistic_shaping_optical](/home/rodrigo/cVAe_2026_shape/knowledge/notes/shu_2020_probabilistic_shaping_optical.md).

## Implicações

O problema desta tese não deve ser reduzido a “ajustar melhor um mapeamento”.
Ele precisa ser formulado como modelagem da lei do canal sob restrições
físicas e sob diferentes geometrias de excitação.
