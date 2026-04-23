# Shaping e Geometria de Suporte

## Propósito

Organizar, em linguagem de tese, a hipótese de que a geometria de suporte do
sinal enviado influencia a dificuldade de modelagem do canal.

## Escopo

Este texto não descreve ainda os runs. Ele esclarece o significado científico
da linha `shape` e da posterior linha `full_circle`.

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [FULL_CIRCLE_NEXT_STEP](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_NEXT_STEP.md)
- [vlc_probabilistic_shaping_strategy](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/vlc_probabilistic_shaping_strategy_2026-04-03.md)
- [support_ablation_e0_e3_comparison](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/support_ablation_e0_e3_comparison_2026-04-07.md)
- [askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance](/home/rodrigo/cVAe_2026_shape/knowledge/notes/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance.md)
- [shu_2020_probabilistic_shaping_optical](/home/rodrigo/cVAe_2026_shape/knowledge/notes/shu_2020_probabilistic_shaping_optical.md)

## Status do texto

`curado`

## Pergunta científica

Se o gargalo residual remanescente estiver associado às regiões extremas do
suporte quadrado, então mudar a geometria de excitação ou o peso dado a essas
regiões altera a fidelidade do twin?

## Síntese

A linha `shape` não começou como coleta real em `full_circle`. Ela começou como
programa de ablação no mesmo `full_square`, com três intenções:

- testar se bordas e cantos eram regiões decisivas do erro
- verificar se o suporte quadrado induzia regimes difíceis e pouco regulares
- distinguir ganho de engenharia de explicação científica

Esse enquadramento é coerente com a literatura técnica já sintetizada no
projeto: papers de shaping óptico mostram que a escolha de geometria ou de
probabilidade de excitação não deve ser avaliada apenas por ganho idealizado em
AWGN, mas sob restrições do canal real e possíveis efeitos de memória,
momento e sensibilidade a extremos do sinal.

Por isso, a leitura correta é:

- `shape` = linha de reinterpretacão de `full_square`
- `full_circle` = teste real e separado da hipótese geométrica

## Cuidados conceituais

- `full_square` não é sinônimo de QAM tradicional.
- `full_circle` não deve ser descrito como simples “filtro circular” do mesmo
  dado.
- proxy geométrico bem-sucedido não prova, por si só, que a aquisição real em
  círculo é a baseline certa.

## Evidências

- A história da linha `shape` e suas leituras mais honestas estão em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md).
- O reposicionamento científico de `full_circle` como trilha separada está em
  [FULL_CIRCLE_NEXT_STEP](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_CIRCLE_NEXT_STEP.md).
- A fundamentação técnica para não reduzir shaping a “ganho de marginal” está
  em
  [askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance](/home/rodrigo/cVAe_2026_shape/knowledge/notes/askari_lampe_2025_probabilistic_shaping_nonlinearity_tolerance.md)
  e
  [shu_2020_probabilistic_shaping_optical](/home/rodrigo/cVAe_2026_shape/knowledge/notes/shu_2020_probabilistic_shaping_optical.md).

## Implicações

Na tese, `shape` deve aparecer como etapa de triagem de hipóteses sobre
suporte. Já `full_circle` deve aparecer como validação experimental própria,
com baseline limpa e sem herdar automaticamente todos os vieses úteis da linha
operacional anterior.
