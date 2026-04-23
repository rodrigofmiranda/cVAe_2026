# Validação Externa 16QAM

## Propósito

Registrar o papel da validação externa em `16QAM` dentro da metodologia da
tese.

## Escopo

Este texto trata do papel metodológico dessa validação. A consolidação numérica
fina ainda pode crescer em versões futuras desta camada.

## Fontes canônicas usadas

- [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md)
- [full_circle_soft_radial_master_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/full_circle_soft_radial_master_table_2026-04-22.md)
- [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md)
- [crossline summary with soft-radial](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260422_plus_soft_radial/crossline_summary/README.md)
- [DATASET_LFS_UPLOAD](/home/rodrigo/cVAe_2026_mdn_return/docs/operations/DATASET_LFS_UPLOAD.md)
- [dataset 16QAM no shape repo](/home/rodrigo/cVAe_2026_shape/data/16qam)

## Status do texto

`rascunho`

## Pergunta científica

Qual é o papel correto de `16QAM` em uma tese cujo foco principal de treino foi
`full_square` e `full_circle`?

## Síntese

`16QAM` entra como validação externa de generalização. Ele é importante porque:

- usa um conjunto discreto e tradicionalmente interpretável
- permite comparar o modelo em um domínio fora do dataset principal
- habilita métricas auxiliares de informação quando o alfabeto é inferível

Mas ele não deve ser tratado como:

- substituto do protocolo principal
- novo conjunto-base de treino desta etapa
- atalho para concluir sobre a fidelidade do twin em `full_square`

## Estado atual

- o papel metodológico de `16QAM` já está reconhecido
- os dados existem e estão organizados
- a comparação crossline de quatro braços já foi consolidada:
  `full_square`, `full_circle clean`, `full_circle geometry-biased` e
  `full_circle geometry-light`
- o foco agora não é mais lançar o quarto braço, mas interpretar corretamente
  o que esse quarto braço muda, ou não muda, na leitura científica

## Linhas já comparadas com 16QAM

Até o momento, a validação externa em `16QAM` foi realizada de forma efetiva em
duas camadas:

- na linha `full_square/shape`, com os artefatos organizados em
  [eval_16qam_sel4curr_review](/home/rodrigo/cVAe_2026_shape/outputs/analysis/eval_16qam_sel4curr_review/README.md)
- na rodada comparativa crossline `full_square` versus `full_circle clean`
  versus `full_circle geometry-biased`, consolidada em
  `eval_16qam_crossline_20260420_clean`
- na extensão posterior dessa rodada, que adicionou o representante
  `full_circle geometry-light` fechado depois na linha `soft-radial`, com
  consolidação em
  `eval_16qam_crossline_20260422_plus_soft_radial/crossline_summary`

## Rodada comparativa full_square vs full_circle

### Objetivo desta rodada

Esta rodada foi aberta para responder a uma pergunta prática importante para a
tese:

- o desempenho em `16QAM` confirma ou enfraquece a leitura obtida no protocolo
  principal quando comparamos `full_square` com `full_circle`?

O papel desta rodada continua sendo de validação externa. Portanto, ela deve
funcionar como balizador de generalização e robustez, e não como substituto da
aceitação principal do gêmeo digital.

### Candidatos selecionados

Foram escolhidos quatro representantes, um da linha `full_square` e três da
linha `full_circle`, para permitir uma leitura comparativa direta:

| linha | candidato | papel na comparação | evidência canônica |
| --- | --- | --- | --- |
| `full_square` | `S27cov_sciv1_lr0p00015` | melhor representante atual da linha `shape/full_square` | [protocol_leaderboard full_square](/home/rodrigo/cVAe_2026_shape/outputs/support_ablation/final_grid/e2_finalists_shortlist/exp_20260414_131231/tables/protocol_leaderboard.csv) |
| `full_circle clean` | `S27cov_fc_clean_lc0p25_t0p03_lat10` | baseline científica limpa, sem viés geométrico explícito | [protocol_leaderboard full_circle clean](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_b/exp_20260417_115142/tables/protocol_leaderboard.csv) |
| `full_circle geometry-biased` | `S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192` | linha operacional que favorece a hipótese geométrica circular | [protocol_leaderboard full_circle disk](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/disk_bs8192_lat10_100k_split_a/exp_20260416_165643/tables/protocol_leaderboard.csv) |
| `full_circle geometry-light` | `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0` | compromisso entre baseline limpa e viés geométrico forte, sem `geom3` e sem `disk_l2` | [protocol_leaderboard full_circle soft-radial](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256/tables/protocol_leaderboard.csv) |

### Regimes avaliados

A rodada comparativa usa o subconjunto já adotado como triagem externa rápida:

- distâncias: `0.8 m`, `1.0 m`, `1.5 m`
- correntes: `100`, `300`, `500`, `700 mA`
- total esperado: `12 regimes`

### Quadro de acompanhamento

- [x] consolidar o histórico já executado em `full_square`
- [x] identificar um candidato campeão `full_square`
- [x] identificar um candidato `full_circle clean`
- [x] identificar um candidato `full_circle geometry-biased`
- [x] lançar a avaliação externa `16QAM` para `full_square`
- [x] lançar a avaliação externa `16QAM` para `full_circle clean`
- [x] lançar a avaliação externa `16QAM` para `full_circle geometry-biased`
- [x] lançar a avaliação externa `16QAM` para `full_circle geometry-light`
- [x] consolidar métricas comparativas em uma tabela única
- [x] registrar uma leitura científica preliminar desta rodada

### Artefatos esperados desta rodada

Os artefatos desta comparação devem ser mantidos como índices curados, sem
duplicação de tabelas brutas nesta camada `Tese`. O esperado é registrar:

- um diretório de saída por candidato
- um manifesto por regime
- um sumário comparativo consolidado
- referência cruzada com plots-chave quando a leitura visual for necessária

### Artefatos consolidados desta rodada

Nesta comparação, dois roots canônicos devem ser citados em conjunto:

- `eval_16qam_crossline_20260420_clean`, que consolida os três braços
  originalmente lançados
- `eval_16qam_crossline_20260422_plus_soft_radial`, que adiciona o quarto braço
  `geometry-light` e traz o sumário comparativo atualizado

- `full_square`
  - [manifest_all_regimes_eval.csv](/home/rodrigo/cVAe_2026_shape/outputs/analysis/eval_16qam_crossline_20260420_clean/full_square_s27cov_sciv1_lr0p00015/manifest_all_regimes_eval.csv)
- `full_circle clean`
  - [manifest_all_regimes_eval.csv](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260420_clean/full_circle_clean_lat10/manifest_all_regimes_eval.csv)
- `full_circle geometry-biased`
  - [manifest_all_regimes_eval.csv](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260420_clean/full_circle_disk_geom3/manifest_all_regimes_eval.csv)
- `full_circle geometry-light`
  - [manifest_all_regimes_eval.csv](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260422_plus_soft_radial/full_circle_soft_rinf_local/manifest_all_regimes_eval.csv)
- sumário comparativo dos quatro braços
  - [README.md](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260422_plus_soft_radial/crossline_summary/README.md)

## Resultado estabilizado desta rodada

### Fechamento operacional

A rodada comparativa foi concluída com `12/12 regimes` completos para os quatro
candidatos selecionados.

### Síntese comparativa

Tomando como indicadores de leitura externa tanto a fidelidade clássica
(`|ΔEVM|`, `|ΔSNR|`, `ΔPSD`) quanto as métricas auxiliares de informação
(`MI`, `GMI`, `NGMI`, `AIR`), a ordenação observada nesta rodada foi:

| candidato | regimes concluídos | `|ΔEVM|` médio | `|ΔSNR|` médio | `ΔPSD` médio | `MI_pred` médio | `GMI_pred` médio | `NGMI_pred` médio | leitura preliminar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_square` | `12/12` | `0.773 %` | `0.263 dB` | `0.060` | `2.257` | `2.180` | `0.545` | melhor robustez global de generalização externa |
| `full_circle clean` | `12/12` | `4.037 %` | `0.843 dB` | `0.103` | `2.083` | `2.000` | `0.500` | competitivo em parte dos regimes de `0.8 m`, mas atrás globalmente |
| `full_circle geometry-biased` | `12/12` | `3.814 %` | `0.923 dB` | `0.107` | `2.104` | `2.033` | `0.508` | algum ganho local de informação, porém com pior estabilidade global |
| `full_circle geometry-light` | `12/12` | `3.956 %` | `0.871 dB` | `0.103` | `2.099` | `2.015` | `0.504` | melhora a baseline clean em boa parte dos regimes médios e longos, mas não muda a liderança global |

Por contagem de vitórias regime a regime, quando os quatro braços são
comparados simultaneamente:

- em `|ΔEVM|`, `|ΔSNR|` e `ΔPSD`, `full_square` venceu `9/12` regimes
- `full_circle clean` venceu `3/12` regimes, todos concentrados em `0.8 m`
- `full_circle geometry-biased` não venceu em `EVM/SNR/PSD`, embora tenha sido
  o melhor em `MI/GMI` em `2/12` regimes
- `full_circle geometry-light` também não venceu em `EVM/SNR/PSD` no ranking
  conjunto, mas foi o melhor em `MI/GMI` em `1/12` regime

Na leitura relativa dentro do próprio `full_circle`, o quarto braço trouxe uma
mudança importante:

- `geometry-light` foi melhor que `clean` em `8/12` regimes para `|ΔEVM|` e
  `|ΔSNR|`, e em `7/12` regimes para `ΔPSD`
- ele superou a baseline clean nos três indicadores clássicos ao mesmo tempo em
  `7/12` regimes: `1.0 m / 100, 300, 500 mA` e `1.5 m / 100, 300, 500, 700 mA`
- contra o `geometry-biased`, o `geometry-light` foi melhor em `4/12` regimes
  nos três indicadores clássicos, mas ainda perdeu a comparação global média

Nos testes estatísticos rápidos, as médias de `q-value` ficaram em:

- `full_square`: `MMD q ≈ 0.0058`, `Energy q ≈ 0.0058`
- `full_circle clean`: `MMD q ≈ 0.0050`, `Energy q ≈ 0.0050`
- `full_circle geometry-biased`: `MMD q ≈ 0.0162`, `Energy q ≈ 0.0216`
- `full_circle geometry-light`: `MMD q ≈ 0.0050`, `Energy q ≈ 0.0050`

Esses números reforçam que `G6/stat_screen` deve continuar separado da leitura
principal do twin: a linha `geometry-biased` teve `q-values` um pouco menos
restritivos, mas isso não se converteu em melhor fidelidade externa global.

### Ponto de atenção

- a linha `full_circle geometry-biased` apresentou um outlier forte em
  `dist_0p8m__curr_500mA`, com explosão de `Δkurt_l2` e do indicador baseado em
  Jarque-Bera; isso enfraquece bastante sua leitura média nesta rodada

### Leitura científica preliminar

Esta rodada não sustentou a hipótese de que `full_circle` já superaria o melhor
representante `full_square` quando submetido a uma validação externa discreta em
`16QAM`.

A leitura mais prudente, no estado atual, é:

- `full_square` continua sendo o melhor balizador externo de generalização
- `full_circle clean` preserva interesse científico porque foi o melhor em
  `3/12` regimes, todos nos casos mais difíceis de `0.8 m`
- `full_circle geometry-biased` pode elevar métricas auxiliares de informação em
  alguns regimes, mas hoje não sustenta a mesma qualidade quando a comparação é
  feita por fidelidade externa global
- `full_circle geometry-light` é o melhor compromisso metodológico dentro da
  família `full_circle`: melhora a baseline clean em boa parte de `1.0 m` e
  `1.5 m`, mantém leitura estatística semelhante à linha clean, mas ainda não
  altera o veredito global a favor de `full_circle`

## Evidências

- O papel forte, mas auxiliar, da validação externa aparece em
  [digital_twin_validation_foundation_table](/home/rodrigo/cVAe_2026_shape/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md).
- A checklist `full_circle` já prevê explicitamente usar o modelo treinado como
  checagem externa em `16QAM`, em
  [FULL_CIRCLE_VALIDATION_CHECKLIST](/home/rodrigo/cVAe_2026_shape_fullcircle/docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md).
- A execução já consolidada da linha `full_square` pode ser auditada em
  [eval_16qam_sel4curr_review](/home/rodrigo/cVAe_2026_shape/outputs/analysis/eval_16qam_sel4curr_review/README.md).
- Os candidatos escolhidos para a rodada comparativa estão ancorados em seus
  respectivos leaderboards de protocolo:
  [full_square](/home/rodrigo/cVAe_2026_shape/outputs/support_ablation/final_grid/e2_finalists_shortlist/exp_20260414_131231/tables/protocol_leaderboard.csv),
  [full_circle clean](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_b/exp_20260417_115142/tables/protocol_leaderboard.csv),
  [full_circle disk](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/disk_bs8192_lat10_100k_split_a/exp_20260416_165643/tables/protocol_leaderboard.csv),
  [full_circle soft-radial](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256/tables/protocol_leaderboard.csv).

## Implicações

Na tese, `16QAM` deve compor a seção de validação externa e generalização, não
o núcleo da aceitação principal do twin.

Ao mesmo tempo, esta validação externa é particularmente útil como árbitro entre
linhas concorrentes quando o protocolo principal produz leituras próximas ou
quando uma linha nova, como `full_circle`, ainda precisa provar robustez fora do
domínio de treino principal.
