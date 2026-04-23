# Shape

## Propósito

Registrar a linha `shape` como reinterpretação científica do gargalo de
`full_square`.

## Escopo

Este documento cobre:

- hipótese de suporte
- ablações iniciais `E0..E4`
- screening científico `A/B/C/D`

## Fontes canônicas usadas

- [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
- [support_ablation_e0_e3_comparison](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/support_ablation_e0_e3_comparison_2026-04-07.md)
- [support_ablation_e3b_e4_followup](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/support_ablation_e3b_e4_followup_2026-04-07.md)
- [support_hyperparameter_scientific_screening](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/support_hyperparameter_scientific_screening_2026-04-08.md)
- [support_scientific_screen_master_table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/support_scientific_screen_master_table_2026-04-10.md)

## Status do texto

`curado`

## Pergunta científica

Se o erro residual remanescente estiver concentrado nas regiões extremas do
suporte quadrado, então intervenções orientadas por suporte melhoram borda,
cobertura e fidelidade?

## Síntese

A linha `shape` foi uma linha de proxy geométrico sobre o mesmo `full_square`.
Ela não deve ser lida como `full_circle` antecipado.

As ablações iniciais `E0..E4` mostraram que:

- pressão orientada por borda e canto podia ajudar
- variantes mais agressivas precisavam ser lidas com cuidado
- ganho operacional não equivalia automaticamente a baseline científica

Depois disso, o programa avançou para um screening científico mais limpo sobre
a família robusta `E2`.

## Resultado do screening A/B/C/D

| Bloco | Ideia principal | Melhor candidato limpo | Resultado resumido |
| --- | --- | --- | --- |
| `A` | localização do peso de suporte | `S27cov_sciv1_ctrl_lc0p25_t0p03_a1p50_tau0p75_tc0p35_wmax3p0` | `5/12` |
| `B` | cobertura e caudas | `S27cov_sciv1_tail98_lc0p25_t0p03` | `4/12` |
| `C` | capacidade de modelo | `S27cov_sciv1_lat10` | `4/12` |
| `D` | otimização e regularização | `S27cov_sciv1_lr0p00015` | `4/12` |

## Leitura estabilizada

O screening mudou a leitura da linha:

- o controle robusto `E2` permaneceu a principal referência
- `lr0p00015` e `tail98` sobreviveram como secundários úteis
- variantes geométricas mais engenheiradas não justificaram promoção como nova
  baseline principal

## Evidências

- A leitura consolidada do pós-screening está explicitamente escrita em
  [FULL_SQUARE_LINEAGE_TO_SHAPE](/home/rodrigo/cVAe_2026_mdn_return/docs/active/FULL_SQUARE_LINEAGE_TO_SHAPE.md)
  e numericamente resumida em
  [support_scientific_screen_master_table](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/support_scientific_screen_master_table_2026-04-10.md).

## Implicações

Na tese, `shape` deve ser apresentado como mecanismo de triagem científica e
engenharia orientada por hipótese, não como comprovação definitiva de que a
solução correta é impor geometria forte ao treinamento.
