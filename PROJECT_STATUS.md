# PROJECT_STATUS

> Atualizado em 2026-04-22.
> Estado oficial da worktree Full Circle antes de retornar para a linha MDN em
> outra pasta e branch.

## Repo E Branch

- repo local: `/home/rodrigo/cVAe_2026_shape_fullcircle`
- branch ativa: `research/full-circle`
- remote ativa: `origin git@github.com:rodrigofmiranda/cVAe_2026.git`
- objetivo desta branch: avaliar a linha `Full Circle` separadamente da linha
  `Full Square`, incluindo a auditoria de vieses geometricos herdados da fase
  anterior

## Contrato Experimental Congelado

Configuracao usada para todos os quick screens desta iteracao:

- protocolo reduzido: `configs/protocol_full_circle_sel4curr.json`
- `--train_once_eval_all`
- `--max_samples_per_exp 100000`
- `--max_val_samples_per_exp 20000`
- `--max_dist_samples 20000`
- `--stat_mode quick`
- `--stat_max_n 2000`
- `seq_bigru_residual` como familia base

Observacao metodologica principal:

- o primeiro shortlist Full Circle chegou a ser lancado incorretamente com full
  data, mas foi interrompido e substituido por um rerun correto com os caps
  acima

## Runs Canonicos Desta Branch

1. Importacao direta dos finalistas Full Square
   - run: `outputs/full_circle/e2_finalists_shortlist_100k/exp_20260416_014727`
   - campeao: `S27cov_sciv1_lr0p00015`
   - resultado: `4/12`
   - leitura: transferencia fraca; `tail98` ficou claramente negativa e `G2`
     apareceu como principal gargalo

2. Shortlist Full Circle orientado a `G2`
   - run: `outputs/full_circle/g2_shortlist_100k/exp_20260416_120619`
   - campeao: `S27cov_lc0p25_tail95_t0p03_disk_geom3`
   - resultado: `7/12`
   - leitura: alinhar a geometria com disco melhorou bastante sobre o shortlist
     herdado do Full Square

3. Follow-up ainda enviesado por geometria (`disk_geom3` + probes ortogonais)
   - split A: `outputs/full_circle/disk_bs8192_lat10_100k_split_a/exp_20260416_165643`
   - split B: `outputs/full_circle/disk_bs8192_lat10_100k_split_b/exp_20260416_165644`
   - melhor resultado: `S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192` com `8/12`
   - leitura: melhor evidencia operacional da branch, mas nao pode ser tratada
     como baseline cientifico neutro porque ainda usa `geom3` + `disk_l2`

4. Restart limpo Full Circle sem priors de borda/geometria
   - split A: `outputs/full_circle/20260416_182317_clean_bs8192_lat10_100k_split_a/exp_20260416_182319`
   - split B: `outputs/full_circle/20260416_182317_clean_bs8192_lat10_100k_split_b/exp_20260416_182319`
   - clean baseline: `S27cov_fc_clean_lc0p25_t0p03` -> `1/12`
   - clean lat10: `S27cov_fc_clean_lc0p25_t0p03_lat10` -> `4/12`

5. Confirmacao limpa posterior do candidato ainda aberto
   - split A: `outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_a/exp_20260417_115142`
   - split B: `outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_b/exp_20260417_115142`
   - clean bs8192: `S27cov_fc_clean_lc0p25_t0p03_bs8192` -> `2/12`
   - clean lat10 rerun: `S27cov_fc_clean_lc0p25_t0p03_lat10` -> `5/12`
   - leitura: o candidato limpo em aberto agora tem resposta direta e continua
     fraco; a linha limpa segue muito abaixo dos runs enviesados por geometria

6. Follow-up soft-radial sem `geom3` e sem `disk_l2`
   - bloco A: `outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256`
   - bloco B: `outputs/full_circle/20260420_233254_soft_radial_block_b_100k/exp_20260420_233257`
   - melhor geometry-light: `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0` -> `6/12`
   - campeao interno negativo do bloco B:
     `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_covsoft_lc0p20_t0p035` -> `0/12`
   - leitura: vies radial suave ajudou sem voltar para `geom3`/`disk_l2`, mas
     `covsoft` falhou sob validacao completa

7. Reruns diretos para fechar o shortlist soft-radial
   - `bs8192`: `outputs/full_circle/20260421_234722_soft_radial_resolve_bs8192_100k/exp_20260421_234723`
   - resultado `bs8192`:
     `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_bs8192` -> `6/12`
   - `tail98`: `outputs/full_circle/20260421_234722_soft_radial_resolve_tail98_100k/exp_20260421_234724`
   - resultado `tail98`:
     `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0_tail98` -> `1/12`
   - leitura: `bs8192` empatou com o melhor geometry-light e `tail98` ficou
     negativo

## Leitura Cientifica Atual

- o Full Circle nao pode herdar cegamente as hipoteses de borda/canto do Full
  Square
- `disk_geom3` e `disk_geom3_bs8192` mostraram que ha ganho quando a geometria
  e informada ao modelo, mas isso continua sendo evidencia secundaria
- quando `support_weight_mode`, `support_feature_mode` e
  `support_filter_mode` sao todos forçados para `none`, a linha perde muito
- portanto, o ganho anterior de `7/12` e `8/12` depende fortemente de priors de
  geometria e nao pode ser vendido como baseline limpo da aquisicao Full Circle
- o `clean_bs8192` ja foi confirmado e fechou em `2/12`; portanto a familia
  limpa atual le `1/12`, `2/12` e `5/12`, ainda bem abaixo da linha com priors
- existe agora uma classe intermediaria geometry-light:
  `soft_rinf_local` e `soft_rinf_local_bs8192`, ambos com `6/12`
- `tail98` e `covsoft` ficaram negativos e nao devem ser promovidos

## Estado Operacional Ao Encerrar Esta Branch

- o padrao de nome de saida com `RUN_STAMP` foi restaurado nos launchers novos
- o launcher de split limpo existe em:
  - `scripts/ops/train_full_circle_clean_bs8192_lat10_split.sh`
- os launchers relevantes desta iteracao sao:
  - `scripts/ops/train_full_circle_g2_shortlist.sh`
  - `scripts/ops/train_full_circle_disk_bs8192_lat10.sh`
  - `scripts/ops/train_full_circle_clean_bs8192_lat10.sh`
  - `scripts/ops/train_full_circle_clean_bs8192_lat10_split.sh`
  - `scripts/ops/train_full_circle_soft_radial_screen.sh`

## O Que Fazer Depois

Este workspace deve ser considerado documentado e estacionado.

Proximo movimento recomendado:

1. voltar para MDN em uma nova pasta e uma nova branch
2. nao continuar o trabalho MDN nesta worktree
3. se a linha Full Circle for reaberta no futuro, usar
   `soft_rinf_local` como referencia geometry-light atual
4. se ainda houver interesse em geometria apos isso, testar apenas vies radial
   suave e continuo, sem `cornerness_norm` e sem `disk_l2` hard filter, e
   apenas com retunes muito locais

## Documentos De Handoff

- `docs/active/WORKING_STATE.md`
- `docs/active/FULL_CIRCLE_EXECUTION_PLAN.md`
- `docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md`
