# TRAINING_PLAN.md - Plano Cientifico Vivo do Digital Twin VLC

> Versao ativa em 2026-03-24.
> Este documento e a fonte de verdade para validacao cientifica, diagnostico e criterios de aceite.

## Branch experimental atual

- `feat/conditional-flow-decoder`

Nesta branch, a prioridade imediata e:

- manter a instrumentacao residual ja pronta
- parar de abrir novos sweeps apenas de MDN
- implementar um decoder flow condicional sobre o residual
- testar primeiro uma prova curta e estrutural antes de qualquer novo grid

## 0. Estado atual

- Pipeline canonico ativo:
  - `python -m src.protocol.run`
- Modos ativos de protocolo:
  - `per_regime_retrain`: diagnostico local com treino separado por regime
  - `train_once_eval_all`: treino global unico + avaliacao por regime sem retreino
- Tabela canonica de validacao:
  - `outputs/exp_*/tables/summary_by_regime.csv`
- Ranking canônico por candidato avaliado:
  - `outputs/exp_*/tables/protocol_leaderboard.csv`
- Artefatos secundarios:
  - `outputs/exp_*/tables/stat_fidelity_by_regime.csv`
  - `outputs/exp_*/tables/residual_signature_by_regime.csv`
  - `outputs/exp_*/tables/residual_signature_by_amplitude_bin.csv`
  - `outputs/exp_*/tables/train_regime_diagnostics_history.csv`
  - JSONs e logs por regime
- Diagnosticos de grid disponiveis:
  - `tables/gridsearch_results.csv`
  - `plots/champion/analysis_dashboard.png`
- Ultima checagem local da suite:
  - `python -m pytest tests -q` -> `46 passed`
- Run pivo de referencia:
  - `exp_20260324_023558`
  - campeao:
    - `S6seq_W7_h64_lat4_b0p003_lmmd1p75_fb0p10_lr0p0003_L128-256-512`
  - resultado:
    - `10/12` passes
    - `2/4` em `0.8 m`
    - `4/4` em `1.0 m`
    - `4/4` em `1.5 m`
  - leitura atual:
    - `lambda_mmd=1.75` virou a melhor direcao recente
    - o gargalo principal segue concentrado em `0.8 m / 100 mA` e `0.8 m / 300 mA`
    - o aumento de capacidade para `h96` continua sem justificativa cientifica
- Overnight amplo que gerou a referencia atual:
  - `grid_preset=seq_overnight_12h`
  - leitura:
    - foi o sweep que elevou a referencia para `10/12`
    - confirmou `W7_h64` como familia lider
    - nao e mais o proximo passo imediato
- Proximo replay curto para ler as metricas novas por eixo:
  - `grid_preset=seq_replay_axis_diagnostics`
  - foco:
    - rerodar so os melhores candidatos ja observados
    - comparar `W7_h64` vs `W7_h96` vs `W9_h96`
    - manter a referencia historica `S2seq_W7_h64_b0p001_lmmd1p0`
    - usar os novos campos por eixo em `summary_by_regime.csv`
  - leitura apos execucao:
    - `exp_20260324_024442` terminou com `8/12`
    - confirmou `W7_h64` como familia lider
    - mostrou que o gargalo residual continua localizado em `0.8 m`
- Proximo grid assertivo de acabamento:
  - `grid_preset=seq_finish_0p8m`
  - foco:
    - manter `W7_h64` fixo
    - usar `lmmd=1.75` como controle
    - testar `lmmd=2.0`
    - incluir so hedges de `beta/lr` ja fortes no overnight
    - atacar apenas `0.8 m / 100 mA` e `0.8 m / 300 mA`
- Preset causal novo da branch experimental:
  - `grid_preset=seq_sampled_mmd_compare`
  - foco:
    - manter a familia `W7_h64_lat4_b0.003`
    - isolar o efeito de `mmd_mode=sampled_residual`
    - usar a nova instrumentacao para decidir a proxima objective
  - execucao recomendada:
    - usar `--no_baseline`
    - o baseline nao faz parte da decisao causal desta branch
    - evitar gastar tempo de protocolo com um comparador que nao entra na decisao atual
- Preset conservador novo para o retry de MDN:
  - `grid_preset=seq_mdn_conservative_proof`
  - foco:
    - manter apenas `mdn3`
    - remover `PSD loss`
    - reduzir `lambda_axis` para `0.01`
    - baixar `lr` para `2e-4`
    - testar um candidato sem `MMD` e um com `lambda_mmd=0.25`
  - motivacao:
    - `seq_mdn_proof` inflou fortemente a variancia do decoder e falhou `12/12`
    - o retry deve ser estruturalmente mais conservador antes de abrir novos knobs
- Preset exploratorio novo para MDN:
  - `grid_preset=seq_mdn_exploratory_quick`
  - foco:
    - manter a linha estavel `mdn3 + lr=2e-4 + axis=0.01 + lmmd=0.25`
    - explorar apenas knobs com evidência diagnostica:
      - `lambda_mmd=0.50`
      - `lambda_axis=0.02`
      - `beta=0.002`
      - `lr=1.5e-4`
      - `mdn4`
    - continuar sem `PSD loss`
  - modo de execucao recomendado:
    - manter os `12` regimes
    - usar quick por cap de amostras, nao por reduzir regimes
    - quick real deve capar treino, validacao e testes estatisticos
    - comando recomendado:
      - `--max_samples_per_exp 100000`
      - `--max_val_samples_per_exp 20000`
      - `--max_dist_samples 20000`
      - `--stat_mode quick`
      - `--stat_max_n 2000`
- Preset exploratorio novo focado em `G5`:
  - `grid_preset=seq_mdn_g5_exploratory_quick`
  - ancora:
    - melhor MDN quick atual:
      - `mdn3`
      - `beta=0.002`
      - `free_bits=0.10`
      - `lr=2e-4`
      - `lambda_axis=0.01`
      - `lambda_mmd=0.25`
  - foco:
    - manter `12` regimes e quick real por cap de amostras
    - atacar apenas as falhas remanescentes de `G5` em `0.8 m`
    - explorar:
      - `beta=0.0015`
      - `free_bits=0.05`
      - `lambda_axis=0.005`
      - `lambda_mmd=0.35`
      - `mdn2`
  - comando recomendado:
    - `--max_samples_per_exp 100000`
    - `--max_val_samples_per_exp 20000`
    - `--max_dist_samples 20000`
    - `--stat_mode quick`
    - `--stat_max_n 2000`
- Preset exploratorio maior focado em `G5`:
  - `grid_preset=seq_mdn_g5_broader_quick`
  - ancora:
    - manter o controle `S14` vencedor (`beta=0.002`, `lmmd=0.25`, `axis=0.01`)
  - foco:
    - explorar uma faixa intermediaria de `beta`:
      - `0.0020`
      - `0.0018`
      - `0.0015`
    - aumentar a ancora distribucional com:
      - `lambda_mmd=0.35`
      - `lambda_mmd=0.50`
    - so relaxar `lambda_axis` para `0.005` na linha mais agressiva de `beta=0.0015`
  - motivacao:
    - `beta=0.0015` sozinho ajudou train-side, mas regrediu `G6`
    - o proximo passo deve ser testar se `MMD` mais forte recupera `G6` sem abrir mao do ganho em `G5`
  - comando recomendado:
    - `--max_samples_per_exp 100000`
    - `--max_val_samples_per_exp 20000`
    - `--max_dist_samples 20000`
    - `--stat_mode quick`
    - `--stat_max_n 2000`
- Proximo plano estrutural da branch:
  - documento canonico:
    - `docs/FLOW_DECODER_PLAN.md`
  - direcao:
    - manter `seq_bigru_residual`
    - substituir a likelihood do decoder por um flow condicional sobre
      `Delta = Y - X`
  - objetivo:
    - atacar a forma da distribuicao do residuo em `0.8 m`
    - sem depender de mais sweeps de mistura
  - regra de execucao:
    - primeiro smoke e prova unica
    - so depois grid curto
    - so depois protocolo full
  - status atual:
    - Phase 1 implementada
    - primeira familia escolhida:
      - flow condicional `sinh-arcsinh` por eixo
    - preset novo:
      - `seq_flow_smoke`
    - primeiro smoke estrutural:
      - `exp_20260326_033237`
    - leitura:
      - build / save / reload / inference / protocolo fecharam
      - o modelo ainda esta cientificamente fraco
      - o proximo passo nao e sweep grande; e uma prova curta real da linha flow
  - preset novo da Phase 2:
    - `grid_preset=seq_flow_proof_quick`
  - desenho:
    - manter `12` regimes
    - manter `W7_h64_lat4`
    - usar `decoder_distribution=flow`
    - usar a objective plain flow:
      - `lambda_mmd=0.0`
      - `lambda_axis=0.0`
      - `lambda_psd=0.0`
    - quick real por cap de treino, validacao e testes
  - comando recomendado:
    - `--max_samples_per_exp 100000`
    - `--max_val_samples_per_exp 20000`
    - `--max_dist_samples 20000`
    - `--stat_mode quick`
    - `--stat_max_n 2000`

### Criterio operacional do teste causal `seq_sampled_mmd_compare`

Objetivo:

- verificar se `mmd_mode=sampled_residual` melhora a fidelidade da distribuicao
  residual perto de `0.8 m` sem degradar os regimes `1.0 m` e `1.5 m`

Arquivos que devem ser lidos nesta ordem:

1. `tables/protocol_leaderboard.csv`
2. `tables/summary_by_regime.csv`
3. `tables/residual_signature_by_regime.csv`
4. `tables/residual_signature_by_amplitude_bin.csv`
5. `plots/best_model/residual_signature_overview.png`
6. `eval/dist_0p8m__curr_100mA/plots/champion/analysis_dashboard.png`
7. `eval/dist_0p8m__curr_300mA/plots/champion/analysis_dashboard.png`

Criterio de sucesso minimo:

- `sampled_residual` nao piora `1.0 m` e `1.5 m`
- `0.8m / 100mA` e `0.8m / 300mA` melhoram em pelo menos dois destes sinais:
  - `stat_mmd_qval`
  - `stat_energy_qval`
  - `delta_wasserstein_I/Q`
  - `delta_jb_stat_rel_I/Q`
- os histogramas do residual deixam de ficar claramente estreitos demais

Leitura de falha:

- se `sampled_residual` nao melhorar `0.8 m`
- ou se abrir degradacao clara em `1.0 m` / `1.5 m`
- ou se os ganhos forem so train-side e nao aparecerem em `summary_by_regime.csv`

Decisao seguinte:

- se o teste for promissor:
  - manter `sampled_residual` e abrir um micro-grid curto de refinamento
- se o teste falhar:
  - parar aqui na objective atual e partir para penalizacao marginal por eixo

## 1. Objetivo

Demonstrar, por regime `(d, I)`, que o digital twin baseado em cVAE reproduz a distribuicao condicional do canal VLC:

`p(y | x, d, I)`

O alvo cientifico nao e apenas boa predicao media, e sim fidelidade distribucional do residuo do canal:

`Delta = Y - X`

### Decisao de arquitetura para o objetivo final

O artefato final desejado para a tese e para a etapa futura de treinamento do
autoencoder transmissor-receptor e:

- um unico modelo global, condicional, estocastico e diferenciavel
- treinado com todo o dataset
- condicionado por `x`, `d` e `I`
- avaliado por regime sem retreinar

Modelos treinados por regime continuam uteis para:

- diagnostico
- comparacao local com o baseline
- ablacoes

Mas nao sao o twin final-alvo.

## 2. Invariantes obrigatorios do pipeline

Estes pontos nao devem ser relaxados nos experimentos:

1. Split por experimento, temporal: `head=train`, `tail=val`, `80/20`
2. Ordem obrigatoria: `split -> cap/reduce(train only) -> treino`
3. `shuffle_train_batches = True`
4. Clamp do decoder calibrado: `[-5.82, -0.69]`
5. Metricas distribucionais sobre `Delta = Y - X`, nao sobre `Y` bruto
6. Inferencia para metricas de distribuicao via `MC-concat`, nao MAP
7. `summary_by_regime.csv` e a tabela canonica de validacao final
8. O diagnostico nao pode olhar apenas o prior em treino; ele deve sempre comparar:
   - treino/latente/prior
   - modelo final vs real
   - modelo final vs baseline
   - campeao vs demais grids testados
9. Para decisoes do twin final, priorizar o modo `train_once_eval_all`:
   - treinar uma vez
   - avaliar todos os regimes com o mesmo modelo
   - usar `per_regime_retrain` apenas como apoio diagnostico

## 3. Protocolo de inferencia por familia de metricas

| Familia | Inferencia canonica | Uso |
|---|---|---|
| `EVM`, `SNR` | `det` / MAP | Metricas de ponto e comparacao operacional |
| `Delta mean/cov/skew/kurt`, `PSD`, `JB` | `mc_concat` | Fidelidade distribucional do residuo |
| `MMD^2`, `Energy`, `PSD distance + IC` | `mc_concat` | Teste formal de duas amostras |
| Ranking rapido do grid | `mc_concat` na parte distribucional | Escolha interna de configuracoes |

Regra pratica: `det` serve para metrica operacional; `mc_concat` serve para dizer se o twin reproduz a distribuicao.

## 4. Metricas ativas

### 4.1 Nivel 1 - Fidelidade fisica

- `evm_real_%`, `evm_pred_%`, `delta_evm_%`
- `snr_real_db`, `snr_pred_db`, `delta_snr_db`

### 4.2 Nivel 2 - Fidelidade distribucional do residuo

- `delta_mean_l2`
- `delta_cov_fro`
- `var_real_delta`, `var_pred_delta`, `var_ratio_pred_real`
- `delta_skew_l2`
- `delta_kurt_l2`
- `delta_psd_l2`
- `jb_p_min`, `jb_log10p_min`
- `jb_real_p_min`, `jb_real_log10p_min`
- `delta_jb_log10p`

### 4.3 Nivel 3 - Testes formais de duas amostras

- `stat_mmd2`, `stat_mmd_pval`, `stat_mmd_qval`, `stat_mmd_bandwidth`
- `stat_mmd2_normalized`
- `stat_energy`, `stat_energy_pval`, `stat_energy_qval`
- `stat_psd_dist`, `stat_psd_ci_low`, `stat_psd_ci_high`

### 4.4 Diagnostico de treino, latente e prior

Estas metricas nao sao gates finais, mas sao obrigatorias no diagnostico:

- `score_v2`
- `active_dims`
- `kl_q_to_p_total`
- `kl_p_to_N_total`
- `var_mc_gen`
- `training_history.json`:
  - `loss`
  - `val_loss`
  - `val_recon_loss`
- plots do grid e do campeao:
  - overlays de constelacao
  - overlay do residuo
  - PSD do residuo
  - atividade latente
  - KL por dimensao
  - historico de treino

### 4.5 Comparacao baseline vs cVAE

- `baseline_*`
- `cvae_*`
- `better_than_baseline_cov`
- `better_than_baseline_kurt`
- `better_than_baseline_psd`

Regra importante: `score_v2` ajuda a ordenar grids, mas nao substitui a leitura cientifica final em `summary_by_regime.csv`.

## 5. Gates G1-G6

Os gates sao criterios de aceite por regime. Eles ja aparecem em `summary_by_regime.csv` como `gate_g1` ... `gate_g6`.

| Gate | Regra atual |
|---|---|
| `G1` | `abs(delta_evm_%) / abs(evm_real_%) < 0.10` |
| `G2` | `abs(delta_snr_db) / abs(snr_real_db) < 0.10` |
| `G3` | `cvae_mean_rel_sigma < 0.10` **e** `cvae_cov_rel_var < 0.20` |
| `G4` | `cvae_psd_l2 < 0.25` |
| `G5` | `cvae_delta_skew_l2 < 0.30` **e** `cvae_delta_kurt_l2 < 1.25` **e** `delta_jb_stat_rel < 0.20` |
| `G6` | `stat_mmd_qval > 0.05` **e** `stat_energy_qval > 0.05` |

### Observacoes criticas

- `G1` e `G2` agora medem fidelidade direta do cVAE ao canal real.
- `baseline_*` continua no CSV para benchmark contra AWGN/modelo tradicional, mas nao decide `validation_status`.
- `G3` usa escala fisica do proprio canal: media em unidades de `sigma_real` e covariancia em unidades de `var_real_delta`.
- `G4` usa `PSD_L2`, que ja e uma distancia normalizada entre os log-PSD do real e do modelo.
- `G5` cobre forma de distribuicao: skew, kurtosis e nao-gaussianidade, sempre contra o real.
- `G6` e o gate formal mais forte para indistinguibilidade de distribuicao.

### Nota futura — Mutual Information

- `mutual information` **nao entra nos gates por enquanto**
- se entrar depois, deve ser adicionada como diagnostico auxiliar:
  - `MI_real = I(X; Y_real)`
  - `MI_cvae = I(X; Y_model)`
  - `mi_gap_rel = |MI_cvae - MI_real| / MI_real`
- usar a mesma rotina de estimacao, o mesmo `N` e bootstrap/IC
- MI so faz sentido como complemento; ela nao substitui `MMD`, `Energy`, `PSD` e as metricas do residual

### Status derivado por regime

- `validation_status = pass` se todos os gates disponiveis forem `True`
- `validation_status = fail` se algum gate disponivel for `False`
- `validation_status = partial` se houver gates indisponiveis

## 6. Diagnostico novo, pensado do zero

Premissa: nenhuma conclusao deve ser tirada olhando apenas o prior em treino. O diagnostico completo precisa fechar o ciclo:

1. `pipeline -> treino/latente -> modelo final -> comparacao vs baseline -> decisao`

### 6.1 Plano A - Sanidade do pipeline

Pergunta: a execucao esta medindo a coisa certa?

Checar:

- `python scripts/verify_pipeline_fixes.py`
- split por experimento e ordem `split -> reduce(train only)`
- clamp correto
- `MC-concat` usado nas metricas distribucionais
- `summary_by_regime.csv` preenchido com colunas `stat_*`, `baseline_*`, `cvae_*`

Se falhar aqui, nao interpretar nenhum resultado como limitacao do modelo.

### 6.2 Plano B - Treino, latente e prior

Pergunta: o modelo aprendeu um latente util ou colapsou?

Checar, por grid:

- `active_dims`
- `kl_q_to_p_total`
- `kl_p_to_N_total`
- `var_mc_gen`
- curvas em `logs/training_history.json`
- plots em `models/grid_*__tag/plots/`

Interpretacao:

- `active_dims` muito baixo sugere colapso parcial
- `KL` muito baixo por muitas epocas sugere latente inutil
- `var_mc_gen` muito distante de `var_real_delta` sugere marginal mal calibrada
- historico de treino ruim indica instabilidade ou overfit

### 6.3 Plano C - Modelo final vs dados reais

Pergunta: o campeao reproduz de fato o canal real?

Checar no run final:

- `EVM`, `SNR`
- `delta_mean_l2`, `delta_cov_fro`, `delta_skew_l2`, `delta_kurt_l2`, `delta_psd_l2`
- `jb_log10p_min`, `jb_real_log10p_min`, `delta_jb_log10p`
- `stat_mmd2`, `stat_mmd_qval`
- `stat_energy`, `stat_energy_qval`
- `stat_psd_dist`, `stat_psd_ci_low`, `stat_psd_ci_high`
- plots em `plots/best_grid_model/`

Regra critica: o diagnostico principal do twin e no modelo final, nao no prior isolado.

### 6.4 Plano D - Modelo final vs baseline

Pergunta: o cVAE esta entregando algo melhor do que um baseline deterministico forte?

Checar:

- `baseline_*` vs `cvae_*`
- `better_than_baseline_cov`
- `better_than_baseline_kurt`
- `better_than_baseline_psd`
- gates `G3` e `G4`

Se o cVAE perde para o baseline nas metricas de distribuicao, ainda nao ha evidencia para escalar o protocolo.

### 6.5 Plano E - Campeao vs todos os grids testados

Pergunta: o campeao venceu por criterio robusto ou por acaso do ranking?

Checar:

- `tables/gridsearch_results.csv`
- top-N por `score_v2`
- dispersao entre `score_v2`, `delta_cov_fro`, `delta_kurt_l2`, `active_dims`
- plots agregados do grid
- bundles de plots de cada candidato

Regra critica: antes de propor mudanca estrutural, verificar se algum grid nao-campeao ja sinaliza melhora em kurtosis, PSD ou MMD.

## 7. Sequencia minima de diagnostico no protocolo reduzido

Usar no minimo o protocolo reduzido `0.8/1.0/1.5 m x 100/300/500/700 mA`.

1. Testes unitarios

```bash
python -m pytest tests -q
```

2. Smoke de protocolo reduzido

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --max_epochs 1 --max_grids 1 --max_experiments 1 --max_samples_per_exp 2000
```

3. Run cientifico do protocolo reduzido com testes estatisticos

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --max_grids 5 \
  --stat_tests --stat_mode full --stat_max_n 10000 --stat_n_perm 1000
```

4. Verificacao de pipeline

```bash
python scripts/verify_pipeline_fixes.py outputs/exp_YYYYMMDD_HHMMSS
```

5. Verificacao de kurtosis no modelo final

```bash
python scripts/check_kurt_pred.py \
  outputs/exp_YYYYMMDD_HHMMSS/studies/within_regime/regimes/dist_1m__curr_300mA \
  --n_eval 5000 --mc_samples 16
```

6. Leitura obrigatoria dos artefatos

- `tables/summary_by_regime.csv`
- `tables/gridsearch_results.csv`
- `models/grid_*__tag/plots/`
- `plots/best_grid_model/`

Roteiro operacional:

- [docs/DIAGNOSTIC_CHECKLIST.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/DIAGNOSTIC_CHECKLIST.md)

## 8. Leitura do run pivo de referencia

Run ancora atual: `exp_20260312_171333`

Sinais observados:

- `baseline_evm_pred_% ~= 26.05`
- `delta_evm_% ~= -0.22`
- `baseline_delta_cov_fro ~= 0.01203`
- `cvae_delta_cov_fro ~= 0.01946`
- `baseline_delta_kurt_l2 ~= 0.783`
- `cvae_delta_kurt_l2 ~= 4.857`
- `cvae_jb_log10p_min ~= -50955.85`
- `stat_mmd_qval ~= 0.001`

Leitura atual:

- o pipeline esta medindo corretamente
- o modelo final ainda nao reproduz bem a distribuicao do canal
- a proxima rodada precisa diagnosticar treino e modelo final juntos

## 9. Matriz de decisao: novo grid search vs mudanca estrutural

### Rodar novo grid search quando:

- `active_dims` esta baixo ou instavel
- `KL` sugere colapso ou regularizacao desequilibrada
- existe variacao relevante entre grids em `delta_cov_fro`, `delta_kurt_l2` ou `stat_mmd_qval`
- algum grid nao-campeao melhora sinais de distribuicao mesmo sem vencer em `score_v2`

Familias de hiperparametros prioritarias:

- `beta`
- `free_bits`
- `latent_dim`
- `layer_sizes`
- `dropout`
- `lr`

### Considerar mudanca estrutural quando:

- pipeline ja foi verificado
- treino e latente parecem saudaveis
- o modelo final continua perdendo para o baseline em `G3` e `G4`
- `G5` e `G6` continuam falhando de forma sistematica apos varrer grids plausiveis
- nenhum grid melhora o sinal de kurtosis, PSD ou MMD de forma consistente

Mudancas estruturais candidatas so entram depois disso:

- prior mais expressivo
- decoder mais expressivo
- familia gerativa diferente para regimes mais nao-lineares

## 10. Etapas cientificas

| Etapa | Objetivo | Status |
|---|---|---|
| A | Infraestrutura e smoke | completa |
| B | Protocolo reduzido de 12 regimes + diagnostico completo | em andamento |
| C | Escala incremental para o full 27-regime | aguardando B |
| D | Generalizacao / hold-out | aguardando C |
| E | Selecao do twin vencedor | aguardando D |

## 11. Documentos complementares

- [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
- [docs/DIAGNOSTIC_CHECKLIST.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/DIAGNOSTIC_CHECKLIST.md)
- [docs/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/PROTOCOL.md)
- [docs/MODELING_ASSUMPTIONS.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/MODELING_ASSUMPTIONS.md)
