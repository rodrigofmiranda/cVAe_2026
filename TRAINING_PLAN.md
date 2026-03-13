# TRAINING_PLAN.md - Plano Cientifico Vivo do Digital Twin VLC

> Versao ativa em 2026-03-13.
> Este documento e a fonte de verdade para validacao cientifica, diagnostico e criterios de aceite.

## 0. Estado atual

- Pipeline canonico ativo:
  - `python -m src.training.train`
  - `python -m src.evaluation.evaluate`
  - `python -m src.protocol.run`
- Modos ativos de protocolo:
  - `per_regime_retrain`: diagnostico local com treino separado por regime
  - `train_once_eval_all`: treino global unico + avaliacao por regime sem retreino
- Tabela canonica de validacao:
  - `outputs/exp_*/tables/summary_by_regime.csv`
- Artefatos secundarios:
  - `outputs/exp_*/tables/stat_fidelity_by_regime.csv`
  - JSONs e logs por regime
- Diagnosticos de grid disponiveis:
  - `tables/gridsearch_results.csv`
  - `models/grid_*__tag/plots/`
  - `plots/best_grid_model/`
- Ultima checagem local da suite:
  - `python -m pytest tests -q` -> `46 passed`
- Run pivo de referencia:
  - `exp_20260312_171333`
  - `G1` e `G2` passaram
  - `G3` a `G6` falharam
  - leitura atual: desempenho operacional aceitavel, fidelidade distribucional ainda insuficiente

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
| `G1` | `baseline_evm_pred_% < 30` |
| `G2` | `abs(delta_evm_%) < 15` |
| `G3` | `cvae_delta_cov_fro < baseline_delta_cov_fro` |
| `G4` | `cvae_delta_kurt_l2 < baseline_delta_kurt_l2` |
| `G5` | `abs(cvae_jb_log10p_min - jb_real_log10p_min) < 1.0` |
| `G6` | `stat_mmd_qval > 0.05` |

### Observacoes criticas

- `G4` mede melhora sobre o baseline, nao "kurtosis zero".
- `G5` deve ser lido em `log10(p)`, nao em `p-value` bruto.
- `G6` e o gate formal mais forte para indistinguibilidade de distribuicao.

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

## 7. Sequencia minima de diagnostico no regime pivo

Usar `1.0m / 300mA` como prova inicial de regime unico.

1. Testes unitarios

```bash
python -m pytest tests -q
```

2. Smoke de protocolo

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA.json \
  --max_epochs 1 --max_grids 1 --max_experiments 1 --max_samples_per_exp 2000
```

3. Run cientifico de regime unico com testes estatisticos

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA.json \
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

- [docs/DIAGNOSTIC_CHECKLIST.md](/workspace/2026/docs/DIAGNOSTIC_CHECKLIST.md)

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
| B | Prova em regime unico + diagnostico completo | em andamento |
| C | Escala incremental em multiplos regimes | aguardando B |
| D | Generalizacao / hold-out | aguardando C |
| E | Selecao do twin vencedor | aguardando D |

## 11. Documentos complementares

- [PROJECT_STATUS.md](/workspace/2026/PROJECT_STATUS.md)
- [docs/DIAGNOSTIC_CHECKLIST.md](/workspace/2026/docs/DIAGNOSTIC_CHECKLIST.md)
- [docs/PROTOCOL.md](/workspace/2026/docs/PROTOCOL.md)
- [docs/MODELING_ASSUMPTIONS.md](/workspace/2026/docs/MODELING_ASSUMPTIONS.md)
