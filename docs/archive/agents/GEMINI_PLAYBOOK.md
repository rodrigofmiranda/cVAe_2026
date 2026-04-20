# GEMINI_PLAYBOOK.md

Playbook operacional para uma segunda IA trabalhar neste repo sem se perder.

## 1. Se a pergunta for "onde eu olho?"

Use esta tabela mental:

- entender o repo:
  - `README.md`
  - `PROJECT_STATUS.md`
  - `TRAINING_PLAN.md`
- entender protocolo e artefatos:
  - `docs/PROTOCOL.md`
- diagnosticar bug ou inconsistencia:
  - `docs/DIAGNOSTIC_CHECKLIST.md`
- escolher melhor grid de treino:
  - `outputs/<run>/tables/gridsearch_results.csv`
  - `outputs/<run>/logs/training_history.json`
  - `outputs/<run>/plots/best_grid_model/`
- julgar se o twin funciona cientificamente:
  - `outputs/exp_*/manifest.json`
  - `outputs/exp_*/tables/summary_by_regime.csv`
  - `outputs/exp_*/tables/stat_fidelity_by_regime.csv`

## 2. Se a pergunta for "qual arquivo decide a verdade?"

Ordem de confianca:

1. `manifest.json`
2. `summary_by_regime.csv`
3. `stat_fidelity_by_regime.csv`
4. `gridsearch_results.csv`
5. `training_history.json`
6. `state_run.json`

Se `state_run.json` contradizer `manifest.json` ou os CSVs, trate `state_run.json` como auxiliar.

## 3. Se o treino parece bom, mas o resultado cientifico parece ruim

Nao pare em:

- `loss`
- `val_loss`
- `best_epoch`

Vá para:

- `summary_by_regime.csv`

Leia obrigatoriamente:

- `delta_evm_%`
- `delta_cov_fro`
- `delta_kurt_l2`
- `stat_mmd_qval`
- `stat_energy_qval`
- `stat_psd_dist`
- `gate_g1` ... `gate_g6`
- `validation_status`

Regra:

- um treino "bonito" pode continuar sendo um twin ruim
- a decisao final e por regime, nao por loss

## 4. Se o cVAE perdeu para o baseline

Abra:

- `summary_by_regime.csv`

Compare:

- `baseline_*`
- `cvae_*`
- `better_than_baseline_cov`
- `better_than_baseline_kurt`
- `better_than_baseline_psd`

Interpretacao:

- se o baseline ainda vence em covariancia, kurtosis e PSD, nao ha evidencia suficiente para promover a arquitetura

## 5. Se voce suspeita de bug de pipeline

Nao chame de limitacao do modelo antes disso:

1. conferir branch e worktree
2. rodar suite de testes
3. rodar smoke de protocolo
4. rodar `scripts/verify_pipeline_fixes.py` sobre o ultimo `exp_*`
5. reler `docs/archive/reference/DIAGNOSTIC_CHECKLIST.md`

Checklist minima:

```bash
cd /home/rodrigo/cVAe_2026_mdn_return
python -m pytest tests -q
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --max_epochs 1 \
  --max_grids 1 \
  --max_experiments 1 \
  --max_samples_per_exp 2000
```

## 6. Se voce suspeita de colapso latente

Abra:

- `outputs/<run>/tables/gridsearch_results.csv`
- `outputs/<run>/models/grid_*__tag/plots/`
- `outputs/<run>/logs/training_history.json`

Leia:

- `active_dims`
- metricas de KL
- `var_mc_gen`
- curvas de treino
- plots de atividade latente

Sinais de alerta:

- `active_dims` muito baixo
- KL quase nulo por muitas epocas
- `var_mc_gen` muito distante da variancia real

## 7. Se voce quer comparar arquiteturas

Compare no mesmo enquadramento:

- mesma raiz de dataset
- mesma politica de split
- mesma familia de metricas
- mesma pergunta cientifica

Nao compare:

- um run de `src.training.train` somente por `val_loss`
com
- um run de `src.protocol.run` julgado por `summary_by_regime.csv`

Comparacao correta:

1. usar `src.training.train` para ver saude do treino e ranking interno
2. usar `src.protocol.run --train_once_eval_all` para a comparacao cientifica final

## 8. Se voce quer analisar um run em andamento

Para o run residual atual:

- run dir: `outputs/residual_small_global`
- log: `outputs/residual_small_global.launch.log`

Perguntas que voce pode responder do log:

- em qual grid esta
- em qual epoca esta
- se houve `ReduceLROnPlateau`
- se houve `NaN` ou crash

Perguntas que o log nao responde sozinho:

- se o twin e melhor que o baseline
- se houve melhora estatistica por regime

Para isso, espere os artefatos finais.

## 9. Se voce quer retomar um experimento anterior conhecido

Referencia da arquitetura anterior:

- `outputs/exp_20260313_153655`

Arquivos a abrir:

- `outputs/exp_20260313_153655/manifest.json`
- `outputs/exp_20260313_153655/global_model/tables/gridsearch_results.csv`
- `outputs/exp_20260313_153655/global_model/logs/training_history.json`
- `outputs/exp_20260313_153655/tables/summary_by_regime.csv`
- `outputs/exp_20260313_153655/tables/stat_fidelity_by_regime.csv`

Licao principal desse experimento:

- o `concat` treinou estavel
- mas falhou cientificamente como modelo global

## 10. Se houver conflito entre "parece bug" e "parece limite do modelo"

Use esta regra:

- se smoke, testes e checklist falham: trate como problema operacional
- se smoke, testes e checklist passam e `G3` a `G6` continuam ruins: trate como limite do modelo ou do desenho experimental

## 11. Perguntas que uma IA deve sempre fazer antes de concluir

- estou olhando treino exploratorio ou validacao cientifica?
- estou usando os artefatos corretos para essa pergunta?
- comparei com o baseline?
- olhei por regime ou so no agregado?
- estou inferindo demais a partir de `loss`?
