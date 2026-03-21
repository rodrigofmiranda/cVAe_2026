# GEMINI_BOOTSTRAP.md

Bootstrap curto para entregar contexto a outra IA sem despejar todo o historico.

## 1. Snapshot desta sessao

Atualize estes campos antes de reutilizar o arquivo:

- repo: `/workspace/2026/feat_delta_residual_adv`
- branch: `feat/channel-residual-architecture`
- commit: `eaef7f0`
- worktree: limpa
- dataset principal: `data/dataset_fullsquare_organized`
- output root: `outputs`
- gpu tensorflow: 1 GPU visivel
- linha de trabalho atual: arquitetura residual `channel_residual`
- run residual ativo nesta sessao: `outputs/residual_small_global`
- log do run residual ativo: `outputs/residual_small_global.launch.log`

## 2. Leitura inicial obrigatoria

Leia nesta ordem:

1. `PROJECT_STATUS.md`
2. `TRAINING_PLAN.md`
3. `README.md`
4. `docs/PROTOCOL.md`
5. `docs/DIAGNOSTIC_CHECKLIST.md`

## 3. Objetivo do projeto

Aprender um digital twin probabilistico do canal VLC:

- alvo: `p(y | x, d, I)`
- `x`: IQ transmitido
- `y`: IQ recebido
- `d`: distancia
- `I`: corrente

O objetivo final nao e um modelo por regime. O alvo cientifico da tese e:

- um unico modelo global
- condicional
- estocastico
- diferenciavel
- treinado uma vez
- avaliado por regime sem retreino

## 4. Entry points canonicos

- `python -m src.training.train`
  - usar para grid search e treino exploratorio do cVAE
- `python -m src.protocol.run`
  - usar para validacao cientifica completa
  - baseline vs cVAE
  - avaliacao por regime
  - testes estatisticos
- `python -m src.evaluation.evaluate`
  - usar para avaliacao isolada quando o modelo ja existe

## 5. Modos do protocolo

- `per_regime_retrain`
  - treino separado por regime
  - usar para diagnostico local
- `train_once_eval_all`
  - treina um modelo global unico
  - avalia esse mesmo modelo em todos os regimes
  - este e o modo alinhado ao objetivo final

## 6. Fontes de verdade

Se houver conflito entre arquivos, confiar nesta ordem:

1. `outputs/exp_*/manifest.json`
2. `outputs/exp_*/tables/summary_by_regime.csv`
3. `outputs/exp_*/tables/stat_fidelity_by_regime.csv`
4. `outputs/*/tables/gridsearch_results.csv`
5. `outputs/*/logs/training_history.json`
6. `state_run.json` como metadado auxiliar

## 7. Invariantes que nao devem ser quebradas

- split temporal por experimento: `head=train`, `tail=val`, `80/20`
- ordem do pipeline: `split -> cap/reduce(train only) -> treino`
- metricas distribucionais calculadas sobre `Delta = Y - X`
- inferencia distribuicional via `MC-concat`, nao MAP
- `summary_by_regime.csv` e a tabela canonica final
- para o twin final, priorizar `train_once_eval_all`

## 8. Estado cientifico atual

- o refactor estrutural principal esta concluido
- o foco atual e validacao cientifica do twin
- a arquitetura anterior canonicamente usada e `concat`
- este branch introduz uma ablacao estrutural nova: `channel_residual`
- a pergunta atual e se a arquitetura residual reduz o vies por regime visto no modelo `concat`

## 9. Resultado anterior que serve de referencia

Experimento de referencia da arquitetura anterior:

- `outputs/exp_20260313_153655`

Leitura resumida:

- melhor grid: `G0_lat4_b0p003_fb0p10_lr0p0003_L128-256-512`
- arquitetura: `concat`
- treino estavel, mas resultado cientifico fraco
- `pass_mmd_qval = 0/27`
- `pass_energy_qval = 0/27`
- `pass_both_qval = 0/27`
- `validation_status = fail` em todos os 27 regimes

Interpretacao:

- em `0.8 m` o modelo degrada forte
- em `1.0 m` ele fica menos ruim
- em `1.5 m` o PSD melhora, mas o vies por regime continua

## 10. Experimentos residuais relevantes

- smoke validado: `outputs/residual_small_smoke`
- run principal desta sessao: `outputs/residual_small_global`

Comando principal:

```bash
python -m src.training.train \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --run_id residual_small_global \
  --no_data_reduction \
  --grid_preset residual_small \
  --max_epochs 120 \
  --patience 12 \
  --reduce_lr_patience 6
```

## 11. Primeiras acoes recomendadas para uma nova IA

1. Ler `PROJECT_STATUS.md` e `TRAINING_PLAN.md`
2. Confirmar branch, commit e estado do run atual
3. Identificar se a tarefa e de treino exploratorio ou validacao cientifica
4. Escolher os artefatos certos para a pergunta feita
5. Nunca concluir apenas com base em `loss` ou `val_loss`
