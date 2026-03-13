# PROJECT_STATUS.md — Estado Atual do Repositório

> Atualizado em 2026-03-13.

## 1. Estado técnico

O refactor de engenharia está concluído no caminho ativo.

### Entry points canônicos

- `python -m src.training.train`
- `python -m src.evaluation.evaluate`
- `python -m src.protocol.run`

### Modos do protocolo

O protocolo agora tem dois modos explícitos:

- `per_regime_retrain`:
  - padrão quando `--train_once_eval_all` não é usado
  - treina um cVAE por regime
  - serve para diagnóstico e comparação local contra o baseline
- `train_once_eval_all`:
  - ativado com `--train_once_eval_all`
  - treina um único cVAE global em `outputs/exp_.../global_model`
  - avalia esse mesmo modelo em todos os regimes, sem retreino por regime
  - este é o modo alinhado ao objetivo final do digital twin universal

### Artefatos canônicos

- `state_run.json`: snapshot de um run individual
- `manifest.json`: snapshot consolidado de um experimento do protocolo
- `tables/summary_by_regime.csv`: tabela canônica de validação por regime
- `tables/stat_fidelity_by_regime.csv`: projeção derivada das métricas estatísticas

### Métricas consolidadas

`summary_by_regime.csv` é a fonte única de verdade para:

- fidelidade física: `EVM`, `SNR`
- fidelidade distribucional do resíduo `Δ = Y - X`
- comparação `baseline` vs `cVAE`
- testes formais `MMD`, `Energy`, `PSD`
- gates `G1`–`G6`
- `validation_status`

### Grid search e plots

O grid search gera:

- bundle de plots por modelo testado em `models/grid_*/plots/`
- bundle do campeão em `plots/best_grid_model/`
- plots agregados do ranking em `plots/gridsearch/`

O campeão também recebe um conjunto executivo no estilo do run legado:

- `analise_completa_vae.png`
- `comparacao_metricas_principais.png`
- `radar_comparativo.png`
- `constellation_overlay.png`

## 2. Estado científico

O foco atual não é mais refactor estrutural, e sim validação científica do twin.

### Pergunta principal

O cVAE heteroscedástico com prior Gaussiano é suficiente para reproduzir a
distribuição do canal VLC por regime?

### Direção do modelo final

Para a etapa final da tese, o twin alvo não é um banco de modelos por regime.
O alvo é um **modelo global, condicional, estocástico e diferenciável**
`p(y | x, d, I)`, treinado uma vez com todo o dataset e depois validado por
regime físico.

### Interpretação atual

- bugs de pipeline conhecidos foram corrigidos
- métricas e gates já estão automatizados
- falhas em `G3`–`G6`, quando persistem após os fixes, devem ser tratadas como
  possível limitação do modelo e não como bug operacional

## 3. Documentos ativos

- [README.md](/workspace/2026/README.md): visão geral e uso
- [PROJECT_STATUS.md](/workspace/2026/PROJECT_STATUS.md): estado atual do código
- [TRAINING_PLAN.md](/workspace/2026/TRAINING_PLAN.md): plano científico e gates
- [docs/PROTOCOL.md](/workspace/2026/docs/PROTOCOL.md): protocolo, artefatos e CLI
- [docs/MODELING_ASSUMPTIONS.md](/workspace/2026/docs/MODELING_ASSUMPTIONS.md): premissas do modelo

## 4. Documentos arquivados

- [docs/archive/REFACTOR_PLAN_legacy.md](/workspace/2026/docs/archive/REFACTOR_PLAN_legacy.md)

## 5. Como retomar rapidamente

```bash
cd /workspace/2026
git status -sb
git log --oneline -5
python -m pytest tests -q
```

Depois disso:

- leia [PROJECT_STATUS.md](/workspace/2026/PROJECT_STATUS.md)
- leia [TRAINING_PLAN.md](/workspace/2026/TRAINING_PLAN.md)
- rode o smoke ou o protocolo desejado
