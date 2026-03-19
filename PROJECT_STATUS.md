# PROJECT_STATUS.md — Estado Atual do Repositório

> Atualizado em 2026-03-19.

## 1. Estado técnico

O refactor de engenharia está concluído no caminho ativo.

### Entry point canônico

- `python -m src.protocol.run`

`src.training.train` permanece apenas como shim de compatibilidade e não deve
mais ser tratado como fluxo público de experimentação.

### Modos do protocolo

O protocolo agora tem dois modos explícitos:

- `per_regime_retrain`:
  - padrão quando `--train_once_eval_all` não é usado
  - treina um cVAE por regime
  - serve para diagnóstico e comparação local contra o baseline
- `train_once_eval_all`:
  - ativado com `--train_once_eval_all`
  - treina um único cVAE global em `outputs/exp_.../train`
  - avalia esse mesmo modelo em todos os regimes, sem retreino por regime
  - este é o modo alinhado ao objetivo final do digital twin universal

### Famílias de arquitetura disponíveis

- `concat`
  - cVAE point-wise original
- `channel_residual`
  - decoder residual point-wise
- `delta_residual`
  - residual-target point-wise
  - melhor linha point-wise atual
- `seq_bigru_residual`
  - residual temporal com janela e BiGRU
  - única linha com referência histórica que já passou todos os gates
- `legacy_2025_zero_y`
  - porta o modelo antigo de 2025 para comparação controlada

A escolha entre essas arquiteturas é feita por `arch_variant` no `cfg` de cada
grid, dentro da mesma branch e do mesmo pipeline.

### Artefatos canônicos

- `state_run.json`: snapshot de um run individual
- `manifest.json`: snapshot consolidado de um experimento do protocolo
- `tables/summary_by_regime.csv`: tabela canônica de validação por regime
- `tables/stat_fidelity_by_regime.csv`: projeção derivada das métricas estatísticas
- `tables/protocol_leaderboard.csv`: ranking canônico derivado do próprio protocolo
- `train/plots/champion/analysis_dashboard.png`: dashboard completo do campeão
- `plots/best_model/heatmap_vae_vs_real_metric_diffs.png`: heatmap compacto do campeão

### Métricas consolidadas

`summary_by_regime.csv` é a fonte única de verdade para:

- fidelidade física: `EVM`, `SNR`
- fidelidade distribucional do resíduo `Δ = Y - X`
- comparação `baseline` vs `cVAE`
- testes formais `MMD`, `Energy`, `PSD`
- gates `G1`–`G6`
- `validation_status`

### Layout atual de saída

O layout atual é:

- `train/`
  - modelo global compartilhado, tabelas de grid e dashboard do campeão
- `eval/`
  - artefatos por regime
- `logs/`
  - logs centralizados do experimento
- `tables/`
  - sumários canônicos do protocolo
- `plots/best_model/`
  - heatmap final do campeão

As pastas antigas `global_model/` e `studies/` não são mais o layout-alvo.

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
- a exploração séria agora é **protocol-first**
- o preset comparativo atual é `best_compare_large`, que compara:
  - candidatos `delta_residual`
  - candidatos `seq_bigru_residual` incluindo o bloco `lambda_mmd`
- a linha seq agora tem fallback não-cuDNN para janelas curtas, permitindo
  rodar também em GPUs mais novas como a RTX 5090

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
