# PROJECT_STATUS.md — Estado Atual do Repositório

> Atualizado em 2026-03-24.

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

Além disso, o modo `train_once_eval_all` agora aceita:

- `--reuse_model_run_dir`
  - pula o treino global
  - reutiliza `models/best_model_full.keras` de um run anterior
  - permite reavaliar o mesmo campeão em outro protocolo sem retrain

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
- `train/tables/grid_training_diagnostics.csv`: diagnóstico operacional resumido de todos os grids
- `train/plots/champion/analysis_dashboard.png`: dashboard completo do campeão
- `train/plots/training/dashboard_analysis_complete.png`: dashboard operacional de treino, evolução e convergência
- `plots/best_model/heatmap_gate_metrics_by_regime.png`: heatmap científico canônico do campeão por regime

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
  - heatmap científico final do campeão

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
- o run `exp_20260324_023558` elevou a referência seq para `10/12` passes com
  `S6seq_W7_h64_lat4_b0p003_lmmd1p75_fb0p10_lr0p0003_L128-256-512`
- as novas métricas por eixo mostraram que o gargalo restante está concentrado
  em `0.8 m / 100 mA` e `0.8 m / 300 mA`
- a auditoria de ruído em [docs/NOISE_DISTRIBUTION_AUDIT.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/NOISE_DISTRIBUTION_AUDIT.md)
  mostrou que o cVAE ainda fica subdisperso nas frentes curtas:
  - histograma previsto estreito demais
  - pico central forte demais
  - caudas curtas demais
- a causa mais provável no código atual é:
  - o termo `MMD` regulariza `y_mean - x`, não uma amostra do residual previsto
  - isso ajuda o centro da nuvem, mas não força a largura/cauda da distribuição gerada
- falhas em `G3`–`G6`, quando persistem após os fixes, devem ser tratadas como
  possível limitação do modelo e não como bug operacional
- a exploração séria agora é **protocol-first**
- o preset comparativo atual é `best_compare_large`, que compara:
  - candidatos `delta_residual`
  - candidatos `seq_bigru_residual` incluindo o bloco `lambda_mmd`

### Melhor referência seq atual

- run:
  - `outputs/exp_20260324_023558`
- campeão:
  - `S6seq_W7_h64_lat4_b0p003_lmmd1p75_fb0p10_lr0p0003_L128-256-512`
- resultado:
  - `10/12` passes
  - `2/4` em `0.8 m`
  - `4/4` em `1.0 m`
  - `4/4` em `1.5 m`

### Próximo grid mais assertivo

- preset:
  - `seq_finish_0p8m`
- foco:
  - manter `W7_h64` fixo
  - centrar em `lambda_mmd=1.75`
  - testar `lambda_mmd=2.0`
  - usar apenas hedges de `beta/lr` já fortes no overnight
- o protocolo reduzido multi-regime atual é `configs/all_regimes_sel4curr.json`
- o protocolo agora consegue separar:
  - dashboard científico do twin por regime
  - dashboard operacional de convergência do treino por grid
- a linha seq agora tem fallback não-cuDNN para janelas curtas, permitindo
  rodar também em GPUs mais novas como a RTX 5090
- o heatmap canônico final agora usa leitura visual padronizada:
  - verde = mais próximo do real
  - vermelho = mais distante
  - anotação preta e maior para facilitar leitura

### Referências recentes que explicam a direção atual

Marco anterior que consolidou a família `W7_h64` antes do salto para `10/12`:

- `outputs/exp_20260322_193738`
- campeão:
  - `S4seq_W7_h64_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`
- protocolo:
  - `configs/all_regimes_sel4curr.json`
  - `train_once_eval_all`

Resumo científico:

- `6/12` regimes aprovados
- `0/4` passes em `0.8 m`
- `2/4` passes em `1.0 m`
- `4/4` passes em `1.5 m`

Leitura:

- `lambda_mmd=1.25` abriu o caminho multi-regime
- esse run mostrou que a família `W7_h64` era a base correta
- ele deixou explícito que o gargalo científico estava concentrado na frente curta `0.8 m`

### Comparação recente que não virou nova referência

O run mais recente de comparação focada foi:

- `outputs/exp_20260323_210309`
- campeão:
  - `S4seq_W7_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`

Resumo:

- treinou `4` candidatos focados em contexto/capacidade
- terminou também com `6/12` passes
- caiu de `gate_pass_ratio=0.805556` para `0.777778`
- perdeu `G6` em `1.0 m / 300 mA`
- a leitura final foi:
  - aumentar `seq_hidden_size` para `96` melhorou o ranking de treino
  - mas não melhorou a validação protocol-first
  - a melhor família continua sendo `W7_h64` com `lambda_mmd=1.25`

### Replay curto mais recente com métricas por eixo

O replay curto que regenerou os melhores candidatos sob as novas métricas
axis-wise foi:

- `outputs/exp_20260324_024442`
- campeão:
  - `S4seq_W7_h64_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512`

Resumo:

- terminou com `8/12` passes
- passou todos os regimes `1.0 m` e `1.5 m`
- manteve todas as falhas restantes concentradas em `0.8 m`
- confirmou que:
  - a família líder continua sendo `W7_h64`
  - o problema restante não é de capacidade global
  - as novas métricas ajudam a localizar o mismatch marginal por eixo nos dois regimes ainda críticos

### Overnight amplo mais recente

O overnight amplo que produziu a referência atual foi:

- preset:
  - `seq_overnight_12h`
- run:
  - `outputs/exp_20260324_023558`
- resultado:
  - `10/12` passes
  - `lambda_mmd=1.75` como nova direção vencedora
  - `W7_h64` confirmado como família líder

Leitura:

- o overnight amplo já respondeu à pergunta de busca larga
- ele não precisa ser repetido como próximo passo imediato
- a partir dele, o trabalho passou de exploração ampla para acabamento focado em `0.8 m`

## 3. Documentos ativos

- [README.md](/workspace/2026/feat_seq_bigru_residual_cvae/README.md): visão geral e uso
- [docs/ACTIVE_CONTEXT.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/ACTIVE_CONTEXT.md): ponto de entrada curto para a branch ativa
- [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md): estado atual do código
- [TRAINING_PLAN.md](/workspace/2026/feat_seq_bigru_residual_cvae/TRAINING_PLAN.md): plano científico e gates
- [docs/FUTURE_ADVERSARIAL_STRATEGY.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/FUTURE_ADVERSARIAL_STRATEGY.md): backlog único para uma volta futura da estratégia adversarial
- [docs/RUN_REANALYSIS_PLAYBOOK.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/RUN_REANALYSIS_PLAYBOOK.md): como reavaliar rapidamente novos `exp_*`
- [docs/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/PROTOCOL.md): protocolo, artefatos e CLI
- [docs/MODELING_ASSUMPTIONS.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/MODELING_ASSUMPTIONS.md): premissas do modelo

## 4. Documentos arquivados

- [docs/archive/REFACTOR_PLAN_legacy.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/archive/REFACTOR_PLAN_legacy.md)

## 5. Como retomar rapidamente

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
git status -sb
git log --oneline -5
python scripts/summarize_experiment.py
```

Depois disso:

- leia [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
- leia [TRAINING_PLAN.md](/workspace/2026/feat_seq_bigru_residual_cvae/TRAINING_PLAN.md)
- rode o smoke ou o protocolo desejado
