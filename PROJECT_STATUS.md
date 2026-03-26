# PROJECT_STATUS

> Atualizado em 2026-03-26.
> Este arquivo e o inventario oficial das worktrees e do estado ativo do repositorio.

## Worktrees

Worktrees git registradas neste repositorio:

1. `/workspace/2026/feat_seq_bigru_residual_cvae`
   - branch: `feat/mdn-g5-recovery`
   - status: worktree ativa atual
   - foco: recuperar as falhas residuais de `G5` perto de `0.8 m` partindo da
     melhor linha MDN estavel

No momento nao ha worktree secundaria registrada por `git worktree list`.

## Branch Atual

- branch ativa: `feat/mdn-g5-recovery`
- entrypoint canonico:
  - `python -m src.protocol.run`

O fluxo publico de experimentacao permanece:

- treino global + avaliacao por regime:
  - `--train_once_eval_all`
- reuso de modelo treinado:
  - `--reuse_model_run_dir`

## Estado Cientifico

Referencias principais hoje:

- referencia gaussiana estavel:
  - run: `outputs/exp_20260324_023558`
  - resultado: `10/12`
- melhor linha MDN ate agora:
  - run: `outputs/exp_20260325_230938`
  - resultado: `9/12`
- ultimo teste negativo da branch atual:
  - run: `outputs/exp_20260326_050257`
  - linha: `regime-weighted resampling`
  - resultado: `3/12`

Leitura atual:

- a familia `seq_bigru_residual` continua sendo a principal linha temporal
- a melhor MDN ficou competitiva, mas ainda abaixo da referencia gaussiana
- `sample-aware MMD`, o flow `sinh-arcsinh` atual e o weighting puro por regime
  devem ser tratados como linhas negativas nesta iteracao
- o gargalo restante segue concentrado em `0.8 m`, principalmente em `G5`
- a linha ativa agora e `MDN v2`:
  - `coverage/tail loss` opcional via `lambda_coverage`
  - ranking do grid por `mini_protocol_v1`
  - `decoder_sensitivity` seq/MDN corrigido e finito
  - `latent_summary` mantido apenas como telemetria de auditoria

## Ponto De Atencao Operacional

Existe um bug ja encontrado e corrigido nesta branch que precisa ser
reverificado sempre que a linha `seq_bigru_residual` for usada em outra branch
ou worktree:

- se houver `max_samples_per_exp` e/ou `max_val_samples_per_exp`, o `df_split`
  precisa ser atualizado para contagens pos-cap antes do windowing
- sem esse ajuste, o center sample continua correto, mas o contexto temporal da
  janela pode vazar entre experimentos
- o risco vale para:
  - treino sequencial
  - `_quick_cvae_predict` / avaliacao sequencial no protocolo
  - qualquer split sequencial em que arrays tenham sido capados por experimento
- o risco nao vale para:
  - runs full sem caps
  - modelos point-wise sem windowing

Commits de referencia desta correcao:

- `a1660e2` `fix(seq): sync df_split after per-exp caps`
- `c6d1a0a` `docs: drop stale smoke b2 notes`

## Familias De Modelo Disponiveis

- `concat`
  - cVAE point-wise original
- `channel_residual`
  - decoder residual point-wise
- `delta_residual`
  - residual-target point-wise
- `seq_bigru_residual`
  - linha temporal principal
- `legacy_2025_zero_y`
  - comparacao historica controlada

## Artefatos Canonicos

Para julgar um run, use:

- `tables/protocol_leaderboard.csv`
- `tables/summary_by_regime.csv`
- `tables/residual_signature_by_regime.csv`
- `tables/stat_fidelity_by_regime.csv`
- `train/tables/gridsearch_results.csv`

Artefatos operacionais principais:

- `manifest.json`
- `train/state_run.json`
- `train/tables/grid_training_diagnostics.csv`
- `train/models/grid_*/logs/train/mini_protocol_summary.json`
- `train/models/grid_*/tables/mini_protocol_by_regime.csv`
- `train/plots/champion/analysis_dashboard.png`
- `plots/best_model/heatmap_gate_metrics_by_regime.png`

## Estrutura De Documentacao Viva

Raiz:

- [README.md](/workspace/2026/feat_seq_bigru_residual_cvae/README.md)
  - guia principal do repositorio para leitura no GitHub
- [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
  - inventario oficial das worktrees e estado ativo
- [CODEX.md](/workspace/2026/feat_seq_bigru_residual_cvae/CODEX.md)
  - stub de auto-discovery para Codex
- [CLAUDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/CLAUDE.md)
  - stub de auto-discovery para Claude

Docs ativos:

- [docs/README.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/README.md)
- [docs/active/WORKING_STATE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/WORKING_STATE.md)
- [docs/reference/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/PROTOCOL.md)
- [docs/reference/EXPERIMENT_WORKFLOW.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/EXPERIMENT_WORKFLOW.md)
- [docs/reference/MODELING_ASSUMPTIONS.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/MODELING_ASSUMPTIONS.md)
- [docs/agents/AI_AGENT_GUIDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/AI_AGENT_GUIDE.md)
- [docs/agents/REVIEW.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/REVIEW.md)

## Como Retomar Rapidamente

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
git status -sb
git worktree list
python scripts/analysis/summarize_experiment.py "$(ls -td outputs/exp_* | head -1)"
```

Depois disso, leia:

1. [README.md](/workspace/2026/feat_seq_bigru_residual_cvae/README.md)
2. [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
3. [docs/active/WORKING_STATE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/WORKING_STATE.md)
4. [docs/reference/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/PROTOCOL.md)

## Arquivo Historico

Tudo que deixou de ser documento vivo foi movido para:

- [docs/archive/README.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/archive/README.md)
