# PROJECT_STATUS

> Atualizado em 2026-04-04.
> Este arquivo e o inventario oficial das worktrees e do estado ativo do repositorio.

## Worktrees

Worktrees git registradas neste repositorio:

1. `/workspace/2026/feat_pointwise_2025_revival`
   - branch: `feat/pointwise-2025-revival`
   - status: worktree ativa atual
   - foco: `S38` — revival point-wise da licao util de 2025 sob protocolo 2026

2. `/workspace/2026/feat_mdn_g5_recovery_remote`
   - branch: `feat/mdn-g5-recovery-explore-remote`
   - status: worktree complementar preparada para execucao remota
   - foco: sweep complementar de KL/coverage e presets fast two-stage

3. worktrees adicionais registradas via `git worktree list`
   - total atual: `9`
   - incluem linhas de diffusion, flow, mdn_route, graybox e a branch base
     `feat/mdn-g5-recovery-run`

## Branch Atual

- branch ativa: `feat/pointwise-2025-revival`
- entrypoint canonico:
  - `python -m src.protocol.run`

O fluxo publico de experimentacao permanece:

- treino global + avaliacao por regime:
  - `--train_once_eval_all`
- reuso de modelo treinado:
  - `--reuse_model_run_dir`

## Estado Cientifico

Leitura especifica desta branch:

- `S38 smoke`:
  - run: `outputs/exp_20260402_165714`
  - resultado: `10/12`
  - leitura: a hipotese point-wise local continua viva
- `S38 quick`:
  - run: `outputs/exp_20260402_234032`
  - resultado: `3/12`
  - leitura: portar hiperparametros grandes estilo 2025 para o protocolo 2026
    foi negativo
- `S38b`:
  - run: `outputs/exp_20260404_150425`
  - status: em andamento
  - leitura pretendida: decidir se o `10/12` do smoke foi uma regiao real de
    hiperparametro pequeno ou apenas um draw favoravel

Referencias principais hoje:

- referencia gaussiana estavel:
  - run: `outputs/exp_20260324_023558`
  - resultado: `10/12`
- melhor linha MDN v2 valida ate agora:
  - run: `outputs/exp_20260327_161311`
  - resultado: `9/12`
  - gates: `gate_g5_pass=10`, `gate_g6_pass=12`
- melhor ancora historica de `10/12` na linha S27:
  - run: `outputs/exp_20260328_153611`
  - campeao: `S27cov_lc0p25_tail95_t0p03`
  - falhas: `0.8m/100mA`, `0.8m/300mA`
- ultimo quick de triagem da continuacao atual:
  - run: `outputs/exp_20260331_155321`
  - campeao: `S35F_fast_e32_b15`
  - resultado: `9/12`
- ultimo full protocol da continuacao atual:
  - run: `outputs/exp_20260331_220659`
  - campeao: `S36F_full_e32_fb05`
  - resultado: `10/12`
  - falhas: `0.8m/300mA`, `0.8m/500mA`
- ultimo probe radial SDN:
  - run: `outputs/exp_20260401_172117`
  - campeao registrado: `S37A_baseline_no_radial`
  - resultado: `10/12` sob caps
  - observacao: os candidatos radiais `S37B/S37C` falharam por bug de
    interface no path sequencial rapido e precisam de rerun apos a correcao
- melhor linha MDN historica (empate em score final):
  - run: `outputs/exp_20260325_230938`
  - resultado: `9/12`
- ultimo teste negativo da branch atual:
  - run: `outputs/exp_20260326_050257`
  - linha: `regime-weighted resampling`
  - resultado: `3/12`

Leitura atual:

- a familia `seq_bigru_residual` continua sendo a principal linha temporal
- a melhor MDN ja igualou o teto de `10/12` em duas configuracoes distintas
  (`S27` e `S36`)
- `sample-aware MMD`, o flow `sinh-arcsinh` atual e o weighting puro por regime
  devem ser tratados como linhas negativas nesta iteracao
- o gargalo restante segue concentrado em `0.8 m`, apenas em `G5`
- a linha atual mostra um tradeoff claro:
  - `S27` passa `0.8m/500mA` e falha `0.8m/100mA`
  - `S36` passa `0.8m/100mA` e falha `0.8m/500mA`
- `0.8m/300mA` continua sendo a falha persistente entre as melhores variantes
- o probe `S37` nao mudou a leitura cientifica ainda:
  - o baseline sem radial reproduziu `10/12`
  - os bracos com `radial_feature=True` nao sao negativos validos
  - a branch agora ja contem a correcao de interface e os testes para rerodar
- proxima linha arquitetural planejada:
  - aproveitar a licao util de 2025 como modelador local condicionado em
    `x_center`
  - plano fisico de retomada salvo em:
    - `docs/active/S38_S40_2025_TRANSFER_PLAN.md`
  - ordem proposta:
    - `S38`: baseline point-wise estilo 2025 sob protocolo 2026
    - `S39`: seq backbone + expert local heteroscedastico
    - `S40`: seq backbone + expert local MDN
- estado local do `S38` nesta branch:
  - `s38_pointwise_2025_smoke`: implementado e validado
  - `s38_pointwise_2025_quick`: implementado e encerrado como negativo para a
    regiao grande/legacy
  - `s38_pointwise_small_local`: implementado e em execucao
- a linha ativa agora e `MDN v2`:
  - `coverage/tail loss` opcional via `lambda_coverage`
  - ranking do grid por `mini_protocol_v1`
  - `decoder_sensitivity` seq/MDN corrigido e finito
  - `latent_summary` mantido apenas como telemetria de auditoria
- a continuacao local adicionou:
  - clamp largo por ambiente (`CVAE_DECODER_LOGVAR_CLAMP_LO/HI`)
  - preset `seq_cond_embed_clamp_large_sweep` (`S33`)
  - preset `seq_cond_embed_fast_stage1` (`S35`)
  - preset `seq_cond_embed_fast_stage2` (`S36`)
  - preset `radial_sdn_test` (`S37`)
  - suporte a `radial_feature=True` no cVAE point-wise e sequencial
- tambem existe agora uma comparacao de throughput opt-in para a linha seq MDN:
  - preset: `seq_mdn_v2_perf_compare_quick`
  - controle atual: `batch_size=4096`, `batch_infer=8192`, `seq_gru_unroll=True`
  - variante de lote maior: `batch_size=8192`, `batch_infer=16384`
  - variante GRU rapida: `seq_gru_unroll=False`
  - ressalva: `seq_gru_unroll=True` continua sendo o default conservador por compatibilidade historica com stacks novos, incluindo a linha com RTX 5090
- resultado do compare operacional mais recente:
  - run: `outputs/exp_20260326_234236`
  - vencedor operacional: `batch_size=8192`, `batch_infer=16384`, `seq_gru_unroll=False`
  - throughput observado no A6000:
    - controle `4096/gruroll1`: ~`14s/epoch`
    - lote maior `8192/gruroll1`: ~`7s/epoch`
    - lote maior `8192/gruroll0`: ~`6s/epoch`
  - preset de continuidade cientifica sobre essa base:
    - `seq_mdn_v2_fastbase_quick`
- resultado do primeiro quick cientifico nessa base rapida:
  - run: `outputs/exp_20260327_021632`
  - campeao: `S22 ... cov0p05_t0p03 ...`
  - resultado: `5/12`
  - ganho principal: `G6` subiu de `4` para `6` passes contra o fastbase sem coverage
  - tradeoff: `G5` caiu de `10` para `9`
- preset de follow-up local:
  - `seq_mdn_v2_g5_followup_quick`
- resultado do follow-up local:
  - run: `outputs/exp_20260327_032019`
  - campeao: `S23 ... cov0p06_t0p03 ...`
  - resultado: `6/12`
  - ganho principal: `0.8m / 700mA` virou `pass`
  - gaps restantes: `0.8m` de baixa corrente, `1.0m / 300-500mA`, `1.5m / 300mA`
  - preset overnight para decidir continuidade da linha:
    - `seq_mdn_v2_overnight_decision_quick`
    - mistura refinamento local do `S23` + probes estruturais/exploratorios
  - ressalva operacional:
    - na RTX 5090, o overnight original nao e seguro para probes estruturais com
      `seq_gru_unroll=False`
    - preset seguro para essa stack:
      - `seq_mdn_v2_overnight_5090safe_quick`
      - mantem `gruroll0` so no ramo local `W7 / h64`
      - força `gruroll1` em `h96`, `W11` e probes combinados
  - preset complementar para o A600:
    - `seq_mdn_v2_a600_tail_explore_quick`
    - abre sweep proprio de `tail_levels`
    - mantém probes estruturais no caminho rapido `gruroll0`
    - serve como contraponto exploratorio ao overnight `5090-safe`
  - resultado do contraponto no A600:
    - run: `outputs/exp_20260327_050422`
    - campeao: `S26 ... lat6 ... tail02-98 ...`
    - resultado: `5/12`
    - leitura: negativo para a hipotese de que abrir `tail_levels` separadamente
      resolveria o gargalo de `0.8 m`
  - resultado do overnight `5090-safe`:
    - run: `outputs/exp_20260327_050158`
    - campeao de treino: `S25 ... h96 / lat6 / gruroll1 ...`
    - protocolo final ficou invalido
    - causa operacional: ambiente de avaliacao sem `matplotlib`
    - efeito: todos os regimes ficaram com `eval_status=failed` e `G1-G3`
      vazios, entao o `0/12` nao pode ser tratado como resultado cientifico
    - sinal util preservado:
      - o melhor candidato veio de probe estrutural
      - `gate_g5_pass=9`
      - `gate_g6_pass=10`
      - isso motivou reavaliar o modelo treinado antes de abrir outro grid
  - resultado da reavaliacao valida:
    - run: `outputs/exp_20260327_161311`
    - status: `completed`
    - campeao validado: `S25 ... W7 / h64 / lat6 / gruroll1 ...`
    - resultado: `9/12`
    - falhas restantes:
      - `0.8m / 300mA` (`G3`)
      - `0.8m / 500mA` (`G5`)
      - `0.8m / 700mA` (`G5`)
    - leitura:
      - melhor MDN v2 valida da branch ate o momento
      - ainda `1` regime abaixo da referencia gaussiana `10/12`
  - regra operacional atual da linha RTX 5090:
    - existe agora retry automatico no gridsearch para candidatos seq com
      erro de runtime do cuDNN/GRU
    - quando o kernel fused falha, o treino reabre o mesmo candidato com
      backend GRU compativel
    - isso evita depender de presets separados por GPU so para contornar o
      `DoRnnForward` da 5090

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

Outro ponto de atencao operacional confirmado nesta iteracao:

- se o ambiente de avaliacao estiver sem `matplotlib`, o protocolo pode gerar
  `metricas_globais_reanalysis.json`
- a avaliacao agora pula o dashboard sem invalidar o resultado final
- referencia concreta:
  - `outputs/exp_20260327_050158`

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
cd /workspace/2026/feat_mdn_g5_recovery_explore
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
