# PROJECT_STATUS

> Atualizado em 2026-04-17.
> Este arquivo e o inventario oficial das worktrees e do estado ativo do repositorio.

## Worktrees

Worktrees git registradas neste repositorio:

1. `/home/rodrigo/cVAe_2026_mdn_return`
  - branch: `research/mdn-return-20260416`
   - status: worktree ativa atual
   - foco: recuperar as falhas residuais de `G5` perto de `0.8 m` partindo da
     melhor linha MDN estavel

No momento nao ha worktree secundaria registrada por `git worktree list`.

## Branch Atual

- branch ativa: `research/mdn-return-20260416`
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
- melhor linha MDN v2 valida ate agora:
  - run: `outputs/exp_20260327_161311`
  - resultado: `9/12`
  - gates: `gate_g5_pass=10`, `gate_g6_pass=12`
- melhor linha MDN historica (empate em score final):
  - run: `outputs/exp_20260325_230938`
  - resultado: `9/12`
- ultimo teste negativo da branch atual:
  - run: `outputs/exp_20260326_050257`
  - linha: `regime-weighted resampling`
  - resultado: `3/12`
- ultimo screening rapido da linha `cond_embed fast`:
  - run: `outputs/20260416_204133_seq_cond_embed_fast_stage1_100k/exp_20260416_204135`
  - preset: `seq_cond_embed_fast_stage1`
  - campeao: `S35C_fast_e64_base`
  - resultado: `7/12`
  - leitura operacional:
    - melhor candidato de promocao no caminho rapido atual
    - nao substitui a ancora cientifica `S27` (`10/12`)
  - falhas remanescentes do campeao:
    - `0.8m / 100mA`
    - `0.8m / 300mA`
    - `0.8m / 500mA`
    - `1.0m / 500mA`
    - `1.5m / 300mA`

Leitura atual:

- a familia `seq_bigru_residual` continua sendo a principal linha temporal
- a melhor MDN ficou competitiva, mas ainda abaixo da referencia gaussiana
- `sample-aware MMD`, o flow `sinh-arcsinh` atual e o weighting puro por regime
  devem ser tratados como linhas negativas nesta iteracao
- o gargalo restante segue concentrado em `0.8 m`, principalmente em `G5`
- a linha `cond_embed fast` agora tem um filtro operacional fechado:
  - `S35C_fast_e64_base` venceu o stage-1 rapido em `7/12`
  - o uso correto desse resultado e promocao para follow-up mais caro, nao troca de ancora
- leitura importada da linha estavel `feat/mdn-g5-recovery`:
  - o sweep amplo `S32` de `cond_embed` fechou em `0/12`
  - como direcao geral, o eixo `cond_embed` esta esgotado
  - o unico follow-up ainda justificavel e a promocao estreita do `e64` rapido do `S35`
- fronteira de escopo importante:
  - a linha `full_circle` foi testada apenas na worktree separada
    `/home/rodrigo/cVAe_2026_shape_fullcircle`
  - leituras sobre geometria/suporte daquela branch `research/full-circle` nao
    devem ser tratadas como baseline automatico desta pasta `mdn_return`
- proximo passo estrutural documentado:
  - a linha `shape` deve ser lida como proxy provisoria via pesos/filtros sobre
    `full_square`, nao como validacao final da hipotese de suporte
  - agora que existe aquisicao real em disco, o proximo teste honesto e fechar
    a validacao limpa de `full_circle`
  - nota ativa: `docs/active/FULL_CIRCLE_NEXT_STEP.md`
- a linha ativa agora e `MDN v2`:
  - `coverage/tail loss` opcional via `lambda_coverage`
  - ranking do grid por `mini_protocol_v1`
  - `decoder_sensitivity` seq/MDN corrigido e finito
  - `latent_summary` mantido apenas como telemetria de auditoria
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

- [README.md](README.md)
  - guia principal do repositorio para leitura no GitHub
- [PROJECT_STATUS.md](PROJECT_STATUS.md)
  - inventario oficial das worktrees e estado ativo
- [CODEX.md](CODEX.md)
  - stub de auto-discovery para Codex
- [CLAUDE.md](CLAUDE.md)
  - stub de auto-discovery para Claude

Docs ativos:

- [docs/README.md](docs/README.md)
- [docs/active/WORKING_STATE.md](docs/active/WORKING_STATE.md)
- [docs/reference/PROTOCOL.md](docs/reference/PROTOCOL.md)
- [docs/reference/EXPERIMENT_WORKFLOW.md](docs/reference/EXPERIMENT_WORKFLOW.md)
- [docs/reference/MODELING_ASSUMPTIONS.md](docs/reference/MODELING_ASSUMPTIONS.md)
- [docs/agents/AI_AGENT_GUIDE.md](docs/agents/AI_AGENT_GUIDE.md)
- [docs/agents/REVIEW.md](docs/agents/REVIEW.md)

## Como Retomar Rapidamente

```bash
cd /home/rodrigo/cVAe_2026_mdn_return
git status -sb
git worktree list
python scripts/analysis/summarize_experiment.py "$(ls -td outputs/exp_* | head -1)"
```

Depois disso, leia:

1. [README.md](README.md)
2. [PROJECT_STATUS.md](PROJECT_STATUS.md)
3. [docs/active/WORKING_STATE.md](docs/active/WORKING_STATE.md)
4. [docs/reference/PROTOCOL.md](docs/reference/PROTOCOL.md)

## Arquivo Historico

Tudo que deixou de ser documento vivo foi movido para:

- [docs/archive/README.md](docs/archive/README.md)
