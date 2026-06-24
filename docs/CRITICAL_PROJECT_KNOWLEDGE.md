# Conhecimento Crítico do Projeto — VLC cVAE (V3)

> Os fatos **imprescindíveis** que devem estar sempre à mão (sobrevivem à
> compactação de contexto). Revisado 2026-06-24. **Regra de ouro: antes de afirmar
> qualquer item técnico daqui, abra a fonte citada e confirme** (ver
> [[feedback-verify-before-asserting]]). O bloco curto de gotchas está no
> `CLAUDE.md`; este doc é a versão completa.

---

## 1. Reprodutibilidade / Determinismo
- **A API `tf.config.experimental.enable_op_determinism()` CRASHA** este stack
  (TF 2.17 + tf_keras): `'float' object cannot be interpreted as an integer` no
  model build. → flag `CVAE_DET_OPDET` fica **0**.
- **Reprodutibilidade vem das ENV VARS** `TF_DETERMINISTIC_OPS=1` +
  `TF_CUDNN_DETERMINISTIC=1`, ligadas automaticamente por `CVAE_DETERMINISTIC=1`
  (+ `CVAE_DET_ENV` default 1) em `src/protocol/run.py:53-55`. Demonstrado
  bit-idêntico no run pesado do eduardo.
- **Config padrão dos treinos**: `CVAE_DETERMINISTIC=1 CVAE_DET_OPDET=0 CVAE_DET_SETSEED=0`.
- Mesmo seedado, sem determinismo a ordem de redução float na GPU varia → a
  "bacia boa" historicamente era um **sorteio**. Por isso comparações usam
  **2 seeds** (mede o espalhamento). Seed preferida = **33** (evitar 42).
- Fonte: `docs/REPRODUCIBILITY_DETERMINISM_141943.md`, `src/training/pipeline.py:202-224`.

## 2. Criação de containers (template canônico)
- **Treinos rodam SEQUENCIAIS** — o teto é a **RAM do host (64 GB)**, não a GPU
  (32 GB). 2 treinos TF concorrentes estouram (swap satura → OOM-killer, sem
  traceback). Para 2+ runs: foreground por run dentro de UM script detached.
- **Proxy de fila correto = `nvidia-smi --query-gpu=memory.used` baixo**, NUNCA
  presença de container (um `bash` ocioso mantém o container "Up" com a GPU livre
  — foi o bug do `trigger_v3fc_e1gauss.sh`).
- Template: `docker run -d --rm --name <nome> --runtime=nvidia
  -e NVIDIA_VISIBLE_DEVICES=all -e CVAE_DETERMINISTIC=1 -e CVAE_DET_OPDET=0
  -e CVAE_DET_SETSEED=0 -e CVAE_DECODER_LOGVAR_CLAMP_LO=-6.61 -e ..._HI=-0.54
  -u $(id -u):$(id -g) -e HOME=$WORKDIR -v $REPO:$WORKDIR
  -v /home/rodrigo/cVAe_2026_full_square/.pydeps:$WORKDIR/.pydeps
  -v /home/rodrigo/1-Data:/data:ro -w $WORKDIR vlc/tf25-gpu-ready:1`. WORKDIR =
  `/workspace/2026/feat_seq_bigru_residual_cvae`.
- Detached que sobrevive a cortes do harness: `setsid nohup bash script.sh ... & disown`.
- **GRU**: o kernel cuDNN fundido falha → fallback automático
  `seq_gru_backend='compat'` (funciona, mais lento). Esperado, não é erro.
- **`cvae_eduardo`**: container de OUTRO usuário (frequentemente shell ocioso) —
  NÃO parar/tocar; `/home/eduardo` é permission-denied p/ rodrigo.
- Exemplos prontos: `/home/rodrigo/launch_v3fc_e1gauss_now.sh` (modelo).

## 3. Dataset / paths
- **V3 = 5 LEDs saudáveis** (chegou 06-08, em uso). V2 = 3 LEDs + 1 meio-queimado.
- Dados: `/home/rodrigo/1-Data/Dataset/V3/{FULLSQUARE,FULL_CIRCLE,4QAM,16QAM,64QAM}_2026_V3_ORGANIZED`,
  **7 dist × 9 curr = 63 regimes** cada.
- Treino lê via `--dataset_root`; em docker: `-v /home/rodrigo/1-Data:/data:ro` +
  `--dataset_root /data/Dataset/V3/FULL_CIRCLE_2026_V3_ORGANIZED`.
- **Split cross-distance** (definido pelo Rodrigo): TREINO = `{0.75, 1.0, 1.35, 1.5}` m
  (4 dist × 9 curr = 36); UNSEEN = `{0.9, 1.16, 1.25}` m (interpolação).
- `/home/rodrigo/knowledge` (papers/notes), `/home/rodrigo/TESE` (tese, git próprio),
  `/home/rodrigo/comparison_v3` (comparações V3), `/home/rodrigo/ai_workflow`
  (hub: `EXPERIMENT_HISTORY.md`). Fonte: [[reference-data-layout-1data]].

## 4. Decisões de projeto
- **Régua V3 dos gates ADOTADA no código 06-16** (commit 915cddc,
  `src/evaluation/validation_summary.py` TWIN_GATE_THRESHOLDS =
  0.036/0.027/0.040/0.110/0.057/0.049/0.150/0.200). `gate_hetero` e `gate_acf`
  ativos; `gate_coverage` GUARDADO (falta plumbar `coverage_95` do reanalysis p/ o
  summary). Fonte: [[project-gate-audit-v3-ruler]].
- **CUIDADO de contagem**: `validation_status` (régua V3 twin) ≠ `n_pass` do
  `protocol_leaderboard.csv`. Comparar modelos sempre pela MESMA fonte.
- **Campeão formal (régua V3)** = híbrido **s35c_g6a twin 9/12** (seed- e
  régua-robusto). Campeão V2 = **S39B 10/12** (run eduardo `exp_20260427_141943`).
- **Twins-base V3 cross-dist** (os mais usados): FC `outputs/v3fc_crossdist_20260613/exp_20260614_144009`,
  FS `outputs/v3fs_crossdist_20260613/exp_20260614_143056`. val_recon ~-4.767 (bacia boa).
- **FC (full-circle) > FS (full-square)** — veredito CONVERGENTE em TODOS os eixos
  (gates cross-dist, BER treinado/unseen 5-6×, xcorr, AWGN). "disco > quadrado".
- **NÃO REPETIR** (negativos comprovados, agora explicados): MDN-k maior (S29),
  λ_kurt (S28), alargar clamp do logvar (S33), embed grande (S32 = 0/12),
  diffusion/flow/spline decoders. **Todos miram cauda de ruído que não existe** —
  ver §5. Fonte: `ai_workflow/EXPERIMENT_HISTORY.md`.

## 5. Conclusões do diagnóstico (a ciência — base do reprojeto)
- **Dentro de cada regime o canal é ~linear + ruído Gaussiano**: Var(Y−X) ≈ 85%
  ganho linear / ~0% não-linear / ~15% ruído Gaussiano; kurt resíduo ≈ 0. → não há
  cauda pesada intra-regime para modelar. A não-linearidade do LED é efeito ENTRE
  pontos de operação (ganho `a(d,c)` muda com bias/distância).
- **Gargalo = DADOS/LOSS, não capacidade**: o braço heavy (W9/h128) NÃO supera o
  light (ambos 30/63). Decoder **Gaussiano basta** (MDN de cauda é redundante;
  S1–S2 Gaussiano já fez 10/12) — e é **reprodutível** (light seed33 = seed7 = 30/63).
- **Duas falhas estruturais** (cross-champion 06-24, 4 campeões FC-line):
  - **0.75 m e 0.9 m falham na família INTEIRA** (8/9 all-fail) → falha estrutural,
    justifica separação de dados. **0.75 m é TREINADA e tem o MAIOR SNR (17.9 dB)**:
    falha por **precisão relativa** (gates são relativos), não física difícil.
  - **Interpolação de distância falha**: não-vistas 1.16/1.25 m passam ≤3/9
    (held-out 1.0m já mostrara EVM 57-107%). Gap de 0.35 m entre 1.0 e 1.35.
  - **`gate_g3` é o culpado UNIVERSAL** (falha em 100% dos regimes de falha-robusta;
    G6/stat ~99%). G3 = média/dispersão do resíduo rel. à escala.
- **Coverage = ameaça e2e que os gates NÃO veem**: cov95 ≈ 0.77-0.81 (alvo 0.95)
  em TODOS os regimes, inclusive os que passam G1-G6 (twin superconfiante).
- **E2 (densificar grid) REGREDIU** (12/63 vs 30/63) — fix B1 sozinho não ajudou.
- Fontes: `comparison_v3/macro_diagnostics/runs/.../REPORT.md`,
  `.../runs/<stamp>_champion_set/REPORT_champion_set.md`, REDESIGN_PLAN.md.

## 6. Linha de trabalho atual (objetivo + experimentos)
- **Objetivo**: separar/reamostrar os dados e ajustar o treino para quebrar o teto,
  mirando o alvo certo (REDESIGN_PLAN). NÃO é trocar arquitetura.
- **Problema A** (near-field 0.75m, precisão relativa) → fixes **A1** loss de erro
  relativo (mais barato), **A2** oversample ∝ 1/σ, **A3** especialista d≤0.9m.
- **Problema B** (interpolação de distância) → **B1** densificar grid, **B2**
  re-espaçar, **B3** prior de suavidade em a/σ(d,c).
- **Mapa REAL dos experimentos (cuidado com nomes de arquivo ≠ rótulos do plano)**:
  - `v3fc_e1gauss_20260617` = Gaussiano (light+heavy), SEM loss relativo → **30/63**.
  - `v3fc_e2gauss_20260620` = fix **B1** densificado (54 regimes) → **12/63** (negativo).
  - `v3fc_e3relloss_recal_s{33,7}` = fix **A1** loss relativo (= a parte que faltava
    do "E1" do plano; o nome "e3" é só do arquivo). EVM batch-norm, λ=7. **Rodando.**
  - A2, A3, B2, B3, E4(prior) = NÃO feitos.
- **Pipeline macro**: `comparison_v3/macro_diagnostics/` (7 camadas, `--fast`/`--full`);
  `champion_set_compare.py` compara N campeões. Re-rodar após cada treino.
- Fontes: REDESIGN_PLAN.md, [[project-macro-diagnostics-pipeline]].

## 7. Infra / hábitos
- **ntfy**: todo run longo avisa em `https://ntfy.sh/projeto_vlc_ia` (Rodrigo larga
  o terminal). Sempre deixar watcher/notificação.
- **Comparações**: SEMPRE consultar `/home/rodrigo/comparison` (V2 congelada) p/
  estilo antes de desenhar comparação nova ([[feedback-comparison-folder]]).
- **Memória de projeto**: `~/.claude/projects/-home-rodrigo/memory/` (índice
  `MEMORY.md`). Atualizar ao tomar decisões/achados duráveis.
- **Nota de bacia (atualização 06-24)**: a memória antiga "máquina não atinge a
  bacia boa (~28 runs em -3.9)" era da reprodução S39B; a **linha V3 light
  (s35c/Gaussiana) ATINGE a bacia boa** (val_recon -4.72/-4.77 em E1/E2/E3). Não
  carregar esse bloqueio como atual sem verificar.
