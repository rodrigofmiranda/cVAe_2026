# Claude Code Instructions

This file stays at repo root because Claude-style tooling commonly looks for
`CLAUDE.md` automatically.

## Shared Project Context

Read the common guide first:

- [docs/agents/AI_AGENT_GUIDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/AI_AGENT_GUIDE.md)
- [docs/active/WORKING_STATE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/WORKING_STATE.md)

Then read:

- [docs/agents/REVIEW.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/agents/REVIEW.md)

## Gotchas / Invariantes (verificar no código antes de afirmar)

📌 **Referência completa "ponta da língua"**: [docs/CRITICAL_PROJECT_KNOWLEDGE.md](docs/CRITICAL_PROJECT_KNOWLEDGE.md)
— reprodutibilidade, containers, dataset, decisões, conclusões do diagnóstico e a
linha de trabalho atual, cada item com a fonte. Os essenciais estão resumidos
abaixo. **Antes de afirmar qualquer um destes, confira a fonte citada** — não
responda de memória.

- **Determinismo**: NÃO use `tf.config.experimental.enable_op_determinism()`
  (`CVAE_DET_OPDET=1`) — crasha com `'float' object cannot be interpreted as an
  integer` no model build (TF2.17/tf_keras). A reprodutibilidade vem das env vars
  `TF_DETERMINISTIC_OPS=1` + `TF_CUDNN_DETERMINISTIC=1`, que `CVAE_DETERMINISTIC=1`
  (+ `CVAE_DET_ENV` default 1) já liga em `src/protocol/run.py:53-55`. Config padrão
  dos runs: `CVAE_DETERMINISTIC=1 CVAE_DET_OPDET=0 CVAE_DET_SETSEED=0`. Doc:
  `docs/REPRODUCIBILITY_DETERMINISM_141943.md`.
- **Teto = RAM do host (64 GB), não a GPU (32 GB)**: 2 treinos TF concorrentes
  estouram a RAM (swap satura, OOM-killer mata sem traceback). Treinos rodam
  **SEQUENCIAIS**. Proxy de fila correto = `nvidia-smi memory.used` baixo, NÃO
  presença de container (um `bash` ocioso mantém o container "Up" com a GPU livre).
- **GRU**: o kernel cuDNN fundido falha → fallback automático
  `seq_gru_backend='compat'` (funciona, mais lento). Esperado, não é erro.
- **Contagem de gates**: a régua V3 (`validation_status` / `gates_summary_*.csv`)
  ≠ `n_pass` do `protocol_leaderboard.csv` (ex.: base FC 29-30/63 régua vs 39/63
  protocolo). Comparar modelos sempre pela MESMA fonte.
- **Join de tabelas** por `(dist_m float, curr_mA int)`, NUNCA por string de regime
  (gate table escreve `dist_1m`, metrics escreve `dist_1p0m`).
- **`cvae_eduardo`**: container de outro usuário (frequentemente shell ocioso) —
  NÃO parar/tocar; `/home/eduardo` é permission-denied p/ rodrigo.

## Claude-Specific Start

- run `/memory` and confirm project memory and rules are loaded
- this project is configured to start in Plan Mode via `.claude/settings.json`
- use `/rename <task-name>` early so the session is easy to resume later

Useful project skills:

- `/seq-bigru-kickoff [phase-or-focus]`
- `/seq-review [scope]`

## Claude-Specific Hygiene

- use `/compact` during long sessions to preserve key decisions
- use `/clear` between unrelated tasks
- for parallel investigations, prefer separate worktrees

## Ignore Unless Asked

- `knowledge/`
- `scripts/knowledge/index_knowledge_chroma.py`
- `.gitignore` changes related to local paper/index tooling

## Out Of Scope

- do not edit `knowledge/`, `data/`, or generated `outputs/` unless explicitly asked
- do not spend time extending local paper-ingestion or retrieval tooling in this branch
