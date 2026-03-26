# REFACTOR_PLAN (Archived)

Este documento foi arquivado em 2026-03-12 para evitar confusão com o estado
atual do repositório.

## Status

O objetivo principal do plano foi concluído:

- treino canônico via `python -m src.training.train`
- avaliação canônica via `python -m src.evaluation.evaluate`
- protocolo canônico via `python -m src.protocol.run`
- `state_run.json` centralizado
- `manifest.json` e `summary_by_regime.csv` produzidos pelo pipeline atual
- módulos monolíticos/legados removidos do caminho ativo

## Documento ativo que substitui este plano

- [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
- [TRAINING_PLAN.md](/workspace/2026/feat_seq_bigru_residual_cvae/TRAINING_PLAN.md)
- [docs/reference/PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/PROTOCOL.md)

## Motivo do arquivamento

O plano original descrevia uma transição de arquitetura quando o repositório
ainda dependia de `cvae_TRAIN_documented.py` e `analise_cvae_reviewed.py`.
Essas referências não representam mais o fluxo atual e manter esse arquivo no
root passou a induzir leitura equivocada do estado do código.
