# Mapa Da Documentação

Esta pasta é organizada por função para que entrypoints públicos, notas de
trabalho ativas, material de referência e conteúdo arquivado não disputem o
mesmo nível.

## Entradas Públicas

Comece por aqui se você chegou pelo GitHub ou está entrando no repositório agora:

- [../README.md](../README.md) - visão geral pública do repositório
- [BRANCH_GUIDE.md](BRANCH_GUIDE.md) - para que serve cada branch pública
- [../PROJECT_STATUS.md](../PROJECT_STATUS.md) - estado atual do código e da pesquisa

## Operação Interna

Especialmente relevante para o servidor compartilhado do laboratório:

- [active/INFRA_GUIDE.md](active/INFRA_GUIDE.md) - SSH, usuários, Docker, tmux, Git LFS
- [active/MULTI_PC_WORKFLOW.md](active/MULTI_PC_WORKFLOW.md) - layout de worktrees, dois slots quentes e operação entre PCs
- [operations/DATASET_LFS_UPLOAD.md](operations/DATASET_LFS_UPLOAD.md) - fluxo padronizado de import/upload de dataset com Git LFS

## Documentos De Referência

- [reference/PROTOCOL.md](reference/PROTOCOL.md) - runner de protocolo, CLI e artefatos
- [reference/MODELING_ASSUMPTIONS.md](reference/MODELING_ASSUMPTIONS.md) - racional de modelagem
- [archive/reference/DIAGNOSTIC_CHECKLIST.md](archive/reference/DIAGNOSTIC_CHECKLIST.md) - workflow diagnóstico histórico
- [reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt](reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt) - caminho da aquisição ao treino

## Runbooks Operacionais

- [operations/CONTAINER_FULL_TRAINING.md](operations/CONTAINER_FULL_TRAINING.md) - fluxo fim a fim de treino em container, da subida à revisão pós-run
- [operations/TRAINING_STATUS_AND_PROJECT_OVERVIEW.md](operations/TRAINING_STATUS_AND_PROJECT_OVERVIEW.md) - resumo dos treinamentos recentes, baseline oficial, conceito de campeão e mapa do projeto
- [operations/FAILURE_DEEP_DIVE_EXP_20260416_005055.md](operations/FAILURE_DEEP_DIVE_EXP_20260416_005055.md) - diagnostico dos 5 fails do baseline e proximo experimento recomendado

## Arquivo Histórico

- [archive/REFACTOR_PLAN_legacy.md](archive/REFACTOR_PLAN_legacy.md) - contexto histórico do refactor
