# Mapa De Scripts

Esta pasta foi dividida por status operacional, em vez de manter todos os
helpers no mesmo nível.

## Ativos

- `ops/`
  Wrappers de host e fluxo usados no caminho atual do projeto.
- `analysis/`
  Helpers pós-run estáveis que ainda se aplicam aos layouts atuais.
- `knowledge/`
  Ferramentas locais de ingestão e recuperação de papers.

## Arquivados

- `archive/`
  Diagnósticos pontuais, smokes históricos ou helpers de análise vinculados a
  bugs, layouts ou ramificações de decisão antigas.

## Status Arquivo A Arquivo

- `ops/train.sh`
  Wrapper atual para `src.protocol.run --train_once_eval_all`.
- `ops/eval.sh`
  Wrapper atual para `src.evaluation.evaluate`.
- `ops/import_dataset_lfs.sh`
  Helper atual para importar novo dataset em `data/` com checagens de estrutura
  e saída pronta para workflow com Git LFS.
- `ops/run_tf25_gpu.sh`
  Launcher atual de container GPU persistente. Faz bootstrap de dependências
  Python persistentes para plots em `.pydeps` local ao iniciar (pode desativar
  com `CVAE_BOOTSTRAP_PLOT_DEPS=0`).
- `ops/enter_tf25_gpu.sh`
  Helper atual para attach em tmux/container.
- `ops/stop_tf25_gpu.sh`
  Helper atual para parar tmux/container.
- `ops/container_bootstrap_python.sh`
  Bootstrap Python do lado do container (define `PYTHONNOUSERSITE=1`,
  `MPLCONFIGDIR` gravável e `.pydeps` persistente com `numpy<2` + `matplotlib`).
- `ops/prune_incomplete_experiments.py`
  Mantido como utilitário ativo de manutenção para limpar runs incompletos antigos.
- `analysis/summarize_experiment.py`
  Mantido como sumarizador compacto principal de runs usado na documentação atual.
- `analysis/compare_protocol_finalists.py`
  Mantido como helper testado para comparar tags candidatas dentro de um run de protocolo.
- `analysis/recompute_validation_gates.py`
  Mantido como ferramenta ativa de backfill quando a lógica de gates de validação muda.
- `knowledge/ingest_papers_docling.py`
  Mantido como entrypoint atual de ingestão de PDFs para base de conhecimento.
- `knowledge/index_knowledge_chroma.py`
  Mantido como entrypoint atual de indexação local para base de conhecimento.
- `archive/benchmark_batchsize_throughput.py`
  Arquivado porque foi um estudo pontual de throughput e não faz parte do fluxo atual.
- `archive/check_kurt_pred.py`
  Arquivado porque foi direcionado a um diagnóstico específico de curtose na investigação G5/G6.
- `archive/cross_reference_grid_history.py`
  Arquivado porque resume histórico de grid, e não o trabalho operacional atual.
- `archive/diag_g6_full.py`
  Arquivado porque é um diagnóstico focado em incidente G6 de um run histórico.
- `archive/eval_final_model.py`
  Arquivado porque é um script pontual de reavaliação para um snapshot específico.
- `archive/smoke_dist_metrics.sh`
  Arquivado porque valida um caminho de smoke antigo e preocupações de layout legado.
- `archive/test_dist1m_2grid.py`
  Arquivado porque é um launcher exploratório pontual para um recorte de teste aposentado.
- `archive/verify_mc_predict.py`
  Arquivado porque foi focado em investigação específica de bug de predição MC.
- `archive/verify_pipeline_fixes.py`
  Arquivado porque verifica um checklist histórico já concluído de correções de pipeline.
