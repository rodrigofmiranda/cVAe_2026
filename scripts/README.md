# Scripts Map

This folder is split by operational status instead of keeping every helper at
the same level.

## Active

- `ops/`
  Host and workflow wrappers used in the current project flow.
- `analysis/`
  Stable post-run helpers that still apply to current experiment layouts.
- `knowledge/`
  Local paper ingestion and retrieval tooling.

## Archived

- `archive/`
  One-off diagnostics, historical smokes, or analysis helpers tied to older
  bugs, layouts, or decision branches.

## File-by-file Status

- `ops/train.sh`
  Current wrapper for `src.protocol.run --train_once_eval_all`.
- `ops/eval.sh`
  Current wrapper for `src.evaluation.evaluate`.
- `ops/import_dataset_lfs.sh`
  Current helper to import a new dataset into `data/` with structure checks and
  Git LFS-ready workflow output.
- `ops/run_tf25_gpu.sh`
  Current persistent GPU container launcher. Bootstraps persistent Python plot
  deps into repo-local `.pydeps` on container start (can disable with
  `CVAE_BOOTSTRAP_PLOT_DEPS=0`).
- `ops/enter_tf25_gpu.sh`
  Current tmux/container attach helper.
- `ops/stop_tf25_gpu.sh`
  Current tmux/container stop helper.
- `ops/container_bootstrap_python.sh`
  Container-side Python bootstrap (sets `PYTHONNOUSERSITE=1`, writable
  `MPLCONFIGDIR`, and persistent `.pydeps` with `numpy<2` + `matplotlib`).
- `ops/prune_incomplete_experiments.py`
  Kept as an active maintenance utility for cleaning stale incomplete runs.
- `analysis/summarize_experiment.py`
  Kept as the main compact run summarizer used in current docs.
- `analysis/compare_protocol_finalists.py`
  Kept as a tested comparison helper for candidate tags inside a protocol run.
- `analysis/recompute_validation_gates.py`
  Kept as an active backfill tool when validation-gate logic changes.
- `knowledge/ingest_papers_docling.py`
  Kept as the current PDF ingestion entrypoint for the knowledge base.
- `knowledge/index_knowledge_chroma.py`
  Kept as the current local indexing entrypoint for the knowledge base.
- `archive/benchmark_batchsize_throughput.py`
  Archived because it was a one-off throughput study and is not part of the
  current experiment workflow.
- `archive/check_kurt_pred.py`
  Archived because it targeted a specific kurtosis diagnostic during the G5/G6
  investigation.
- `archive/cross_reference_grid_history.py`
  Archived because it summarizes historical grid history rather than current
  operational work.
- `archive/diag_g6_full.py`
  Archived because it is a focused G6 incident diagnostic for a historical run.
- `archive/eval_final_model.py`
  Archived because it is a one-off re-evaluation script for a specific model
  snapshot.
- `archive/smoke_dist_metrics.sh`
  Archived because it validates an older smoke path and older layout concerns.
- `archive/test_dist1m_2grid.py`
  Archived because it is a one-off exploratory launcher for a retired test
  slice.
- `archive/verify_mc_predict.py`
  Archived because it targeted a specific MC prediction bug investigation.
- `archive/verify_pipeline_fixes.py`
  Archived because it verifies a completed historical pipeline-fix checklist.
