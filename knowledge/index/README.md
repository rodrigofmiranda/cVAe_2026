# Knowledge Index

Camada local de retrieval sobre os artefatos parseados pelo Docling.

## Papel desta pasta

- `chroma_db/`: base vetorial local
- `index_state.json`: estado incremental do indexador
- `chroma_db_*`: bases vetoriais auxiliares para corpora importados
- `index_state_*.json`: estados incrementais dessas bases auxiliares

## Fonte canonica

Esta camada **nao** le PDFs brutos.

Ela indexa o Markdown que ja foi extraido em:

```text
knowledge/papers/parsed/<paper_id>/document.md
```

Tambem pode indexar corpora importados que tenham sido normalizados para o
mesmo layout, por exemplo:

```text
knowledge/imports/docling_cache_md/parsed/<paper_id>/document.md
```

## Fluxo

1. colocar PDFs em `knowledge/papers/raw/`
2. converter com `scripts/knowledge/ingest_papers_docling.py`
3. indexar com `scripts/knowledge/index_knowledge_chroma.py`
4. recuperar trechos relevantes para prompts, notas e sinteses

## Comando

```bash
python scripts/knowledge/index_knowledge_chroma.py
```

Ou com paths explicitos:

```bash
python scripts/knowledge/index_knowledge_chroma.py \
  --parsed-dir knowledge/papers/parsed \
  --db-dir knowledge/index/chroma_db \
  --state-path knowledge/index/index_state.json
```

Exemplo para uma base importada:

```bash
python scripts/knowledge/index_knowledge_chroma.py \
  --parsed-dir knowledge/imports/docling_cache_md/parsed \
  --db-dir knowledge/index/chroma_db_imported_docling_cache_md \
  --state-path knowledge/index/index_state_imported_docling_cache_md.json \
  --collection-name knowledge_imported_docling_cache_md
```

## Consulta local

Depois de indexar, consulte a base local com:

```bash
python scripts/knowledge/query_knowledge.py "probabilistic shaping nonlinearity"
```

O query roda em modo offline por padrao e usa CPU, aproveitando o cache local
do modelo de embeddings.

Buscar nas duas bases e deduplicar por paper:

```bash
python scripts/knowledge/query_knowledge.py \
  "visible light communication digital twin nonlinearity" \
  --source both \
  --dedupe-paper \
  --show-paths
```

Filtrar por um paper especifico:

```bash
python scripts/knowledge/query_knowledge.py \
  "average power constraint" \
  --paper-pattern "askari|shu"
```

Atalho pratico para as perguntas recorrentes da branch `shape`:

```bash
scripts/knowledge/query_shape.sh shape
scripts/knowledge/query_shape.sh twin
scripts/knowledge/query_shape.sh e2e
scripts/knowledge/query_shape.sh askari
scripts/knowledge/query_shape.sh shu
```

## Por que esta camada existe

Docling resolve parsing.

Esta pasta resolve retrieval:

- chunking
- embeddings
- indexacao incremental
- busca local para alimentar outra IA
