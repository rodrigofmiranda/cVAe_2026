# Knowledge Index

Camada local de retrieval sobre os artefatos parseados pelo Docling.

## Papel desta pasta

- `chroma_db/`: base vetorial local
- `index_state.json`: estado incremental do indexador

## Fonte canonica

Esta camada **nao** le PDFs brutos.

Ela indexa o Markdown que ja foi extraido em:

```text
knowledge/papers/parsed/<paper_id>/document.md
```

## Fluxo

1. colocar PDFs em `knowledge/papers/raw/`
2. converter com `scripts/ingest_papers_docling.py`
3. indexar com `scripts/index_knowledge_chroma.py`
4. recuperar trechos relevantes para prompts, notas e sinteses

## Comando

```bash
python scripts/index_knowledge_chroma.py
```

Ou com paths explicitos:

```bash
python scripts/index_knowledge_chroma.py \
  --parsed-dir knowledge/papers/parsed \
  --db-dir knowledge/index/chroma_db \
  --state-path knowledge/index/index_state.json
```

## Por que esta camada existe

Docling resolve parsing.

Esta pasta resolve retrieval:

- chunking
- embeddings
- indexacao incremental
- busca local para alimentar outra IA
