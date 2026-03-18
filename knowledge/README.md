# Knowledge Base

Base local de conhecimento para apoiar pesquisa, prompting e retrieval com uma
segunda IA.

## Estrutura

```text
knowledge/
  manifest.yaml
  papers/
    raw/          # coloque aqui os PDFs originais
    parsed/       # saidas geradas pelo Docling
  index/          # base vetorial local e estado incremental
  notes/          # research cards por paper
  syntheses/      # sinteses tematicas e comparativas
```

## Onde colocar os PDFs

Coloque os artigos originais em:

```text
knowledge/papers/raw/
```

Sugestao de organizacao:

- um PDF por arquivo
- nomes curtos e estaveis
- preferir `autor_ano_tema_curto.pdf`

Exemplos:

- `kingma_2014_autoencoding_variational_bayes.pdf`
- `sonderby_2016_ladder_vae.pdf`
- `residual_decoder_vlc_2025.pdf`

## Fluxo recomendado

1. colocar os PDFs em `knowledge/papers/raw/`
2. rodar o ingest com Docling
3. revisar os arquivos parseados em `knowledge/papers/parsed/`
4. indexar os `document.md` em `knowledge/index/`
5. criar um research card em `knowledge/notes/`
6. consolidar temas em `knowledge/syntheses/`

## Script inicial

Para converter os PDFs com Docling:

```bash
python scripts/ingest_papers_docling.py
```

Ou com diretorios explicitos:

```bash
python scripts/ingest_papers_docling.py \
  --input-dir knowledge/papers/raw \
  --output-dir knowledge/papers/parsed
```

## Script de indexacao

Depois do ingest, indexe o Markdown parseado:

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

## O que o ingest gera

Para cada PDF, o script gera:

- Markdown
- JSON lossless do `DoclingDocument`
- texto plano
- `metadata.yaml` local com referencias para os artefatos

## Limite intencional desta camada

Docling entra aqui como camada de parsing.

Estamos usando Docling porque ele transforma PDFs de pesquisa em artefatos mais
uteis para IA:

- `document.md` para leitura e chunking
- `document.json` para estrutura e proveniencia
- `document.txt` para fallback simples
- OCR, tabelas e layout quando o PDF exigir isso

Ele nao substitui:

- curadoria por paper
- sintese tematica
- retrieval por pergunta

Por isso a pipeline separa:

- `raw`
- `parsed`
- `index`
- `notes`
- `syntheses`
