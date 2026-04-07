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
  imports/        # acervos externos importados sem misturar com a base canonica
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
python scripts/knowledge/ingest_papers_docling.py
```

Ou com diretorios explicitos:

```bash
python scripts/knowledge/ingest_papers_docling.py \
  --input-dir knowledge/papers/raw \
  --output-dir knowledge/papers/parsed
```

## Script de indexacao

Depois do ingest, indexe o Markdown parseado:

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

## Script de consulta

Para buscar trechos relevantes na base local:

```bash
python scripts/knowledge/query_knowledge.py "probabilistic shaping VLC"
```

Para consultar tambem o acervo importado:

```bash
python scripts/knowledge/query_knowledge.py \
  "digital twin nonlinearity visible light communication" \
  --source both \
  --dedupe-paper \
  --show-paths
```

Para o nosso fluxo de `shape`, existe um atalho com presets:

```bash
scripts/knowledge/query_shape.sh shape
scripts/knowledge/query_shape.sh twin
scripts/knowledge/query_shape.sh e2e
scripts/knowledge/query_shape.sh project
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
- `imports`
- `index`
- `notes`
- `syntheses`

## Acervos importados

Quando um corpus externo chegar ja em Markdown, como um cache Docling vindo de
outro ambiente, ele deve entrar primeiro em `knowledge/imports/`.

Fluxo recomendado:

1. copiar o corpus bruto para `knowledge/imports/<nome_do_corpus>/`
2. normalizar para o layout `document.md + metadata.yaml`
3. indexar em uma base vetorial separada
4. promover apenas os papers realmente relevantes para a base canonica

Isso evita misturar:

- os papers de referencia que guiam decisoes do projeto
- e um acervo amplo de apoio para busca exploratoria
