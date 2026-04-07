# Imported Corpora

Esta pasta guarda acervos externos que ainda nao fazem parte da base canonica
de papers do projeto.

## Regra de organizacao

- `knowledge/papers/raw` e `knowledge/papers/parsed`:
  referencias curadas e centrais para o projeto
- `knowledge/imports/...`:
  corpora amplos, caches externos e material de apoio para busca exploratoria

## Fluxo recomendado

1. copiar o corpus bruto para uma subpasta propria
2. normalizar para o layout `document.md + metadata.yaml`
3. indexar em uma base vetorial separada
4. usar esse acervo para descoberta e triangulacao
5. promover apenas o que realmente virar referencia de projeto

## Exemplo atual

- `docling_cache_md/`: cache importado em Markdown recebido por SSH
