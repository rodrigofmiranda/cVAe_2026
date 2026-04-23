# Benchmark `16QAM`

Primeiro benchmark transversal oficial do projeto.

## Papel

- comparar `full_square` e `full_circle`
- validar generalizacao fora do dataset principal de shaping
- servir de molde para `4QAM` e futuras modulacoes

## Artefatos esperados

- manifesto por execucao
- tabela longa por regime
- resumo por candidato
- tabela de vitorias crossline
- leitura curta consolidada

## Camadas associadas

- `scripts/benchmarks/modulations/16qam/`
- `knowledge/syntheses/benchmarks/modulations/16qam/`
- `Tese/06_validacao_do_gemeo/16qam/`

## Onde ficam os resultados

Os resultados nao ficam em um deposito central `outputs/benchmarks/...`.

Cada avaliacao `16QAM` deve ficar junto da arquitetura/candidato que foi
testado:

```text
/home/rodrigo/cVAe_2026_full_square/outputs/architectures/<arquitetura>/<candidato>/benchmarks/16qam/
/home/rodrigo/cVAe_2026_full_circle/outputs/architectures/<arquitetura>/<candidato>/benchmarks/16qam/
```

Sumarios crossline podem existir como indices derivados, mas devem apontar para
os manifests locais de cada candidato.
