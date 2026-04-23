# Canonical Output Layout

## Purpose

Definir uma arvore canonica curta para artefatos de saida, principalmente para
avaliacoes `16QAM`, evitando a proliferacao de caminhos longos como
`benchmarks/16qam/eval_...`.

## Canonical Rule

Para artefatos por arquitetura/candidato, o caminho oficial e:

```text
outputs/architectures/<familia>/<candidato>/16qam/<run_tag>/
```

Exemplo:

```text
outputs/architectures/clean_baseline/S27cov_fc_clean_lc0p25_t0p03_lat10/16qam/crossline_20260420_clean/
```

Para comparacoes agregadas entre linhas, o caminho oficial e:

```text
outputs/architectures/_crossline/16qam/<run_tag>/
```

Exemplo:

```text
outputs/architectures/_crossline/16qam/crossline_20260422_plus_soft_radial/
```

## What To Avoid

Nao criar novas arvores no formato:

```text
outputs/architectures/<familia>/<candidato>/benchmarks/16qam/eval_16qam_.../
outputs/analysis/eval_16qam_.../
```

Esses formatos sao considerados legados e devem ser migrados para a arvore
canonica curta.

## Regime-Level Layout

Dentro de cada rodada `16QAM`, a estrutura por regime permanece:

```text
<run_tag>/
  manifest_all_regimes_eval.csv
  manifest_all_regimes_eval.json
  run.log
  dist_1p0m__curr_100mA/
    logs/
    plots/
      champion/
```

A granularidade por regime e preservada; o objetivo desta padronizacao e cortar
niveis intermediarios desnecessarios antes da pasta `16qam`.

## Operational Notes

- Se o launcher souber `familia` e `candidato`, ele deve gravar direto na arvore
  `outputs/architectures/.../<candidato>/16qam/<run_tag>/`.
- Quando um artefato historico vier de um caminho antigo, usar
  `scripts/analysis/migrate_16qam_output_layout.py` para mover os arquivos e
  reescrever os manifests para os caminhos canônicos locais.
- `run_tag` deve ser curto. O prefixo legado `eval_16qam_` deve ser removido.
