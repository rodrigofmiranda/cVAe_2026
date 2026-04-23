# Padrao Canonico De Dois Slots

Este documento define o padrao operacional minimo para manter apenas dois runs
em paralelo de forma previsivel.

Use este documento quando a rotina diaria for manter exatamente dois slots
paralelos, sem multiplicar sessoes e containers.

O objetivo e evitar:

- proliferacao de `tmux` e containers
- confusao de worktree
- logs com nomes improvisados
- runs disparados no repo errado

Para a politica consolidada de carga, seguranca de GPU e o que deve ou nao ser
tratado como arranjo canonico em host de GPU unica, ver tambem:

- [DUAL_RUN_MANDATORY_RULES.md](/home/rodrigo/cVAe_2026_mdn_return/docs/operations/DUAL_RUN_MANDATORY_RULES.md)

## 1. Slots Canonicos

Os dois slots permanentes sao:

| slot | sessao `tmux` | container | repo host | repo dentro do container |
| --- | --- | --- | --- | --- |
| `algo1` | `algo1` | `cvae_algo1` | `/home/rodrigo/cVAe_2026_shape_fullcircle` | `/workspace/2026/feat_seq_bigru_residual_cvae` |
| `algo2` | `algo2` | `cvae_algo2` | `/home/rodrigo/cVAe_2026_shape_fullcircle` | `/workspace/2026/feat_seq_bigru_residual_cvae` |

Regras:

- `algo1` e `algo2` sao nomes neutros e permanentes
- os dois podem ser apontados para a mesma linha de pesquisa quando isso fizer
  mais sentido operacionalmente
- nao criar novas sessoes paralelas para a mesma finalidade sem necessidade

## 2. Helper Canonico

O helper operacional central e:

- [canonical_dual_run.sh](/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh)

Ele concentra quatro operacoes:

- `status`
- `enter`
- `send`
- `run`

### 2.1. Ver status

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh status
```

Isso mostra:

- se `algo1` esta ligado
- se `algo2` esta ligado
- quais sao os diretorios padrao de cada slot

### 2.2. Entrar em uma sessao

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh enter algo1
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh enter algo2
```

### 2.3. Enviar um comando simples

O helper sempre troca antes para o diretorio canonico dentro do container.

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh send algo1 -- pwd
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh send algo2 -- git branch --show-current
```

## 3. Comando Padrao De Lancamento

O modo recomendado para iniciar um run novo e `run`.

Formato:

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh run <algo1|algo2> <run_tag> -- <comando...>
```

Exemplo:

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh run algo1 e2_full -- \
  scripts/ops/train_support_final_full.sh e2
```

Outro exemplo:

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh run algo2 eval16qam_disk -- \
  python3 scripts/analysis/run_eval_16qam_all_regimes.py --help
```

Esse wrapper injeta automaticamente no shell do run:

- `RUN_STAMP`
- `RUN_SLOT`
- `RUN_TAG`
- `RUN_LOG_PATH`

E tambem escreve o log em:

```text
outputs/_launch_logs/<RUN_STAMP>_<RUN_SLOT>_<RUN_TAG>.log
```

## 4. Diretorio Padrao De Cada Sessao

Para evitar ambiguidade, todo comando enviado por `send` ou `run` comeca em:

```text
/workspace/2026/feat_seq_bigru_residual_cvae
```

Isso significa:

- os comandos sempre partem do root do repo montado no container
- `outputs/` sera relativo a esse root
- `scripts/ops/...` e `scripts/analysis/...` podem ser chamados sem prefixos

No host, os repos de referencia desta configuracao atual sao:

- `algo1`: `/home/rodrigo/cVAe_2026_shape_fullcircle`
- `algo2`: `/home/rodrigo/cVAe_2026_shape_fullcircle`

## 5. Convencao De Nomes

### 5.1. Nomes fixos de infraestrutura

- sessao `tmux` de `algo1`: `algo1`
- sessao `tmux` de `algo2`: `algo2`
- container de `algo1`: `cvae_algo1`
- container de `algo2`: `cvae_algo2`

Nao usar variacoes ad hoc como:

- `algo1_tmp`
- `algo2_new`
- `rodrigo2`
- `gpu_new`

### 5.2. Tag do run

`run_tag` deve ser curto e sem espacos.

Formato recomendado:

```text
<familia>_<objetivo>[_<variante>]
```

Exemplos bons:

- `e2_full`
- `e3c_16qam`
- `clean_lat10`
- `disk_geom3`
- `ablation_s27`

Caracteres aceitos:

- letras
- numeros
- `_`
- `-`
- `.`

### 5.3. Log canonicamente nomeado

Padrao:

```text
outputs/_launch_logs/<UTCSTAMP>_<slot>_<run_tag>.log
```

Exemplo:

```text
outputs/_launch_logs/20260420_231500_algo1_e2_full.log
```

### 5.4. Nome de output do proprio experimento

Quando o launcher permitir, prefira repetir a mesma semantica no output do run:

```text
outputs/<familia>/<RUN_STAMP>_<run_tag>
```

ou, para analise:

```text
outputs/analysis/<campanha>/<RUN_STAMP>_<run_tag>
```

## 6. Mini Rotina Recomendada

Antes de disparar qualquer coisa:

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh status
```

Para um run serio:

1. escolher o slot certo: `algo1` ou `algo2`
2. escolher um `run_tag` curto e sem ambiguidade
3. lancar pelo helper `run`
4. usar `RUN_STAMP` dentro do proprio comando se precisar montar `output_base`
5. registrar depois o diretório `exp_...` ou o `manifest` canônico no memo da linha

## 7. Leitura Pratica

Regra curta:

- `algo1` e `algo2` sao slots neutros
- o importante e registrar no `run_tag` e no memo qual linha de pesquisa esta
  usando cada um
- lancar sempre pelo helper `run` quando quiser log canonico
- entrar pelo helper `enter` quando quiser acompanhar manualmente
