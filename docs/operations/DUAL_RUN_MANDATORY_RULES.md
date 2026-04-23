# Regras Obrigatorias Para Dois Runs

Memo curto e canonico para quando houver duvida sobre o que e permitido ao
manter `2` runs em paralelo na mesma rotina operacional.

Este texto nao substitui os guias longos. Ele consolida, em um unico lugar, as
regras espalhadas nos documentos de infraestrutura, slots canonicos e screening
cientifico.

## Escopo

Este memo vale para:

- uma mesma maquina
- uma mesma GPU visivel para os slots ativos
- dois slots operacionais quentes

Fontes canonicas usadas:

- [../active/INFRA_GUIDE.md](../active/INFRA_GUIDE.md)
- [../active/MULTI_PC_WORKFLOW.md](../active/MULTI_PC_WORKFLOW.md)
- [CANONICAL_DUAL_RUN_STANDARD.md](CANONICAL_DUAL_RUN_STANDARD.md)
- [../../../cVAe_2026_shape_fullcircle/knowledge/syntheses/support_hyperparameter_scientific_screening_2026-04-08.md](../../../cVAe_2026_shape_fullcircle/knowledge/syntheses/support_hyperparameter_scientific_screening_2026-04-08.md)
- [../active/WORKING_STATE.md](../active/WORKING_STATE.md)

## O Que Ja Estava Explicito Nas Fontes

As fontes ja eram explicitas em quatro pontos:

1. manter apenas `2` slots quentes por maquina
2. usar nomes canonicos de `tmux`, container e worktree
3. usar `OUTPUT_BASE` distinto para runs paralelos
4. nao assumir que usuario Unix, Docker ou worktree isolam a GPU

O ponto que este memo torna explicito e a consequencia operacional disso:

- em host de GPU unica, `2` slots nao significam `2` treinos pesados seguros

## Regras Obrigatorias

### 1. Apenas dois slots quentes

- Manter no maximo `2` slots ativos por maquina.
- Nao abrir sessoes e containers extras para a mesma finalidade.
- No fluxo atual, usar os slots canonicos existentes em vez de nomes ad hoc.

### 2. Um worktree por linha

- Cada linha cientifica deve ter seu proprio worktree nomeado.
- Nao trocar de branch dentro de um worktree nomeado.
- Nao usar o repo neutro para editar, treinar ou comitar.

### 3. Lancamento canonico obrigatorio

- Antes de lancar qualquer run, verificar o estado do ambiente.
- Lancar pelo helper canonico do slot, nao por comandos improvisados em shells
  soltos.
- Cada run deve ter `run_tag`, `RUN_STAMP` e log canonico.

Comando minimo de verificacao:

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/canonical_dual_run.sh status
```

### 4. Separacao obrigatoria de artefatos

- Todo run paralelo deve usar `OUTPUT_BASE` distinto.
- O log deve ficar em `outputs/_launch_logs/<RUN_STAMP>_<slot>_<run_tag>.log`.
- Sem essa separacao, o run pode ate existir operacionalmente, mas nao deve ser
  tratado como registro cientifico limpo.

### 5. GPU e sempre recurso compartilhado

- O host continua sendo um so.
- A GPU continua sendo uma so.
- Usuario Unix, Docker e worktree isolam arquivos e processos, mas nao isolam o
  uso real da GPU.

Consequencia pratica:

- colisao de memoria, lentidao variavel e falhas CUDA continuam possiveis mesmo
  quando os slots estao "organizados"

### 6. Politica consolidada de carga por slot

Esta e a leitura operacional consolidada a partir das fontes.

Permitido como arranjo canonico:

- `1` job pesado + `1` job leve
- `2` jobs leves

Nao tratar como arranjo canonico em host de GPU unica:

- `2` jobs pesados de treino global + avaliacao multi-regime ao mesmo tempo

Para este memo:

- job pesado: `train_once_eval_all`, full data, grids longos, inferencia
  estocastica multi-regime, ou qualquer run que combine treino longo com
  avaliacao ampla
- job leve: docs, analise tabular, plots, reavaliacao pontual, dry-run,
  verificacao curta, ou screen reduzido/capado

Se dois jobs pesados precisarem ocorrer em paralelo, o caminho recomendado e:

- outra maquina
- outra GPU
- ou janela separada no tempo

### 7. Se um paralelo falhar por GPU, nao canonizar

Se um dos slots falhar com erro de GPU, CUDA ou inferencia:

- nao usar o resultado como evidencia cientifica final
- manter o artefato apenas como registro operacional
- relancar o candidato sozinho antes de comparar com a linha vencedora

Caso de referencia:

- [DUAL_RUN_HEAVY_LOAD_POSTMORTEM_2026-04-22.md](DUAL_RUN_HEAVY_LOAD_POSTMORTEM_2026-04-22.md)

### 8. Registrar contexto minimo

Cada run serio deve deixar registrado:

- host
- worktree
- branch
- slot usado
- `run_tag`
- `OUTPUT_BASE`
- caminho do log

## Checklist Curto Antes De Rodar

1. Confirmar que so existem `2` slots quentes.
2. Confirmar qual linha esta em cada worktree.
3. Rodar `status` do helper canonico.
4. Definir `run_tag` curto e sem ambiguidade.
5. Garantir `OUTPUT_BASE` distinto.
6. Decidir se o segundo slot sera leve ou pesado.
7. Se ambos forem pesados, mover um deles para outra maquina ou outra janela.

## Comando Padrao Para Acompanhar O Log

Quando o run for lancado corretamente, acompanhe por:

```bash
tail -f /home/rodrigo/cVAe_2026_shape_fullcircle/outputs/_launch_logs/<RUN_STAMP>_<slot>_<run_tag>.log
```

## Regra De Ouro

Se houver duvida entre "caber em dois slots" e "ser seguro para a GPU", vale a
segunda interpretacao:

- dois slots organizam a rotina
- nao garantem seguranca para dois jobs pesados na mesma GPU

Leitura aprofundada do caso que motivou esta regra:

- [DUAL_RUN_HEAVY_LOAD_POSTMORTEM_2026-04-22.md](DUAL_RUN_HEAVY_LOAD_POSTMORTEM_2026-04-22.md)
