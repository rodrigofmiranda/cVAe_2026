# Post-Mortem Operacional: Paralelo Pesado Em GPU Unica

Data do incidente analisado: `2026-04-22`

## Proposito

Registrar, de forma auditavel, por que um par de runs `algo1`/`algo2` que
parecia similar a paralelos anteriores falhou durante a inferencia auxiliar,
mesmo havendo historico recente de `2` runs em paralelo funcionando na mesma
maquina.

Este memo complementa:

- [DUAL_RUN_MANDATORY_RULES.md](DUAL_RUN_MANDATORY_RULES.md)
- [CANONICAL_DUAL_RUN_STANDARD.md](CANONICAL_DUAL_RUN_STANDARD.md)

## Resposta Curta

O problema nao foi "paralelismo proibido" de forma geral.

O problema foi um caso especifico de `2` jobs pesados sobrepostos no mesmo host
de GPU unica:

- ambos em `train_once_eval_all`
- ambos em `full data`
- um deles entrando em avaliacao multi-regime e `quick_predict` estocastico
  enquanto o outro ainda fazia treino global pesado

Os paralelos que tinham dado certo antes eram significativamente mais leves,
porque usavam caps agressivos de treino e validacao.

## Escopo Do Incidente

### Run que permaneceu ativo

- [20260422_130815_algo1_fc_soft_local_full12_exact.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260422_130815_algo1_fc_soft_local_full12_exact.log)

### Run que falhou

- [20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log)

### Runs paralelos anteriores usados como contraste

- [20260421_234722_algo1_full_circle_soft_radial_resolve_bs8192.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260421_234722_algo1_full_circle_soft_radial_resolve_bs8192.log)
- [20260421_234722_algo2_full_circle_soft_radial_resolve_tail98.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260421_234722_algo2_full_circle_soft_radial_resolve_tail98.log)

## O Que Mudou De Verdade

### 1. O paralelo atual era muito mais pesado

No run que falhou, o `algo2` foi disparado sem caps de treino/validacao:

- `8,640,000` pontos de treino
- `2,160,000` pontos de validacao globais

Isso aparece em:

- [20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log#L41)

Nos paralelos que funcionaram, havia caps explicitos:

- `max_samples_per_exp=100000`
- `max_val_samples_per_exp=20000`

Isso aparece em:

- [20260421_234722_algo1_full_circle_soft_radial_resolve_bs8192.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260421_234722_algo1_full_circle_soft_radial_resolve_bs8192.log#L49)

Leitura operacional:

- treino efetivo anterior: `1.2M`
- treino efetivo atual: `8.64M`
- validacao efetiva anterior: `240k`
- validacao efetiva atual: `2.16M`

Portanto, o envelope de carga nao era equivalente.

### 2. O erro ocorreu na inferencia auxiliar, nao na organizacao dos artefatos

O run atual tinha `OUTPUT_BASE` proprio e log canonico proprio:

- [20260422_130815_algo1_fc_soft_local_full12_exact.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260422_130815_algo1_fc_soft_local_full12_exact.log#L19)
- [20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log#L19)

Logo:

- nao houve evidencia de colisao de `OUTPUT_BASE`
- nao houve evidencia de sobrescrita de `manifest` ou `leaderboard`
- nao houve evidencia de erro por convensao de nomes

As regras de separacao de artefatos estavam sendo obedecidas. O problema foi
de carga de GPU/runtime, nao de higiene de pastas.

### 3. A avaliacao canonica do regime chegou a terminar antes da falha

No `algo2`, a sequencia observada foi:

1. carregar o modelo
2. rodar a avaliacao do regime
3. salvar `metricas_globais_reanalysis.json`
4. gerar plots e tabelas
5. entrar no bloco auxiliar de `quick_predict`
6. falhar com `CUDA_ERROR_ILLEGAL_ADDRESS`

Trecho relevante:

- [20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log#L1214)

Isso importa porque mostra que a quebra nao aconteceu logo no inicio do regime,
nem durante a simples escrita de artefatos da avaliacao principal.

### 4. O caminho seq monta as janelas antes do cap de `quick_predict`

No protocolo, o helper de `quick_predict`:

- carrega o modelo e os graphs de inferencia
- detecta o caminho seq
- monta `X_arr_w`
- so depois aplica `max_points`
- e, em modo estocastico, faz `mc_samples` chamadas de `predict`

Trechos:

- [run.py](../../../cVAe_2026_shape_fullcircle/src/protocol/run.py#L1056)
- [run.py](../../../cVAe_2026_shape_fullcircle/src/protocol/run.py#L1238)

No caso que falhou, o log mostra:

- `X_arr_w=(180000, 7, 2)`
- `quick_predict sample cap: 180,000 -> 4,096`

Trecho:

- [20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log#L1265)

Nos casos anteriores bem-sucedidos, esse mesmo passo operava com:

- `val=20,000`
- `quick_predict sample cap: 20,000 -> 4,096`

Trecho:

- [20260421_234722_algo1_full_circle_soft_radial_resolve_bs8192.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260421_234722_algo1_full_circle_soft_radial_resolve_bs8192.log#L1704)

Ou seja, embora o alvo final continue sendo `4096`, o caminho preparatorio antes
do cap ficou muito mais pesado no incidente.

### 5. O erro nao foi `OOM`; foi `illegal address`

O log do run que falhou registra:

- `CUDA_ERROR_ILLEGAL_ADDRESS`
- `Unexpected Event status: 1`

Trecho:

- [20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log](../../../cVAe_2026_shape_fullcircle/outputs/_launch_logs/20260422_130817_algo2_fc_soft_local_bs8192_full12_exact.log#L1267)

Isso significa que a protecao pensada apenas para `OOM` nao cobre este caso.
Aqui o indutor mais plausivel foi:

- sobrecarga concorrente
- fragilidade do runtime CUDA
- estado invalido no backend durante a inferencia auxiliar

### 6. O caminho seq ja era reconhecido como mais delicado em stacks novas

A implementacao documenta explicitamente que a familia seq/GRU possui caminhos
de backend distintos e que GPUs mais novas motivaram cautela extra com o GRU:

- [cvae_sequence.py](../../../cVAe_2026_shape_fullcircle/src/models/cvae_sequence.py#L155)
- [grid_plan.py](../../../cVAe_2026_shape_fullcircle/src/training/grid_plan.py#L2121)

Isso nao prova, por si so, que o modelo estava errado. Mas reforca que a linha
seq em GPUs recentes pode ser operacionalmente mais sensivel do que um caso
pointwise ou capado.

## Comparacao Objetiva

| aspecto | paralelos que funcionaram | paralelo que falhou |
| --- | --- | --- |
| modo | `train_once_eval_all` | `train_once_eval_all` |
| treino global | capado | full data |
| validacao global | capada | full data |
| val por regime | `20,000` | `180,000` antes do cap |
| `N_eval` | `20,000` | `40,000` |
| `quick_predict` | `20,000 -> 4,096` | `180,000 -> 4,096` |
| concorrencia | paralelo leve/moderado | treino pesado + avaliacao pesada |
| desfecho | `Protocol complete` | `CUDA_ERROR_ILLEGAL_ADDRESS` |

## O Que Nao Foi Suportado Pela Evidencia

As evidencias deste incidente nao sustentam as afirmacoes abaixo:

- "qualquer inferencia em paralelo e impossivel"
- "o candidato `bs8192` e cientificamente invalido"
- "o problema foi colisao de pasta"
- "foi um `OOM` simples"

## Conclusao Operacional

O incidente confirma uma leitura mais restritiva, mas mais precisa:

- `2` slots organizam a rotina
- `2` slots nao autorizam `2` jobs pesados na mesma GPU unica

Em particular, este padrao nao deve ser tratado como canonicamente seguro:

- `2` runs `train_once_eval_all`
- `2` runs `full data`
- inferencia estocastica multi-regime em paralelo com treino global ainda ativo

## Regra Consolidada

Em host de GPU unica:

- permitido como arranjo canonico: `1` job pesado + `1` job leve
- permitido como arranjo canonico: `2` jobs leves
- nao tratar como arranjo canonico: `2` jobs pesados ao mesmo tempo

Exemplos de job leve:

- `eval-only`
- reanalise curta
- plots
- tabulacao
- `dry-run`
- screen capado

Exemplos de job pesado:

- `train_once_eval_all`
- full data
- grids longos
- inferencia estocastica multi-regime

## Acao Corretiva Recomendada

Quando um dos dois slots falhar neste contexto:

1. nao usar o artefato como evidencia cientifica final
2. preservar o log como registro operacional
3. relancar o candidato sozinho, preferencialmente em `eval-only` se o treino ja
   existir
4. so depois comparar com a linha vencedora

## Uso Futuro Deste Memo

Este memo deve ser citado quando surgir a pergunta:

- "mas eu ja tinha conseguido rodar dois em paralelo, por que agora falhou?"

A resposta curta e:

- porque antes o paralelo era capado e mais leve
- agora a sobreposicao foi de dois jobs pesados no mesmo host/GPU
