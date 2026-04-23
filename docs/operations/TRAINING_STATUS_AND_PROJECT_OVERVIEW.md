# Training Status And Project Overview

Data de referencia: 2026-04-23.

Este documento consolida o que foi feito nos treinamentos recentes, qual e o
melhor resultado atual, o que significa "campeao" no projeto e como as partes
principais do repositorio se conectam.

## Resumo Executivo

O melhor ponto conhecido ate agora e:

- experimento: `outputs/exp_20260416_005055`
- campeao: `G3_lat6_b0p001_fb0p10_lr0p0002_bs16384_anneal100_L128-256-512`
- resultado: `7 pass / 0 partial / 5 fail` em 12 regimes
- origem do modelo: `outputs/exp_20260415_161029/train`
- comando conceitual: reavaliacao do modelo treinado anteriormente com `--stat_tests --stat_mode quick`

As tentativas posteriores testaram treino focado, protocolos intermediarios e
novos criterios de ranking. Nenhuma superou esse baseline. A recomendacao atual
e congelar `exp_20260416_005055` como baseline oficial e parar de relancar
variações de `protocol_faceoff_short` ate haver uma mudanca de modelagem/loss ou
uma analise mais especifica dos regimes de `0.8m`.

## O Que Este Projeto Faz

O projeto constrói um gemeo digital de canal VLC (Visible Light Communication)
orientado a dados.

A tarefa cientifica central e aprender a distribuicao condicional:

```text
p(y | x, d, c)
```

Onde:

- `x`: amostra I/Q transmitida em banda base.
- `y`: amostra I/Q recebida depois do canal fisico.
- `d`: distancia LED-fotodetector.
- `c`: corrente de acionamento do LED.

O objetivo nao e apenas prever uma media. O objetivo e gerar amostras recebidas
que preservem a distribuicao estatistica do canal fisico: variancia, caudas,
assimetria, kurtosis, espectro, autocorrelacao e testes estatisticos de
fidelidade.

## O Que E Um cVAE Aqui

O cVAE e um Autoencoder Variacional Condicional. Neste projeto, ele aprende uma
representacao latente `z` condicionada por `x`, `d` e `c`.

Durante o treino:

- o encoder observa `x`, `d`, `c` e `y`;
- o prior aprende uma distribuicao latente condicionada em `x`, `d`, `c`;
- o decoder gera uma saida compatível com `y`;
- a loss combina reconstrucao com regularizacao KL.

Durante a geracao/avaliacao:

- o modelo recebe `x`, `d`, `c`;
- amostra `z` do prior condicional;
- gera uma predicao/amostra `y_pred`;
- o protocolo compara `y_pred` contra as medicoes reais do regime.

## Principais Partes Do Repositorio

```text
configs/
```

Define protocolos e presets de grid. Exemplos importantes:

- `configs/protocol_default.json`: protocolo reduzido canonico de 12 regimes.
- `configs/regimes_fail_focus_5.json`: recorte com os 5 regimes problemáticos.
- `configs/regimes_0p8m_plus_anchors_8.json`: recorte intermediario com todos os `0.8m` mais ancoras.

```text
data/
```

Contem datasets versionados por Git LFS. O dataset usado nos treinos recentes foi:

```text
data/dataset_fullsquare_organized
```

Ele esta organizado por distancia e corrente, por exemplo:

```text
dist_0.8m/curr_100mA/
dist_1.0m/curr_300mA/
dist_1.5m/curr_700mA/
```

```text
src/protocol/
```

Contem o runner canonico de protocolo. O entrypoint principal e:

```bash
python -m src.protocol.run
```

Ele orquestra:

- carregamento do protocolo;
- treino global ou por regime;
- avaliacao por regime;
- geracao de `summary_by_regime.csv`;
- geracao de `protocol_leaderboard.csv`;
- escrita de `manifest.json`.

```text
src/training/
```

Contem o fluxo de treino, grid search e ranking de candidatos.

Arquivos importantes:

- `src/training/grid_plan.py`: define os candidatos do grid e presets como `protocol_faceoff_short`.
- `src/training/gridsearch.py`: treina cada candidato, calcula `score_v2`, aplica ranking e salva o melhor modelo.
- `src/training/pipeline.py`: integra dados, treino, grid search e artefatos.

```text
src/models/
```

Contem as arquiteturas e callbacks.

Arquivos importantes:

- `src/models/cvae.py`: modelos cVAE ponto-a-ponto.
- `src/models/cvae_sequence.py`: carregamento/compatibilidade dos modelos, incluindo fallback de `safe_mode=False` para Lambda confiavel.
- `src/models/callbacks.py`: diagnosticos por regime e mini reanalysis durante o treino.

```text
src/evaluation/
```

Contem validacao cientifica, tabelas de resumo, plots e testes estatisticos.

O protocolo final usa gates `G1` a `G6`, descritos abaixo.

```text
outputs/
```

Contem os experimentos executados. Cada `exp_*` tem, quando completo:

- `manifest.json`: metadados, status, campeao e configuracoes.
- `train/`: artefatos de treino e modelo.
- `tables/summary_by_regime.csv`: veredito por regime.
- `tables/protocol_leaderboard.csv`: ranking final do protocolo.
- `tables/stat_fidelity_by_regime.csv`: testes estatisticos quando `--stat_tests` esta ativo.
- `plots/`: heatmaps, dashboards e figuras de analise.

```text
scripts/
```

Contem scripts auxiliares. Exemplos:

- `scripts/ops/`: operacao de container/tmux.
- `scripts/analysis/summarize_experiment.py`: resumo de experimentos.

```text
docker/
```

Contem suporte ao ambiente containerizado com TensorFlow/GPU.

```text
tests/
```

Contem testes unitarios e de integracao usados para proteger o pipeline.

## O Que E O "Campeao"

"Campeao" e o candidato escolhido como melhor modelo em uma etapa de selecao.
Existem dois niveis importantes:

1. Campeao do grid de treino.
2. Campeao do protocolo final.

No modo `train_once_eval_all`, o fluxo e:

1. O grid search treina varios candidatos.
2. O ranking escolhe um candidato vencedor.
3. Esse vencedor e salvo como:

```text
train/models/best_model_full.keras
train/models/best_decoder.keras
train/models/best_prior_net.keras
```

4. O protocolo reusa esse modelo global e avalia todos os regimes.
5. O `protocol_leaderboard.csv` registra o campeao final e o placar.

O nome do campeao e um `grid tag`. Exemplo:

```text
G3_lat6_b0p001_fb0p10_lr0p0002_bs16384_anneal100_L128-256-512
```

Esse nome codifica hiperparametros:

- `G3`: familia/grupo do grid.
- `lat6`: dimensao latente 6.
- `b0p001`: beta da KL igual a 0.001.
- `fb0p10`: free bits 0.10.
- `lr0p0002`: learning rate 0.0002.
- `bs16384`: batch size 16384.
- `anneal100`: KL annealing por 100 epocas.
- `L128-256-512`: camadas internas 128, 256, 512.

Importante: o campeao do grid pode parecer bom no treino e ainda falhar no
protocolo cientifico. Foi exatamente isso que observamos em alguns runs recentes.

## Gates Do Protocolo

O protocolo avalia cada regime com gates `G1` a `G6`.

De forma pratica:

- `G1/G2`: sanidade de metricas principais de sinal, como EVM/SNR e proximidade geral.
- `G3`: fidelidade distributiva de baixa ordem, como media/covariancia.
- `G4`: fidelidade espectral/autocorrelacao.
- `G5`: fidelidade de forma da distribuicao, especialmente caudas, skew/kurtosis e residuos.
- `G6`: fidelidade estatistica por testes como MMD, Energy e criterios PSD.

Um regime so vira `pass` quando os gates necessarios passam. Se faltam testes
estatisticos, o regime pode ficar como `partial`; quando `--stat_tests` e
executado, `G6` passa a decidir de forma mais completa.

## Linha Do Tempo Dos Treinamentos Recentes

| Run | Papel | Status | Campeao | Resultado |
|---|---|---|---|---|
| `exp_20260415_161029` | Primeiro treino completo de 12 regimes | `completed` | `G3_lat6_b0p001_fb0p10_lr0p0002_bs16384_anneal100_L128-256-512` | `0 pass / 10 partial / 2 fail` |
| `exp_20260416_005055` | Reavaliacao do campeao anterior com `stat_tests` | `completed` | `G3_lat6_b0p001_fb0p10_lr0p0002_bs16384_anneal100_L128-256-512` | `7 pass / 0 partial / 5 fail` |
| `exp_20260416_123729` | Treino focado nos 5 fails | `completed` | `G3_lat6_b0p001_fb0p10_lr0p0002_bs16384_anneal100_L128-256-512` | `0 pass / 0 partial / 5 fail` |
| `exp_20260416_204608` | Treino intermediario com `0.8m` + ancoras | `completed` | `G1_lat8_b0p003_fb0p10_do0p0_lr0p0003_L128-256-512` | `3 pass / 0 partial / 5 fail` |
| `exp_20260417_130146` | Treino 12 regimes com mini protocol defaults | `completed` | `G3_lat4_b0p002_fb0p10_lr0p0002_bs16384_anneal60_L128-256-512` | `6 pass / 0 partial / 6 fail` |
| `exp_20260417_232746` | Faceoff curto antes do ranking reforcado | `completed` | `G3_lat8_b0p003_fb0p10_lr0p0002_bs16384_anneal80_L128-256-512` | `4 pass / 0 partial / 8 fail` |
| `exp_20260418_021327` | Faceoff curto com ranking reforcado | `completed` | `G3_lat6_b0p002_fb0p10_lr0p0002_bs8192_anneal80_L128-256-512` | `3 pass / 0 partial / 9 fail` |

## Melhor Baseline Atual

O baseline oficial recomendado e:

```text
outputs/exp_20260416_005055
```

Ele reavaliou o modelo treinado em:

```text
outputs/exp_20260415_161029/train
```

Placar por regime:

| Regime | Status | Observacao |
|---|---|---|
| `dist_0p8m__curr_100mA` | `fail` | Falha em `G3`, `G5`, `G6` |
| `dist_0p8m__curr_300mA` | `fail` | Falha em `G3`, `G5`, `G6` |
| `dist_0p8m__curr_500mA` | `fail` | Passa `G3/G5`, falha `G6` |
| `dist_0p8m__curr_700mA` | `pass` | Melhor caso em `0.8m` |
| `dist_1m__curr_100mA` | `fail` | Falha `G6` |
| `dist_1m__curr_300mA` | `pass` | Regime estavel |
| `dist_1m__curr_500mA` | `pass` | Regime estavel |
| `dist_1m__curr_700mA` | `pass` | Regime estavel |
| `dist_1p5m__curr_100mA` | `pass` | Regime estavel |
| `dist_1p5m__curr_300mA` | `pass` | Regime estavel |
| `dist_1p5m__curr_500mA` | `fail` | Falha `G6` |
| `dist_1p5m__curr_700mA` | `pass` | Regime estavel |

## O Que Aprendemos

1. Ativar `stat_tests` foi essencial.

O primeiro treino parecia `0 pass / 10 partial / 2 fail`, mas os partials eram
em grande parte falta de `G6`. A reavaliacao com testes estatisticos transformou
isso em `7 pass / 5 fail`, que e o melhor estado conhecido.

2. Treinar somente nos fails nao ajudou.

O run `exp_20260416_123729` treinou somente nos 5 regimes problemáticos e
terminou `0/5 pass`. Isso sugere que remover contexto global prejudica o modelo.

3. Recortar demais para `0.8m` tambem piorou.

O run intermediario `exp_20260416_204608` fez `3/8 pass` e degradou regimes que
antes passavam. O problema nao parece ser resolvido apenas aumentando foco no
subconjunto dificil.

4. Ranking melhor ajuda o processo, mas nao resolve o problema cientifico.

Criamos `mini_protocol_v1`, `protocol_faceoff_short` e depois reforcamos o
ranking para penalizar falhas em `0.8m`, `G5`, kurtosis e PSD. Mesmo assim, o
melhor resultado posterior ficou abaixo do baseline.

5. O gargalo real esta em `0.8m`, principalmente baixa corrente.

Os regimes mais difíceis sao:

- `0.8m / 100mA`
- `0.8m / 300mA`
- `0.8m / 500mA`

Os dois primeiros quebram de forma estrutural em `G3`, `G5` e `G6`.

## Mudancas De Codigo Feitas Nesta Rodada

1. Loader Keras mais robusto.

Arquivo:

```text
src/models/cvae_sequence.py
```

O carregamento agora tenta o modo seguro primeiro. Se o Keras bloquear uma
`Lambda layer` confiavel por `safe_mode`, ele repete o carregamento com
`safe_mode=False`. Isso destravou a avaliacao dos modelos `.keras`.

2. Manifesto com status de interrupcao.

Arquivos:

```text
src/protocol/run.py
src/protocol/experiment_tracking.py
```

Se o protocolo for interrompido manualmente, o `manifest.json` passa a registrar
`run_status: "interrupted"` em vez de ficar preso em `running`.

3. Preset curto de confronto.

Arquivo:

```text
src/training/grid_plan.py
```

Foi criado `protocol_faceoff_short`, um grid de 8 candidatos para comparar:

- familia do baseline `G3_lat6_b0p001...`;
- familia nova `G3_lat4...`;
- controle `G0`;
- falso positivo conhecido `G1/G3 lat8`.

4. Ranking de grid revisado.

Arquivos:

```text
src/training/gridsearch.py
src/models/callbacks.py
```

O `mini_protocol_v1` passou a considerar mais cedo:

- falhas totais;
- falhas em `0.8m`;
- falhas `G5` em `0.8m`;
- falhas `G5` gerais;
- falhas `G6`;
- `delta_kurt_l2`;
- `delta_psd_l2`;
- cobertura;
- JB;
- `score_v2` apenas como desempate tardio.

Essa mudanca e util como infraestrutura, mas ainda nao produziu novo campeao
melhor que o baseline.

## Estado Operacional Atual

Nao ha treino rodando agora.

Na ultima verificacao:

- a sessao `tmux` antiga `cvae_eduardo` nao existia mais;
- havia apenas uma sessao `tmux` chamada `eduardo`;
- `docker ps` nao mostrava container ativo;
- os artefatos do ultimo run estavam completos.

Para voltar ao ambiente de container/GPU, o caminho operacional e:

```bash
cd ~/cVAe_2026/scripts/ops
./run_tf25_gpu.sh
./enter_tf25_gpu.sh
```

## Recomendacao Atual

Nao rodar outro treino grande imediatamente.

Proximo passo recomendado:

1. Congelar `exp_20260416_005055` como baseline oficial.
2. Usar esse baseline como referencia em qualquer paper/relatorio/proximo PR.
3. Fazer analise especifica dos regimes `0.8m / 100mA` e `0.8m / 300mA`.
4. Propor mudanca de modelagem/loss para `G3/G5/G6`, em vez de apenas mudar ranking.
5. So depois abrir nova rodada curta para testar a hipotese.

Hipoteses candidatas para a proxima fase:

- loss auxiliar para caudas/kurtosis/residual shape;
- tratamento condicional mais forte para distancia/corrente;
- amostragem/ponderacao de treino mais sutil, sem remover contexto global;
- arquitetura que preserve melhor variancia e caudas em baixa corrente;
- avaliacao isolada do motivo de `0.8m` degradar quando o ranking escolhe modelos aparentemente estaveis.

## Comandos Uteis

Resumo de um experimento:

```bash
python scripts/analysis/summarize_experiment.py outputs/exp_20260416_005055
```

Reavaliar o baseline oficial:

```bash
python3 -u -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/protocol_default.json \
  --train_once_eval_all \
  --reuse_model_run_dir outputs/exp_20260415_161029/train \
  --stat_tests --stat_mode quick
```

Inspecionar tabelas principais:

```bash
less outputs/exp_20260416_005055/tables/summary_by_regime.csv
less outputs/exp_20260416_005055/tables/stat_fidelity_by_regime.csv
less outputs/exp_20260416_005055/tables/protocol_leaderboard.csv
```

