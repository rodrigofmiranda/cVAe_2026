# Handoff - 2026-04-24 - Edgegap Targeted Follow-up

## Contexto Atual

- repositorio: `cVAe_2026`
- data de referencia: `2026-04-24`
- branch local ativa: `feat/mdn-g5-recovery-explore-remote`
- worktree usado no container:
  - `/workspace/2026/feat_seq_bigru_residual_cvae`
- objetivo atual:
  - melhorar os fails estruturais em `0.8m`, especialmente:
    - `dist_0p8m__curr_100mA`
    - `dist_0p8m__curr_300mA`
  - sem reabrir grid amplo

## Baseline Oficial

- baseline recomendado:
  - `outputs/exp_20260416_005055`
- resultado:
  - `7 pass / 0 partial / 5 fail`
- documentos principais:
  - `docs/operations/TRAINING_STATUS_AND_PROJECT_OVERVIEW.md`
  - `docs/operations/FAILURE_DEEP_DIVE_EXP_20260416_005055.md`

## Run Recente Que Nao Superou O Baseline

- run:
  - `outputs/exp_20260423_174212`
- preset:
  - `seq_edgegap_recovery_short`
- campeao:
  - `S38D_smplmmd_cov30_t02_tail02-98_e96_emb3_resid_w08x18`
- resultado final:
  - `0 pass / 8 partial / 4 fail`
- leitura pratica:
  - a linha `edgegap_recovery_short` nao superou o baseline
  - nao vale insistir em nova rodada ampla dessa familia

## Mudanca Feita Nesta Sessao

- arquivo alterado:
  - `src/training/grid_plan.py`
- novo preset adicionado:
  - `seq_edgegap_targeted_short`

Esse preset foi desenhado como:

- `1` controle
- `2` variantes novas
- foco em:
  - `lr` menor
  - peso mais cirurgico em `0.8m / 100mA` e `0.8m / 300mA`

### Candidatos Do Novo Preset

- `S39A_edgegap_ctrl_s38d`
  - controle
  - replica a ideia do melhor candidato recente `S38D`
- `S39B_edgegap_lowlr_all08_w18_p120`
  - mesmo foco global em `0.8m`
  - `lr` reduzido
  - `patience=120`
- `S39C_edgegap_lowlr_lowcurr_w24_p120`
  - `lr` reduzido
  - `patience=120`
  - peso extra so para:
    - `dist_0p8m__curr_100mA`
    - `dist_0p8m__curr_300mA`

## Desfecho Das Rodadas Seguintes

- run `outputs/exp_20260424_122014`:
  - preset: `seq_edgegap_targeted_short`
  - campeao: `S39B_edgegap_lowlr_all08_w18_p120`
  - resultado:
    - `0 pass / 11 partial / 1 fail`
  - leitura:
    - melhor sinal estrutural recente
    - unico fail remanescente: `dist_0p8m__curr_300mA`
    - como `stat_tests` nao estavam ativos, os outros regimes ficaram `partial`

- run `outputs/exp_20260425_162033`:
  - reavaliacao do mesmo `S39B` com:
    - `--max_samples_per_exp 200000`
    - `--stat_tests`
    - `--stat_mode quick`
  - resultado:
    - `0 pass / 0 partial / 12 fail`
  - leitura:
    - a versao `200k/exp` foi operacionalmente rapida
    - mas nao sustentou o protocolo cientifico final

## Comando Da Rodada Full Mais Relevante

- comando disparado no container para `exp_20260424_122014`:

```bash
python3 -m src.protocol.run \
  --dataset_root /workspace/2026/feat_seq_bigru_residual_cvae/data/dataset_fullsquare_organized \
  --output_base /workspace/2026/feat_seq_bigru_residual_cvae/outputs \
  --train_once_eval_all \
  --protocol configs/protocol_default.json \
  --grid_preset seq_edgegap_targeted_short \
  --no_data_reduction
```

## Arquivos-Chave Para Ler Ao Retomar

- `outputs/exp_20260424_122014/manifest.json`
- `outputs/exp_20260424_122014/tables/protocol_leaderboard.csv`
- `outputs/exp_20260424_122014/tables/summary_by_regime.csv`
- `outputs/exp_20260424_122014/train/tables/gridsearch_results.csv`
- `outputs/exp_20260425_162033/manifest.json`
- `outputs/exp_20260425_162033/tables/protocol_leaderboard.csv`
- `outputs/exp_20260425_162033/tables/summary_by_regime.csv`
- `outputs/exp_20260416_005055/tables/protocol_leaderboard.csv`
- `outputs/exp_20260423_174212/tables/protocol_leaderboard.csv`

## O Que Fazer Ao Retomar

1. usar `exp_20260416_005055` como baseline oficial
2. usar `exp_20260424_122014` como melhor sinal estrutural recente
3. tratar `exp_20260425_162033` como evidencia negativa para `S39B` com cap `200k`
4. comparar sempre contra:
   - `exp_20260416_005055`
   - `exp_20260423_174212`
   - `exp_20260424_122014`
   - `exp_20260425_162033`
5. decidir promocao so se houver ganho real nos regimes duros com `stat_tests`

## Criterio De Sucesso

- melhorar `dist_0p8m__curr_100mA` e idealmente `dist_0p8m__curr_300mA`
- nao piorar fortemente o placar global
- nao abrir regressao em:
  - `G1`
  - `G2`
  - `G4`
- com `stat_tests` ativos

## Leitura Operacional

O objetivo nao e encontrar um novo grid amplo campeao.

O objetivo real do projeto e chegar a `12/12 pass`, mas esse marco so conta se
vier com `stat_tests` e com estabilidade suficiente para nao parecer apenas um
draw favoravel.

As duas leituras mais importantes agora sao:

- `exp_20260424_122014`:
  - mostrou que a linha `S39B` melhora bastante a estrutura (`G1..G5`)
- `exp_20260425_162033`:
  - mostrou que essa mesma linha nao sustenta o protocolo final quando rodada
    com `max_samples_per_exp=200000`

Portanto, o cap `200k` nao deve ser tratado como substituto direto do treino
full para este candidato.
