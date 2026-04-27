# Handoff - 2026-04-27 - S39B Conservative Next Step

## Contexto

- projeto: `cVAe_2026`
- data de referencia: `2026-04-27`
- branch local ativa: `feat/mdn-g5-recovery-explore-remote`
- objetivo cientifico final:
  - chegar a `12/12 pass` no protocolo final
  - com `stat_tests` ativos

## Estado Cientifico Atual

- baseline oficial:
  - `outputs/exp_20260416_005055`
  - resultado:
    - `7 pass / 0 partial / 5 fail`
  - campeao:
    - `G3_lat6_b0p001_fb0p10_lr0p0002_bs16384_anneal100_L128-256-512`

- melhor sinal estrutural recente:
  - `outputs/exp_20260424_122014`
  - preset:
    - `seq_edgegap_targeted_short`
  - campeao:
    - `S39B_edgegap_lowlr_all08_w18_p120`
  - resultado:
    - `0 pass / 11 partial / 1 fail`
  - leitura:
    - estruturalmente muito promissor
    - unico fail remanescente:
      - `dist_0p8m__curr_300mA`
    - ainda nao substitui o baseline porque ficou sem `stat_tests`

## Evidencia Contra `200k/exp`

- `outputs/exp_20260425_162033`
  - candidato:
    - `S39B_edgegap_lowlr_all08_w18_p120`
  - config:
    - `max_samples_per_exp=200000`
    - `stat_tests=true`
    - `stat_mode=quick`
  - resultado:
    - `0 pass / 0 partial / 12 fail`

- `outputs/exp_20260425_180613`
  - candidato:
    - baseline oficial `G3_lat6_b0p001_fb0p10_lr0p0002_bs16384_anneal100_L128-256-512`
  - config:
    - `max_samples_per_exp=200000`
    - `stat_tests=true`
    - `stat_mode=quick`
  - resultado:
    - `2 pass / 0 partial / 10 fail`

Conclusao:

- `200k/exp` pode servir como triagem operacional leve
- `200k/exp` nao esta servindo como proxy cientifico confiavel
- o problema nao foi exclusivo da linha `S39B`

## Decisao Atual

Seguir uma estrategia conservadora.

Isto significa:

1. manter `exp_20260416_005055` como baseline oficial
2. usar `exp_20260424_122014` como melhor evidencia estrutural recente
3. nao abrir novo grid
4. nao insistir em `200k/exp` para decisao cientifica
5. esperar uma janela segura de GPU
6. rerodar apenas `S39B` com full data + `stat_tests`

## Proxima Rodada Recomendada

Rodar apenas:

- `S39B_edgegap_lowlr_all08_w18_p120`

Com:

- full data
- `stat_tests` ativos
- `stat_mode=quick`
- sem abrir novo grid

Comando:

```bash
python3 -u -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/protocol_default.json \
  --train_once_eval_all \
  --grid_tag S39B_edgegap_lowlr_all08_w18_p120 \
  --no_data_reduction \
  --stat_tests --stat_mode quick
```

## O Que Nao Fazer Agora

- nao abrir outro grid amplo
- nao usar `max_samples_per_exp=200000` como criterio de promocao
- nao promover `exp_20260424_122014` a baseline oficial sem `stat_tests`

## Arquivos Mais Importantes

- `docs/operations/TRAINING_STATUS_AND_PROJECT_OVERVIEW.md`
- `docs/operations/FAILURE_DEEP_DIVE_EXP_20260416_005055.md`
- `docs/operations/HANDOFF_2026_04_24_EDGEGAP_TARGETED.md`
- `src/training/grid_plan.py`
