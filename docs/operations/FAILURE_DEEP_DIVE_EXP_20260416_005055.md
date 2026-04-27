# Failure Deep Dive - exp_20260416_005055

Data de referencia: 2026-04-23.

Este documento detalha os 5 regimes que ainda falham no baseline oficial
`outputs/exp_20260416_005055` e transforma o resultado em uma decisao pratica
para a proxima rodada de trabalho.

## Conclusao Curta

Nao vale a pena rodar outro grid amplo agora.

O baseline ja passa os gates diretos de sinal em todos os regimes:

- `G1`: 12/12 pass.
- `G2`: 12/12 pass.
- `G4`: 12/12 pass.

As falhas restantes se dividem em dois grupos:

- Grupo A: `0.8m / 100mA` e `0.8m / 300mA` falham em `G3`, `G5` e `G6`.
- Grupo B: `0.8m / 500mA`, `1.0m / 100mA` e `1.5m / 500mA` falham apenas em `G6`.

A proxima acao recomendada é atacar primeiro o Grupo A, porque ele indica erro
estrutural de forma/covariancia residual, nao apenas rejeicao estatistica formal.

## Thresholds Relevantes

Os gates sao calculados em `src/evaluation/validation_summary.py`.

| Gate | Regra principal | Threshold |
|---|---|---:|
| `G1` | erro relativo de EVM | `< 0.10` |
| `G2` | erro relativo de SNR | `< 0.10` |
| `G3` | `mean_rel_sigma` e `cov_rel_var` | `< 0.10` e `< 0.20` |
| `G4` | `delta_psd_l2` | `< 0.25` |
| `G5` | skew, kurtosis e JB relativo | `< 0.30`, `< 1.25`, `< 0.20` |
| `G6` | q-values MMD e Energy | `> 0.05` |

## Fails Do Baseline

| Regime | Gates que falham | Causa imediata | Leitura |
|---|---|---|---|
| `dist_0p8m__curr_100mA` | `G3`, `G5`, `G6` | `cov_rel_var=0.2194`, `delta_jb_stat_rel=0.8116`, `mmd_q=0.0299`, `energy_q=0.0299` | Falha estrutural mais forte. A forma residual esta errada mesmo com EVM/SNR bons. |
| `dist_0p8m__curr_300mA` | `G3`, `G5`, `G6` | `cov_rel_var=0.2322`, `delta_jb_stat_rel=0.3247`, `mmd_q=0.0299`, `energy_q=0.0299` | Parecido com 100mA, mas menos severo em JB. Ainda e problema de covariancia/forma. |
| `dist_0p8m__curr_500mA` | `G6` | `mmd_q=0.0398`; `energy_q=0.0955` passa | Quase passa. Estrutura residual passa; falta robustez no teste MMD. |
| `dist_1m__curr_100mA` | `G6` | `mmd_q=0.0299`, `energy_q=0.0299` | Estrutura passa; falha e estatistica formal. |
| `dist_1p5m__curr_500mA` | `G6` | `mmd_q=0.0299`, `energy_q=0.0299` | Estrutura passa; falha e estatistica formal. |

## Diagnostico Por Grupo

### Grupo A - Fails Estruturais Em 0.8m Baixa Corrente

Regimes:

- `dist_0p8m__curr_100mA`
- `dist_0p8m__curr_300mA`

Padrao observado:

- `G1/G2` passam, entao o nivel medio de qualidade de sinal nao e o gargalo.
- `G4` passa, entao a distancia PSD agregada nao e o gargalo principal.
- `G3` falha por `cvae_cov_rel_var` acima de `0.20`.
- `G5` falha por `delta_jb_stat_rel` acima de `0.20`.
- `G6` tambem falha, provavelmente como consequencia da forma residual errada.

Interpretacao:

O modelo esta acertando medias gerais, EVM, SNR e PSD, mas ainda nao reproduz a
geometria residual desses regimes de curta distancia e baixa corrente. A falha
nao parece ser resolvida por mais ranking, porque o ranking ja penaliza `0.8m`,
`G5`, kurtosis e PSD.

### Grupo B - Fails Estatisticos G6-Only

Regimes:

- `dist_0p8m__curr_500mA`
- `dist_1m__curr_100mA`
- `dist_1p5m__curr_500mA`

Padrao observado:

- `G1` a `G5` passam.
- `G6` falha por q-value.
- `0.8m / 500mA` esta perto: `energy_q=0.0955` passa, mas `mmd_q=0.0398` falha.

Interpretacao:

Esses regimes nao devem comandar o proximo grande desenho de arquitetura. Eles
podem melhorar como efeito colateral se o modelo reproduzir melhor a familia de
residuos, mas nao sao o gargalo cientifico primario.

## O Que Nao Fazer Agora

Nao repetir imediatamente:

- outro `protocol_faceoff_short` sem mudar modelagem;
- treino somente nos fails;
- treino focado demais em `0.8m` sem ancoras;
- grid amplo variando hiperparametros classicos sem hipotese nova.

Essas rotas ja foram testadas e ficaram abaixo do baseline.

## Proximo Experimento Recomendado

Prioridade: experimento curto de modelagem/loss focado no Grupo A, preservando
contexto global.

Hipotese:

O gargalo em `0.8m / 100mA` e `0.8m / 300mA` esta na forma residual e na
covariancia condicional. Portanto, a proxima rodada deve aumentar pressao sobre
`cov_rel_var` e `delta_jb_stat_rel` sem destruir `G1/G2/G4`.

Desenho sugerido:

- manter treino global, nao fail-only;
- avaliar sempre nos 12 regimes;
- usar uma rodada curta com poucos candidatos;
- incluir o campeao atual como controle;
- testar uma variacao de loss/modelagem voltada a forma residual;
- promover candidato apenas se melhorar Grupo A sem perder muitos passes globais.

Aceitacao minima para continuar:

- `0.8m / 100mA` ou `0.8m / 300mA` precisa passar ao menos `G3` ou `G5`;
- o total global nao deve cair abaixo de `7 pass / 0 partial / 5 fail`;
- nenhum regime que ja passava deve abrir falha em `G1` ou `G2`;
- `G4` deve permanecer 12/12 pass.

## Plano Pratico Imediato

1. Congelar `outputs/exp_20260416_005055` como baseline oficial.
2. Criar uma variante pequena de grid com controle + 2 ou 3 candidatos novos.
3. Direcionar a variante para covariancia/forma residual em `0.8m`.
4. Rodar primeiro um treino curto no container GPU.
5. Comparar somente contra os criterios deste documento antes de escalar.

## Comandos Uteis

Resumo do baseline:

```bash
python scripts/analysis/summarize_experiment.py outputs/exp_20260416_005055
```

Ver os fails diretamente:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/exp_20260416_005055/tables/summary_by_regime.csv")
cols = [
    "regime_id", "validation_status",
    "gate_g3", "gate_g5", "gate_g6",
    "cvae_cov_rel_var", "delta_jb_stat_rel",
    "stat_mmd_qval", "stat_energy_qval",
]
print(df[df["validation_status"].eq("fail")][cols].to_string(index=False))
PY
```
