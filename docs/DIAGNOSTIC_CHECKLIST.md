# DIAGNOSTIC_CHECKLIST.md

Checklist executavel para diagnosticar o twin cVAE no regime pivo e decidir entre:

- novo grid search
- ajuste de hiperparametros
- mudanca estrutural do modelo

Este documento operacionaliza o [TRAINING_PLAN.md](/workspace/2026/feat_seq_bigru_residual_cvae/TRAINING_PLAN.md).

## 1. Pre-condicoes

- Estar em `/workspace/2026/feat_seq_bigru_residual_cvae`
- Dataset acessivel em `data/dataset_fullsquare_organized`
- Branch e worktree conferidos:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae
git status -sb
git log --oneline -3
```

## 2. Variaveis de sessao

Use estas variaveis para evitar erro manual:

```bash
cd /workspace/2026/feat_seq_bigru_residual_cvae

export DATASET_ROOT=data/dataset_fullsquare_organized
export OUTPUT_BASE=outputs
export PROTOCOL=configs/one_regime_1p0m_300mA.json
```

Para localizar o run mais recente:

```bash
export EXP_DIR="$(ls -td outputs/exp_* | head -1)"
echo "$EXP_DIR"
```

## 3. Etapa A - Sanidade do pipeline

### A1. Suite de testes

```bash
python -m pytest tests -q
```

Registrar:

- numero de testes
- warnings relevantes

### A2. Smoke de protocolo

```bash
python -m src.protocol.run \
  --dataset_root "$DATASET_ROOT" \
  --output_base "$OUTPUT_BASE" \
  --protocol "$PROTOCOL" \
  --max_epochs 1 \
  --max_grids 1 \
  --max_experiments 1 \
  --max_samples_per_exp 2000
```

### A3. Verificacao automatica dos fixes

Atualizar o ponteiro do run:

```bash
export EXP_DIR="$(ls -td outputs/exp_* | head -1)"
echo "$EXP_DIR"
```

```bash
python scripts/verify_pipeline_fixes.py "$EXP_DIR"
```

Ler obrigatoriamente:

- `tables/summary_by_regime.csv`
- `tables/stat_fidelity_by_regime.csv` se existir

Passa nesta etapa se:

- a suite estiver verde
- o protocolo completar
- `verify_pipeline_fixes.py` nao acusar regressao

## 4. Etapa B - Run cientifico do regime pivo

Executar um run completo no regime `1.0m / 300mA`:

```bash
python -m src.protocol.run \
  --dataset_root "$DATASET_ROOT" \
  --output_base "$OUTPUT_BASE" \
  --protocol "$PROTOCOL" \
  --max_grids 5 \
  --stat_tests \
  --stat_mode full \
  --stat_max_n 10000 \
  --stat_n_perm 1000
```

Atualizar o ponteiro do run:

```bash
export EXP_DIR="$(ls -td outputs/exp_* | head -1)"
echo "$EXP_DIR"
```

## 5. Etapa C - Leitura obrigatoria do modelo final

### C1. Verificacao de kurtosis do campeao

```bash
python scripts/check_kurt_pred.py \
  "$EXP_DIR/studies/within_regime/regimes/dist_1m__curr_300mA" \
  --n_eval 5000 \
  --mc_samples 16
```

### C2. Leitura da tabela canonica

Abrir e registrar, no minimo:

- `baseline_evm_pred_%`
- `delta_evm_%`
- `baseline_delta_cov_fro`
- `cvae_delta_cov_fro`
- `baseline_delta_kurt_l2`
- `cvae_delta_kurt_l2`
- `jb_real_log10p_min`
- `cvae_jb_log10p_min`
- `delta_jb_log10p`
- `stat_mmd2`
- `stat_mmd_qval`
- `stat_energy`
- `stat_energy_qval`
- `stat_psd_dist`
- `validation_status`
- `gate_g1` ... `gate_g6`

Comando util:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
import os

exp_dir = Path(os.environ["EXP_DIR"])
df = pd.read_csv(exp_dir / "tables" / "summary_by_regime.csv")
cols = [
    "regime_id",
    "baseline_evm_pred_%",
    "delta_evm_%",
    "baseline_delta_cov_fro",
    "cvae_delta_cov_fro",
    "baseline_delta_kurt_l2",
    "cvae_delta_kurt_l2",
    "jb_real_log10p_min",
    "cvae_jb_log10p_min",
    "delta_jb_log10p",
    "stat_mmd2",
    "stat_mmd_qval",
    "stat_energy",
    "stat_energy_qval",
    "stat_psd_dist",
    "gate_g1",
    "gate_g2",
    "gate_g3",
    "gate_g4",
    "gate_g5",
    "gate_g6",
    "validation_status",
]
print(df[cols].to_string(index=False))
PY
```

### C3. Leitura visual do campeao

Inspecionar em `"$EXP_DIR"/studies/.../regimes/.../plots/` e em `plots/best_grid_model/`.
Os bundles agora sao agrupados por tema e incluem um `README.txt` no diretorio
raiz com a ordem sugerida de leitura.

Abrir primeiro:

- `reports/summary_report.png`
- `core/overlay_constellation.png`
- `core/overlay_residual_delta.png`
- `distribution/psd_residual_delta.png`
- `latent/latent_activity_std_mu_p.png`
- `latent/latent_kl_per_dim.png`
- `training/training_history.png`
- `legacy/analise_completa_vae.png`
- `legacy/comparacao_metricas_principais.png`
- `legacy/radar_comparativo.png`
- `legacy/constellation_overlay.png`

Perguntas obrigatorias:

- o campeao melhora em covariancia e kurtosis contra o baseline?
- o sinal da kurtosis predita combina com o real?
- o PSD do residuo parece plausivel?
- ha multimodalidade artificial evidente na constelacao ou no residuo?

## 6. Etapa D - Diagnostico do treino, latente e prior

### D1. Ler o ranking do grid

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
import os

exp_dir = Path(os.environ["EXP_DIR"])
df = pd.read_csv(exp_dir / "studies" / "within_regime" / "regimes" / "dist_1m__curr_300mA" / "tables" / "gridsearch_results.csv")
cols = [
    "grid_id",
    "tag",
    "score_v2",
    "delta_evm_%",
    "delta_snr_db",
    "delta_cov_fro",
    "delta_skew_l2",
    "delta_kurt_l2",
    "delta_psd_l2",
    "active_dims",
    "latent_dim",
    "kl_q_to_p_total",
    "kl_p_to_N_total",
    "var_mc_gen",
]
cols = [c for c in cols if c in df.columns]
print(df.sort_values("score_v2", ascending=True)[cols].head(10).to_string(index=False))
PY
```

### D2. Inspecionar os plots dos candidatos

Inspecionar pelo menos:

- top-3 por `score_v2`
- melhor candidato em `delta_kurt_l2`
- melhor candidato em `delta_cov_fro`
- melhor candidato em `active_dims`

Arquivos:

- `models/grid_*__tag/plots/reports/summary_report.png`
- `models/grid_*__tag/plots/training/training_history.png`
- `models/grid_*__tag/plots/latent/latent_activity_std_mu_p.png`
- `models/grid_*__tag/plots/latent/latent_kl_qp_per_dim.png`
- `models/grid_*__tag/plots/distribution/psd_residual_delta.png`

Perguntas obrigatorias:

- ha colapso latente (`active_dims` muito baixo)?
- ha `KL` quase nulo por muitas configuracoes?
- o campeao do `score_v2` tambem e competitivo em `delta_kurt_l2` e `delta_cov_fro`?
- existe algum nao-campeao com sinal melhor para distribuicao?

## 7. Etapa E - Comparacao final campeao vs baseline vs real

Ordem de leitura:

1. `summary_by_regime.csv`
2. `gridsearch_results.csv`
3. `plots/best_grid_model/`
4. `models/grid_*__tag/plots/`

Checklist minimo:

- `G1` e `G2` passaram?
- `G3` e `G4` passaram?
- `G5` passou em `delta_jb_log10p < 1.0`?
- `G6` passou em `stat_mmd_qval > 0.05`?
- `validation_status` ficou `pass`, `partial` ou `fail`?

## 8. Decisao: novo grid ou mudanca estrutural

### Rodar novo grid search se:

- o pipeline estiver correto
- houver sinais de colapso latente
- `active_dims`, `KL` e `var_mc_gen` sugerirem calibracao ruim
- existir pelo menos um grid nao-campeao com melhora distribuicional relevante

Hiperparametros prioritarios:

- `beta`
- `free_bits`
- `latent_dim`
- `layer_sizes`
- `dropout`
- `lr`

### Considerar mudanca estrutural se:

- o pipeline estiver correto
- treino e latente parecerem saudaveis
- varios grids falharem da mesma forma em `G3` a `G6`
- o modelo final continuar pior que baseline em covariancia e kurtosis
- `MMD`, `Energy` e `JB` continuarem ruins de forma consistente

## 9. Template minimo de registro

Registrar ao fim de cada rodada:

- commit atual
- comando executado
- `EXP_DIR`
- regime avaliado
- top-3 grids
- valores de `G1` a `G6`
- `validation_status`
- leitura curta:
  - pipeline ok ou nao
  - treino/latente ok ou nao
  - modelo final ok ou nao
  - baseline venceu ou perdeu
  - proximo passo: novo grid ou mudanca estrutural
