# TRAINING_PLAN.md — Digital Twin de Canal VLC (FullSquare) via cVAE + Baselines
> **Versão:** v3 — incorpora fixes de pipeline, novo formato de dataset (X.npy/Y.npy),
> suite de fidelidade estatística, e diagnósticos empíricos do canal.

---

## 0) Estado atual — o que já está resolvido

### 0.1 Dataset

| Item | Status | Detalhe |
|---|---|---|
| Estrutura de pastas | ✅ | `dist_Xm/curr_YmA/exp_name/{IQ_data/, metadata.json, report.json}` |
| Arquivos canônicos | ✅ | `IQ_data/X.npy` (transmitido) e `IQ_data/Y.npy` (recebido, sync+fase+âncora) |
| Arquivos intermediários | ✅ ignorados | `x_sent.npy`, `y_recv.npy`, `y_recv_sync.npy`, `y_recv_norm.npy` |
| Regimes | ✅ 27 | 3 distâncias × 9 correntes; 1.8m excluído (ruído de quantização, SNR≈0dB) |
| Amostras por regime | ✅ ~899k | Após skip de 10% do startup GNU Radio |
| Clamp decoder calibrado | ✅ | `[-5.82, -0.69]` derivado do q1%-1nat / q99%+1nat do log(var) empírico |
| Loader atualizado | ✅ | `ALT_SENT=[X.npy, sent_data_tuple.npy]`, `ALT_RECV=[Y.npy, ...]` |
| `report.json` integrado | ✅ | Métricas por regime no `df_info` (evm_pct, snr_dB, log_var_I/Q, JB p-values) |

### 0.2 Bugs de pipeline corrigidos

| Bug | Fix aplicado |
|---|---|
| **Clamp decoder [-6.0, 1.0]** → σ²_max=2.72 (5× acima do real) | `[-5.82, -0.69]` em `losses.py` e `analise_cvae_reviewed.py` |
| **`shuffle_train_batches=False`** → viés de gradiente inter-regime | `True` em `defaults.py` |
| **Eval não estratificada** → avalia só regime 0.8m/100mA | Amostragem uniforme por (Dn, Cn) no gridsearch |
| **Split após reduction** → leakage temporal | Split por experimento (head_tail) → reduce só no train |
| **Loader com nomes antigos** → `FileNotFoundError` no novo dataset | `ALT_SENT` / `ALT_RECV` com fallback legado |

### 0.3 Conhecimento empírico do canal (guia o design do modelo)

**Droop de eficiência do LED:** P_rx máximo em 200–300mA; cai monotonicamente acima disso.
O cVAE precisa aprender a relação **não-monotônica** entre corrente e SNR.

**Gradiente de kurtosis por regime:**
- 0.8m / 100–400mA: κ ≈ −0.04 → ruído quase Gaussiano (térmico + shot equilibrados)
- 0.8m / 500–900mA: κ = −0.27 a −0.67 → distorção não linear do LED dominante
- 1.0m: κ = −0.62 a −0.97 → transição; ruído de quantização emergente
- 1.5m: κ = −1.08 a −1.10 → plateau próximo a −1.2 (ADC)

**Implicação para o modelo:** um modelo Gaussiano homoscedástico falha por design em 0.8m/500-900mA.
O cVAE heteroscedástico é a escolha mínima necessária; a fidelidade estatística irá revelar se é suficiente.

---

## 1) Princípios do plano

1. **Provar em 1 regime** (single-regime proof) antes de escalar.
2. **Split sempre por experimento** (head=train / tail=val, 80/20). Proibido shuffle global.
3. **Ordem obrigatória:** split → reduce(train only) → treino. Val nunca toca reduction.
4. **Baseline forte obrigatório** como piso de comparação antes de qualquer resultado do cVAE.
5. **Cada etapa tem critério de aceite explícito.** Sem "treinar mais e ver" como resposta a falhas estruturais.
6. **Leakage zero:** decoder recebe apenas `(x, d, I, z)`; encoder pode ver `y`; decoder **nunca** vê `y`.
7. **Métricas em cascata** (do mais barato ao mais caro):
   - Nível 1 — Downstream: EVM(%), SNR(dB)
   - Nível 2 — Distribuição rápida: Δmean, Δcov, Δskew, Δkurt, PSD L2, JB(resíduo)
   - Nível 3 — Two-sample (quando `--stat_tests`): MMD², Energy distance, FDR q-values
8. **report.json por experimento é fonte de verdade** para JB p-values e log_var — usar para comparar com as predições do modelo.

---

## 2) Mapa de arquivos por regime

```
dataset_fullsquare_organized/
  dist_Xm/curr_YmA/exp_name/
    IQ_data/
      X.npy              ← cVAE entrada: sinal transmitido (N×2, float32)
      Y.npy              ← cVAE alvo: sinal recebido, sync+fase+normalização âncora (N×2, float32)
      [intermediários ignorados pelo loader]
    metadata.json        ← dist_m, curr_mA, regime_id, conversion.factor_ref, lag, fase
    report.json          ← evm_pct, snr_dB, log_var_I/Q, skew_I/Q, kurt_excess_I/Q,
                           jb_p_real_residual, jb_p_imag_residual, factor_ref
    plots/               ← amp_hist.png, overlay.png, psd_welch.png (diagnóstico visual)

  _report/
    REPORT.md            ← clamp recomendado, estatísticas globais do dataset
    summary_by_regime.csv ← 27 linhas, todas as métricas por regime
    heatmaps/            ← EVM, SNR, kurtosis, log_var por (dist, curr)
```

**Parâmetros herdados do dataset para o cVAE:**

| Parâmetro | Origem | Valor | Onde usar |
|---|---|---|---|
| Clamp lower | `_report/REPORT.md`, q1%−1nat | −5.82 | `losses.py`, `analise_cvae_reviewed.py` |
| Clamp upper | `_report/REPORT.md`, q99%+1nat | −0.69 | idem |
| Regime central | `summary_by_regime.csv`, mediana log_var | 1.0m/300mA | protocolo single-regime |
| JB p-value canal | `report.json`, `jb_p_real_residual` | por regime | comparação com JB predito |

---

## 3) Baseline — definição obrigatória

O baseline é o **piso de comparação**. O cVAE só tem valor científico se superar o baseline em fidelidade de distribuição, não apenas em EVM pontual.

### Baseline determinístico heteroscedástico

**Modelo:** regressão `x → ŷ = f(x, d, I)` com estimativa de variância por regime.

**Implementação:** `src/baselines/deterministic.py`

**O que o baseline captura:** média do canal condicionada em (x, d, I).
**O que o baseline não captura:** variância residual, assimetria, kurtosis não-Gaussiana, multimodalidade.

**Métricas do baseline (devem ser calculadas identicamente ao cVAE):**

| Métrica | Baseline esperado | cVAE deve superar |
|---|---|---|
| EVM (%) | Depende do regime | Em fidelidade de distribuição |
| Δmean L2 | Próximo a zero (baseline é a média) | Comparável |
| Δcov Frobenius | Alta (variância subestimada) | **Redução obrigatória** |
| Δkurt L2 | Alta (Gaussiano não captura kurtosis) | **Redução obrigatória** |
| JB p-value resíduo predito | Baixo (resíduo não-Gaussiano) | p-value maior = melhor |
| MMD² q-value | Baixo (distribuição incorreta) | q-value maior = melhor |

---

## 4) Métricas e interpretação

### 4.1 Nível 1 — Downstream

```
EVM(%) = sqrt(E[|ŷ - y|²] / E[|y|²]) × 100
SNR(dB) = 10 log₁₀(P_y / P_{y-ŷ})
```

Critério: |ΔEVM| < 5pp e |ΔSNR| < 1dB entre real e predito, para aprovação operacional.

### 4.2 Nível 2 — Distribuição rápida

Calculado sobre o resíduo `r = Y - X` (real) vs `r̂ = Ŷ - X` (predito):

- **Δmean L2:** norma da diferença entre médias (deve ser < 0.01 para canais centrados)
- **Δcov Frobenius:** diferença de covariâncias (captura variância e correlação I/Q)
- **Δskew L2, Δkurt L2:** assimetria e excesso de curtose (diagnóstico de caudas e multimodalidade)
- **PSD L2:** distância na densidade espectral de potência — captura memória temporal do canal
- **JB test no resíduo predito:** testa se o resíduo predito é Gaussiano (p-value baixo = não-Gaussiano, desejável para regimes de alta corrente)

**Interpretação do JB no contexto VLC:**
- 0.8m / 100–400mA: resíduo quase Gaussiano → JB p-value alto esperado no real e no predito
- 0.8m / 500–900mA: resíduo não-Gaussiano (LED não linear) → JB p-value baixo no real; modelo deve replicar
- 1.5m: ADC quantization dominante → JB p-value ≈ 0 no real; modelo deve capturar

### 4.3 Nível 3 — Two-sample (stat_tests)

Testa se as amostras do modelo e do canal real são indistinguíveis como distribuições:

- **MMD² (Maximum Mean Discrepancy):** com kernel RBF. p-value via permutação.
- **Energy distance:** sensível a localização e dispersão.
- **FDR (Benjamini-Hochberg):** correção para múltiplos testes nos 27 regimes.
  - q-value > 0.05: hipótese nula não rejeitada → distribuições indistinguíveis ✅
  - q-value < 0.05: rejeição → modelo falha na distribuição desse regime ❌

**Hierarquia de evidência para a tese:**
1. Se q-value > 0.05 em todos os 27 regimes: evidência forte do twin
2. Se q-value > 0.05 em regimes de baixa corrente mas < 0.05 nos de alta: modelo captura canal linear mas falha na não-linearidade → limite quantificado
3. Se q-value < 0.05 em todos: falha estrutural

---

## 5) Etapas de execução

---

### ETAPA A — Infraestrutura e smoke tests

> **Critério de saída:** pipeline roda do zero em 1 regime, 2 epochs, sem erros.
> Todos os artefatos obrigatórios gerados.

#### A1 — Verificar loader com novo dataset
**Objetivo:** confirmar que `load_experiments_as_list` encontra 27 experimentos,
carrega `X.npy` e `Y.npy`, e enriquece `df_info` com métricas do `report.json`.

**Smoke:**
```bash
python -c "
from src.data.loading import load_experiments_as_list
from pathlib import Path
exps, df = load_experiments_as_list(Path('data/dataset_fullsquare_organized'), verbose=True)
print(df[['dist_m','curr_mA','n_samples','evm_pct','snr_dB','log_var_I']].to_string())
"
```

**Critérios de aceite:**
- 27 linhas no df, status=ok em todas
- colunas `evm_pct`, `snr_dB`, `log_var_I`, `log_var_Q`, `factor_ref` presentes e não-NaN
- `sent_path` aponta para `X.npy`, `recv_path` para `Y.npy`

**Commit:** `fix(data): update loader for new dataset format (X.npy/Y.npy + report.json)`

#### A2 — Smoke de treino (1 regime, 2 epochs)
**Objetivo:** pipeline treino→eval sem erros de IO, shape ou NaN.

**Smoke:**
```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --max_epochs 2 --max_grids 1 --max_experiments 1 --max_samples_per_exp 2000
```

**Critérios de aceite:**
- `state_run.json` gerado com `shuffle_train_batches=true`
- clamp em `state_run.json`: `decoder_logvar_clamp: [-5.82, -0.69]`
- `df_info` no state tem 1 regime com `evm_pct` não-NaN
- nenhum NaN em `train_loss` ou `val_loss`

**Commit:** `test(smoke): training pipeline green with new dataset + fixes`

#### A3 — Baseline smoke
**Objetivo:** baseline roda, gera métricas comparáveis ao cVAE.

**Smoke:** idem A2 com `--no_cvae --baseline_only`

**Critérios de aceite:**
- `summary_by_regime.*` com colunas baseline e cVAE
- EVM baseline plausível para o regime (15–20% para 0.8m/100mA)

**Commit:** `feat(baseline): smoke test green`

#### A4 — Stat fidelity smoke
**Objetivo:** MMD²/Energy/FDR rodam sem erros em `stat_mode=quick`.

**Smoke:**
```bash
python -m src.protocol.run ... \
  --max_epochs 2 --max_grids 1 --max_experiments 1 \
  --stat_tests --stat_mode quick
```

**Critérios de aceite:**
- `tables/stat_fidelity_by_regime.*` gerado
- `plots/stat_tests/*.png` gerado (4 plots)
- `manifest.json` contém `stat_fidelity_config` e `stat_acceptance`

**Commit:** `refactor(stat): stat fidelity suite green in smoke`

---

### ETAPA B — Single-Regime Proof (ciência)

> **Regime pivô: 1.0m / 300mA**
> Justificativa: log_var = −3.20 (mediana do dataset, 0.29 nats da mediana global),
> SNR = 9.1dB (moderado, representativo), próximo ao pico de eficiência do LED.

> **Critério de saída desta etapa:** cVAE supera baseline em Δcov, Δkurt e JB replicado,
> E q-value MMD² > 0.05 no regime pivô.

#### B1 — Treino real no regime pivô
**Config:**
```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA.json \
  --max_grids 1 \
  --stat_tests --stat_mode full --stat_max_n 5000 --stat_n_perm 1000
```

**Artefatos obrigatórios:**
- `logs/metricas_globais_reanalysis.json` com todas as métricas Nível 1 e 2
- `tables/stat_fidelity_by_regime.*` com MMD² p-value, Energy p-value, q-values FDR
- `plots/stat_tests/`: heatmap_mmd2, scatter_mmd2_vs_evm, heatmap_psd_dist
- `tables/latent_diagnostics.xlsx`: active_dims, KL(q||p), KL(p||N)
- `plots/`: overlay constelação, histogramas I e Q, PSD comparativo

**Gates de aceite (obrigatórios para avançar):**

| Gate | Critério | Falha → ação |
|---|---|---|
| G1 — Baseline converge | EVM < 30% em 1.0m/300mA | Verificar loader |
| G2 — cVAE não explode | \|ΔEVM\| < 15pp vs real | Debug B2 |
| G3 — cVAE supera baseline em cov | Δcov_cVAE < Δcov_baseline | Revisar decoder |
| G4 — cVAE supera baseline em kurtosis | Δkurt_cVAE < Δkurt_baseline | Revisar β/latent_dim |
| G5 — JB replicado | JB p-value predito ≈ JB p-value real (±20%) | Revisar clamp |
| G6 — MMD² q-value | q-value > 0.05 | Se falhar → B2-debug |

**Gate G5 — detalhe crítico:**
O `report.json` de cada experimento contém `jb_p_real_residual` e `jb_p_imag_residual`
(Jarque-Bera no resíduo real do canal). O modelo deve reproduzir esse valor.
Para 1.0m/300mA, o resíduo real é quase-Gaussiano (SNR moderado, sem distorção forte).
Se o modelo prediz resíduo muito mais ou menos Gaussiano, o clamp ou o β estão errados.

**Commit (só se todos os gates passarem):**
`feat(results): single-regime proof B1 passes all gates — 1.0m/300mA`

#### B2 — Debug (se B2 falhar em algum gate)
**Diagnóstico estruturado:**

```
G2 falha (ΔEVM explode):
  → Verificar que decoder não recebe Y
  → Verificar normalização D,C no state_run vs eval

G3 falha (Δcov pior que baseline):
  → latent_dim muito pequeno → aumentar
  → β muito alto (colapso posterior) → reduzir β ou aumentar free_bits

G4 falha (Δkurt pior que baseline):
  → clamp muito restritivo → verificar [-5.82, -0.69] aplicado
  → modelo não captura caudas: considerar decoder não-Gaussiano (futuro)

G5 falha (JB não replicado):
  → σ² predito muito diferente do real → revisar clamp
  → var_pred / var_real >> 1 → clamp upper muito alto

G6 falha (MMD² rejeitado):
  → first check G3 e G4 (problemas de distribuição)
  → n_perm muito baixo → aumentar para 2000
  → regime muito difícil → documentar como limite do modelo
```

**Commit por patch mínimo:**
`fix(cvae): [descrição do bug específico encontrado]`

#### B3 — Diagnóstico não-linearidade
**Objetivo:** verificar se o resíduo cresce com |x| (assinatura de não-linearidade do LED).

**Análise:** binning de |x| em 10 faixas; calcular RMS do resíduo por faixa para real e predito.

**Critério:** se baseline tem resíduo crescente com |x| e cVAE replica o padrão, isso confirma que o modelo capturou a não-linearidade.

**Commit:** `feat(analysis): residual vs amplitude — nonlinearity diagnostics`

---

### ETAPA C — Escala incremental

> **Pré-condição:** Etapa B completa com todos os gates G1–G6.

#### C0 — Regimes de baixa corrente (100–300mA, 3 distâncias)
**Justificativa:** esses regimes têm resíduo quase-Gaussiano (κ ≈ −0.04).
São os mais fáceis para o modelo; servem como validação de consistência.

**Esperado:** q-value > 0.05 em todos, JB replicado.
**Se falhar aqui:** problema estrutural, não de dificuldade do canal.

**Commit:** `feat(protocol): low-current proof (100-300mA × 3 distances)`

#### C1 — Regimes de alta corrente (700–900mA, 0.8m)
**Justificativa:** máxima não-linearidade (κ = −0.67 a −1.0).
Teste da capacidade do cVAE de capturar caudas não-Gaussianas.

**Esperado:** q-value pode ser < 0.05 — documentar como limite quantificado do modelo.
**Critério real:** q-value_cVAE > q-value_baseline (mesmo que ambos rejeitem).

**Commit:** `feat(protocol): high-current nonlinear regimes (700-900mA × 0.8m)`

#### C2 — Grade completa (27 regimes)
**Config:** sem `--max_experiments`, `stat_mode=full`, `n_perm=1000`.

**Artefatos obrigatórios adicionais:**
- `tables/summary_by_regime.xlsx` com todas as métricas por regime
- Heatmap 3×9 de q-values (MMD² e Energy) — visualiza onde o modelo falha
- Heatmap 3×9 de ΔEVM e ΔΔkurt (real vs predito)

**Critério global:**
- > 20/27 regimes com q-value > 0.05 = evidência forte
- 15–20/27 = evidência moderada (documentar quais regimes falham e por quê)
- < 15/27 = falha — retornar a B2-debug com evidências dos 27 regimes

**Commit:** `feat(protocol): full-grid 27-regime training + stat fidelity`

---

### ETAPA D — Generalização (hold-out)

> Só iniciar se C2 passou.

#### D1 — Hold-out por corrente
Treinar em correntes pares (200, 400, 600, 800mA), validar em ímpares (100, 300, 500, 700, 900mA).

**Critério:** degradação de EVM < 10pp entre regimes vistos e não vistos.
Isso valida interpolação no condicionamento `C`.

#### D2 — Hold-out por distância
Treinar em 0.8m e 1.5m, validar em 1.0m.

**Critério:** EVM em 1.0m deve ser interpolado corretamente entre 0.8m e 1.5m.
Isso valida interpolação no condicionamento `D`.

---

### ETAPA E — Seleção do twin vencedor

**Comparação final:**

| Critério | Peso | Baseline | cVAE |
|---|---|---|---|
| EVM/SNR downstream | 20% | referência | ΔEVM vs baseline |
| Δcov Frobenius | 20% | referência | deve ser menor |
| Δkurt L2 | 20% | referência | deve ser menor |
| JB replicado | 15% | N/A | |ΔJBP| < 20% |
| MMD² q-value | 15% | N/A | > 0.05 em N/27 |
| Generalização hold-out | 10% | referência | degradação < 10pp |

**Saída:** `docs/TWIN_SELECTION.md` com tabela de evidências por critério.

---

## 6) Parâmetros fixos para todos os runs

```python
# Dataset
DATASET_ROOT = "data/dataset_fullsquare_organized"
N_REGIMES    = 27  # 3 dist × 9 curr; 1.8m excluído

# Pipeline
SPLIT_MODE             = "per_experiment"
SPLIT_ORDER            = "head_tail"
VALIDATION_SPLIT       = 0.20
WITHIN_EXP_SHUFFLE     = False
SHUFFLE_TRAIN_BATCHES  = True   # Fix 2
REDUCTION_TARGET       = 200_000  # por experimento, aplicado SÓ no train

# Decoder
DECODER_LOGVAR_CLAMP_LO = -5.82  # Fix 1 — calibrado empiricamente
DECODER_LOGVAR_CLAMP_HI = -0.69  # Fix 1 — calibrado empiricamente

# Eval
N_EVAL_SAMPLES_STRATIFIED = 40_000  # Fix 3 — estratificado por (D, C)
SEED = 42
```

---

## 7) Checklist por commit (obrigatório)

- [ ] `git status` limpo antes e depois
- [ ] `git diff --stat` pequeno e justificado
- [ ] Smoke test específico da etapa executado
- [ ] Evidências registradas (paths em `outputs/exp_*`)
- [ ] Nenhum arquivo de dados ou flags no git

---

## 8) Definition of Done global

O plano está concluído quando:

1. Twin vencedor identificado com evidência reprodutível (tabela em `TWIN_SELECTION.md`)
2. Generalização hold-out quantificada (degradação documentada)
3. Limites do modelo documentados (quais regimes o modelo falha e por quê)
4. Pipeline completamente auditável (manifests, inventories, splits, all metrics)
5. Figuras e tabelas prontas para tese (EVM por regime, heatmap de q-values, overlay de constelações)
