# TRAINING_PLAN.md — Plano Científico Vivo do Digital Twin VLC

> Versão ativa em 2026-03-12.
> Este documento é a fonte de verdade para validação científica e critérios de aceite.

## 1. Objetivo

Demonstrar, por regime `(d, I)`, que o digital twin baseado em cVAE reproduz a
distribuição condicional do canal VLC:

`p(y | x, d, I)`

O objetivo não é apenas boa predição média, e sim fidelidade distribucional do
resíduo do canal:

`Δ = Y - X`

## 2. Invariantes obrigatórios do pipeline

Estes pontos não devem ser relaxados nos experimentos:

1. Split por experimento, temporal: `head=train`, `tail=val`, `80/20`
2. Ordem obrigatória: `split -> cap/reduce(train only) -> treino`
3. `shuffle_train_batches = True`
4. Clamp do decoder calibrado: `[-5.82, -0.69]`
5. Métricas distribucionais sobre `Δ = Y - X`, não sobre `Y` bruto
6. Inferência para métricas de distribuição via `MC-concat`, não MAP
7. `summary_by_regime.csv` é a tabela canônica de validação final

## 3. Métricas ativas

### Nível 1 — Fidelidade física

- `evm_real_%`, `evm_pred_%`, `delta_evm_%`
- `snr_real_db`, `snr_pred_db`, `delta_snr_db`

### Nível 2 — Fidelidade distribucional do resíduo

- `delta_mean_l2`
- `delta_cov_fro`
- `var_real_delta`, `var_pred_delta`, `var_ratio_pred_real`
- `delta_skew_l2`
- `delta_kurt_l2`
- `delta_psd_l2`
- `jb_p_min`, `jb_log10p_min`
- `jb_real_p_min`, `jb_real_log10p_min`

### Nível 3 — Testes formais de duas amostras

- `stat_mmd2`, `stat_mmd_pval`, `stat_mmd_qval`, `stat_mmd_bandwidth`
- `stat_mmd2_normalized`
- `stat_energy`, `stat_energy_pval`, `stat_energy_qval`
- `stat_psd_dist`, `stat_psd_ci_low`, `stat_psd_ci_high`

### Comparação baseline vs cVAE

- `baseline_*`
- `cvae_*`
- `better_than_baseline_cov`
- `better_than_baseline_kurt`
- `better_than_baseline_psd`

## 4. Gates G1–G6

Os gates são critérios de aceite por regime. Eles já aparecem em
`summary_by_regime.csv` como `gate_g1` … `gate_g6`.

| Gate | Regra atual |
|---|---|
| `G1` | `baseline_evm_pred_% < 30` |
| `G2` | `abs(delta_evm_%) < 15` |
| `G3` | `cvae_delta_cov_fro < baseline_delta_cov_fro` |
| `G4` | `cvae_delta_kurt_l2 < baseline_delta_kurt_l2` |
| `G5` | `abs(cvae_jb_log10p_min - jb_real_log10p_min) < 1.0` |
| `G6` | `stat_mmd_qval > 0.05` |

### Observação crítica sobre G5

A formulação antiga baseada em diferença percentual de `JB p-value` não é mais
canônica, porque sofre underflow numérico em amostras grandes. A forma válida é
em `log10(p)`.

### Status derivado por regime

- `validation_status = pass` se todos os gates disponíveis forem `True`
- `validation_status = fail` se algum gate disponível for `False`
- `validation_status = partial` se houver gates indisponíveis (`NaN` / `None`)

## 5. Regime pivô e sequência mínima de validação

### Regime pivô

Usar `1.0m / 300mA` como prova inicial de regime único.

### Sequência mínima

1. Testes unitários

```bash
python -m pytest tests -q
```

2. Smoke de protocolo

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA.json \
  --max_epochs 1 --max_grids 1 --max_experiments 1 --max_samples_per_exp 2000
```

3. Run científico de regime único com testes estatísticos

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/one_regime_1p0m_300mA.json \
  --max_grids 1 \
  --stat_tests --stat_mode full --stat_max_n 5000 --stat_n_perm 1000
```

## 6. Como ler o resultado

### Evidência forte

- `G1`–`G6` passam no regime pivô
- `validation_status = pass`
- `stat_mmd_qval > 0.05`
- cVAE supera baseline em covariância e kurtosis

### Resultado operacional, mas não científico

- `G1` e `G2` passam
- `G3`–`G6` falham

Isso significa que o modelo acerta melhor a parte de telecomunicações
(`EVM`/`SNR`) do que a distribuição completa do canal.

### Falha estrutural provável

Se, após todos os fixes de pipeline, `G3`–`G6` continuam falhando de forma
sistemática, interpretar como limitação do modelo atual
(ex.: prior Gaussiano / decoder insuficiente), não como bug de implementação.

## 7. Próximos passos científicos

1. Fechar a prova de regime único em `1.0m / 300mA`
2. Rodar o protocolo em múltiplos regimes e mapear onde `G3`–`G6` falham
3. Separar claramente:
   - regimes quase-Gaussianos, onde o modelo já pode ser suficiente
   - regimes não-lineares / platocúrticos, onde pode haver limite estrutural
4. Só depois discutir mudança de família de modelo

## 8. Documentos complementares

- [PROJECT_STATUS.md](/workspace/2026/PROJECT_STATUS.md)
- [docs/PROTOCOL.md](/workspace/2026/docs/PROTOCOL.md)
- [docs/MODELING_ASSUMPTIONS.md](/workspace/2026/docs/MODELING_ASSUMPTIONS.md)
