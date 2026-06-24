# Macro Diagnostics — Pipeline de Diagnóstico Pós-Modelagem (V3)

Pipeline **ordenado** que consolida todas as comparações pós-treino do gêmeo
digital cVAE (FS e FC) numa sequência de camadas com um veredito único por
regime. O objetivo final não é só comparar — é **decidir como separar os dados
para o próximo treino** (estratificação / oversample / especialista /
hold-out de generalização), com trilha de evidência.

> Sucede e *reorganiza* (não substitui) os diagnósticos dispersos em
> `comparison_v3/{awgn,cross_correlation,fs_vs_fc_crossdist,modulations,modulations_unseen}`.
> Consome os artefatos que eles já geram; só re-roda inferência pesada no modo `--full`.

## Conclusões que o pipeline parte (e re-verifica)

1. **A falha é distribucional/estocástica, não linear.** A cross-correlation
   `R_xy(τ)` mostra que a resposta linear determinística é interpolada com
   precisão (`peak_lag_err = 0`, `xcorr_l2 < 1%`) inclusive em distâncias
   não-vistas. Logo a separação de dados deve mirar a **lei do ruído / cauda /
   variância**, não o ganho físico.
2. **O cluster de falha é sistemático**: near-field `0.75 m` (todas as
   correntes), extrapolação de distância não-vista (`0.9/1.16/1.25 m`) e alta
   corrente (droop do LED). É o análogo V3 do gargalo histórico "0.8 m / G5"
   (ver `ai_workflow/EXPERIMENT_HISTORY.md`).
3. **O histórico já levanta a hipótese global × especialista.** As rotas de
   decoder (MDN-k, kurt, clamp, embed) não fecham o teto. Isso motiva
   diretamente o estudo de **separação dos dados** que este pipeline instrui.

## Camadas (ordem de leitura)

| # | Camada | Pergunta | Fonte (artefato consumido) |
|---|--------|----------|----------------------------|
| 0 | **Census de dados** | Como é a *distribuição real* por regime? (kurt/skew/var de δ, het-slope, SNR, N, potência) | `regime_census.py` (model-free, docker/CPU) |
| 1 | **Resposta linear** | O ganho determinístico interpola? | `cross_correlation/xcorr_table_*.csv` |
| 2 | **Fidelidade distribucional** | O gêmeo casa a lei do ruído? | `awgn/{fc,fs}/metrics_table.csv`, `awgn/shot_noise_coefficients_*.csv` |
| 3 | **Explicação física** | Onde a não-linearidade do LED quebra a Gaussianidade? | `lib/plot_led_synthesis_v3.py` (port V2, `--full`) |
| 4 | **Veredito por gate** | Quais gates falham em quais regimes? | `fs_vs_fc_crossdist/gates_summary_*.csv` |
| 5 | **Métrica de aplicação** | O BER bate? | `modulations*/.../ber_table_*.csv` |
| 6 | **Contraste baseline** | O quão melhor que o AWGN? | `awgn/{fc,fs}/metrics_table.csv` (method=awgn) |
| 7 | **Síntese / separação** | Quais dados separar para melhorar o treino? | `build_macro_report.py` (este pipeline) |

## Saídas (por execução, em `runs/<stamp>/`)

- `summary_master.csv` — **uma linha por (modelo, regime)** com todos os sinais
  consolidados + `difficulty_score` + `cluster` + `separation_action`.
- `REPORT.md` — narrativa em camadas + **tabela de recomendação de separação de
  dados** + nota de governança (regimes screening / final-validation /
  generalização).
- `manifest.json` — proveniência (fontes consumidas, modo, timestamp).
- (modo `--full`) figuras macro re-geradas: LED synthesis, residual KDE grid.

## Reprojeto do treino

`REDESIGN_PLAN.md` — plano concreto de separação de dados + currículo derivado do
diagnóstico: dois problemas distintos (fronteira near-field de alto SNR em 0.75m,
e interpolação de distância nos gaps), com splits, governança, currículo e os
experimentos ranqueados. Inclui o "NÃO fazer" (cauda de MDN — alvo errado).

## Como rodar

```bash
# Modo rápido (default): só agrega os CSVs já existentes. CPU, sem docker.
bash run_macro_diagnostics.sh

# Modo completo: re-roda inferência/figuras a partir dos modelos FS/FC (docker, GPU)
# e a censura model-free, depois agrega.
bash run_macro_diagnostics.sh --full
```

O agregador `build_macro_report.py` é **pure-stdlib** (csv/json/math) — roda em
qualquer python. A censura e os plotters de cauda/LED usam numpy → rodam em
docker (regra do projeto: nunca host python).

## Design

- `build_macro_report.py` — cérebro pure-stdlib. Lê os CSVs, normaliza chaves de
  junção por `(dist_m: float, curr_mA: int, model)` (NÃO por string de regime,
  porque `dist_1m` do gate ≠ `dist_1p0m` do metrics), calcula difficulty/cluster,
  escreve `summary_master.csv` + `REPORT.md`.
- `regime_census.py` — model-free, docker/CPU. Carrega `X.npy/Y.npy` por regime,
  computa estatísticas reais de δ=Y−X (a base da decisão de separação).
- `lib/` — ports parametrizados dos plotters macro do V2 congelado
  (`comparison/plot_led_synthesis.py`, `plot_residual_kde_grid.py`).
- `run_macro_diagnostics.sh` — orquestrador `--fast`/`--full` + ntfy.
