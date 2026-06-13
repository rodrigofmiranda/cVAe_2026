# Cross-Correlation — Identificação da Resposta do Canal VLC

Estudo de validação do gêmeo digital pela **cross-correlation entrada–saída**,
como terceiro eixo (ortogonal aos gates G1–G6 e à calibração coverage).

## Embasamento científico

A fundamentação completa está na TESE:
`TESE/06_validacao_do_gemeo/cross_correlation_fundamento_2026-06-13.md`.

Resumo: sob a excitação **full-square** (2^16 padrões, banda larga ⇒ entrada
quase branca, `R_xx ≈ δ`), a cross-correlation entrada–saída estima a resposta
ao impulso do canal: `R_xy(τ) ≈ h(τ)` (identificação de sistemas por
cross-correlation / Wiener–Hopf). Para a estrutura **Hammerstein** do canal VLC
(não-linearidade estática do LED → filtro linear; `dang_2022` na KB), `R_xy`
identifica o **bloco linear** (memória/ISI). É **necessária mas não suficiente**:
não captura a não-linearidade estática (exigiria kernels de Volterra).

## Aplicação (estudo cross-distance)

Comparar `R_xy^real(τ)` (medição) vs `R_xy^twin(τ)` (gêmeo, inferência
determinística = média condicional) nas distâncias **não vistas**
(0.9/1.16/1.25 m), com o real como gabarito. Mede se o canal aprendido nas
âncoras (0.75/1.0/1.35/1.5 m) reproduz a resposta linear do canal onde nunca
foi treinado.

## Conteúdo

- `compute_xcorr.py` — núcleo do estimador, **testado** (self-test: recupera `h`
  de dados I/O com entrada branca, corr=1.0; checagem de whiteness; discrepância).
  - `normalized_xcorr(x, y, max_lag)` → `R_xy(τ)`
  - `xcorr_discrepancy(r_real, r_twin)` → L2, erro de pico (ganho), erro de atraso
  - `input_whiteness(x)` → pré-condição de validade (quão branca é a entrada)
- `run_xcorr_crossdist.py` — driver (PENDENTE): carrega o modelo cross-dist
  salvo, roda inferência determinística no `x` real por regime, computa
  `R_xy` real vs twin nas 7 distâncias, salva tabela em `results/` e figuras em
  `figures/`. A ser escrito/validado quando o run `v3fs_crossdist_20260613`
  fechar (~15h) — depende do formato dos artefatos de eval.
- `results/`, `figures/` — saídas curadas (versionadas; dados brutos ficam em `outputs/`).

## Pré-condição a verificar empiricamente

Antes de interpretar `R_xy` como `h`, confirmar `input_whiteness(x)` baixo nos
dados full-square reais (a teoria assume entrada branca). Documentar o desvio.
