# Plano de Reprojeto do Dataset/Treino cVAE V3

Derivado do diagnóstico macro (`macro_diagnostics/runs/20260617_032037/`). Objetivo:
separar/reamostrar os dados e ajustar o treino para quebrar o teto, **mirando o
alvo certo**. Não retreina aqui — entrega o desenho.

## Premissa (o que o diagnóstico estabeleceu)

1. **Dentro de cada regime o canal é ~linear + ruído Gaussiano.** Orçamento de
   Var(Y−X): **85% ganho linear, ~0% não-linear, ~15% ruído Gaussiano** (held-out,
   até em 0.75m/900mA). `kurt` do resíduo final ≈ 0. → Não há cauda pesada nem
   warp não-linear intra-regime para modelar.
2. **A não-linearidade do LED é efeito ENTRE pontos de operação** — o ganho
   `a(d,c)` muda com bias/distância. O trabalho do modelo é estimar `(ganho,
   escala-de-ruído)` por regime e **interpolar** em `(d,c)`.
3. **As falhas têm duas naturezas distintas** (tabela real do run):

   | dist (m) | papel | SNR méd | FC | FS | natureza da falha |
   |---|---|---|---|---|---|
   | 0.75 | treinada | **17.9** | 0/9 | 0/9 | **precisão relativa** (SNR alto) |
   | 0.9  | não-vista | 9.7 | 0/9 | 0/9 | interpolação (gap) |
   | 1.0  | treinada | 7.0 | 8/9 | 8/9 | ok |
   | 1.16 | não-vista | 4.9 | 2/9 | 0/9 | interpolação (gap) |
   | 1.25 | não-vista | 4.1 | 3/9 | 0/9 | interpolação (gap) |
   | 1.35 | treinada | 3.4 | 9/9 | 9/9 | ok |
   | 1.5  | treinada | 2.7 | 7/9 | 6/9 | ok |

   **0.75m é treinada, tem o maior SNR, e falha 100%.** Como os gates V3 são
   **relativos**, onde o ruído é mínimo um erro minúsculo de ganho/escala do
   modelo vira erro relativo grande. Não é "física difícil" — é precisão.

## Dois problemas, dois fixes

### Problema A — fronteira near-field de alto SNR (0.75m treinada falha)
Não é falta de dados (já está no treino); é o modelo **global sacrificar
precisão** na ponta de alto SNR. Fixes (do mais barato ao mais estrutural):
- **A1 — loss com peso relativo / normalizado por regime**: penalizar erro
  relativo (não absoluto), para que regimes de alto SNR não sejam afogados pelo
  bulk de baixo SNR. *Barato, primeiro a testar.*
- **A2 — oversample por necessidade de precisão**: peso de amostragem ∝ 1/σ(d,c)
  (alto SNR pesa mais), NÃO ∝ cauda.
- **A3 — especialista near-field vs global**: treinar um modelo dedicado a
  d≤0.9m. Resolve a tensão global×especialista que o histórico levantou — agora
  com evidência de que é precisão relativa de fronteira, não cauda.

### Problema B — interpolação de distância (não-vistas nos gaps)
Grid de treino atual `{0.75, 1.0, 1.35, 1.5}` tem **gap de 0.35m entre 1.0 e
1.35**, exatamente onde 1.16/1.25 caem e falham. Fixes:
- **B1 — densificar o grid de treino**: treinar em `{0.75, 0.9, 1.0, 1.16, 1.35,
  1.5}` e manter só `{1.25}` como prova de generalização. Fecha o gap grande.
- **B2 — re-espaçar mantendo 4 treinadas**: `{0.75, 1.0, 1.25, 1.5}` (gaps 0.25
  uniformes) em vez de `{0.75, 1.0, 1.35, 1.5}` (gap 0.35 + cluster 1.35/1.5);
  held-out `{0.9, 1.16, 1.35}`. Melhora interpolação sem novas capturas.
- **B3 — prior de suavidade/monotonicidade** em `a(d,c)` e `σ(d,c)`: o ganho e a
  escala variam monotonicamente com a distância (SNR cai 17.9→2.7 dB monotônico),
  então impor suavidade no condicionamento ajuda a interpolar.

## Governança dos dados (papéis — lacuna #1 da síntese de validação)

- **Screening** (dev rápido): núcleo bem-modelado `{1.0, 1.35 m}` × correntes.
- **Final-validation** (aceite): cobertura treinada completa `{0.75 … 1.5 m}`.
- **Generalização** (advisory, nunca no treino): as distâncias mantidas como
  held-out em B1/B2.

## Currículo de amostragem (treino)

Como cada regime é linear+Gaussiano, o currículo foca **média condicional +
escala**, não distribuição:
1. **Fase 1 — backbone**: núcleo de baixo erro `{1.0, 1.35 m}` → fixa `a(d,c)` e
   `σ(d,c)` no miolo.
2. **Fase 2 — fronteiras com peso relativo**: adiciona `0.75 m` (alto SNR, peso
   A1/A2) e `1.5 m` (ponta longa).
3. **Fase 3 — densificação**: adiciona as intermediárias do grid escolhido (B1/B2)
   para travar a interpolação.

## Arquitetura / loss (toque leve — foco é dado)

- **Reduzir capacidade de cauda do MDN** → decoder Gaussiano (ou 1–2 componentes)
  basta. **Corroboração**: no histórico o S1–S2 **Gaussiano** já atingiu 10/12;
  consistente com "ruído é Gaussiano, MDN de cauda é desnecessário".
- **Loss de erro relativo / normalizado por regime** (Problema A).
- Capacidade liberada do MDN vai para **média condicional + condicionamento em
  amplitude/distância**.

## NÃO fazer (negativos agora explicados mecanicamente)

- **MDN-k maior (S29), λ_kurt (S28), alargar clamp (S33)** — miram cauda de ruído
  que **não existe** (frac não-linear ≈ 0, resíduo Gaussiano). O diagnóstico
  explica por que esses caminhos bateram no teto.

## Próximos experimentos (ranqueados, barato→caro)

| # | Experimento | Custo | Testa |
|---|---|---|---|
| E1 | Re-treino: decoder Gaussiano + loss relativo, grid atual | baixo | Problema A (0.75m melhora?) |
| E2 | Grid densificado B1 (ou re-espaçado B2) | médio | Problema B (não-vistas) |
| E3 | Especialista near-field d≤0.9m | médio | A3 (global×especialista) |
| E4 | Prior de suavidade em a/σ(d,c) | médio | B3 (interpolação) |

**Métrica de sucesso**: subir o PASS em 0.75m (hoje 0/9) e nas distâncias
intermediárias, sem regredir o núcleo 1.0–1.35m. Re-rodar este pipeline
(`run_macro_diagnostics.sh --fast`) após cada treino para comparar o
`summary_master.csv`.

## Ressalva

O veredito "sem não-linearidade intra-regime" vem de um fit de ganho dependente de
amplitude `g(P)·x` (held-out, robusto). Se a não-linearidade real tiver outra
forma, parte do "ganho linear" poderia ser não-linear disfarçado — mas isso não
muda o plano (o alvo continua sendo ganho/escala condicional + interpolação, não
cauda).
