# Reprojeto: do cVAE caixa-preta ao espelho gray-box de canal OWC

> **Status**: proposta fundamentada (2026-06-27). Decorre do esgotamento empírico da
> linha "loss/prior sobre os dados vistos" (E1/A1/A4/B3) para o problema de
> **generalização de distância**. Fonte-mãe dos fatos: [CRITICAL_PROJECT_KNOWLEDGE.md](CRITICAL_PROJECT_KNOWLEDGE.md)
> §5–§6 e `scripts/analysis/macro_diagnostics/REDESIGN_PLAN.md`.

---

## 1. O problema, reformulado corretamente
Objetivo real: o **espelho estatístico-comportamental de um canal OWC** que
**generaliza** para pontos de operação NÃO vistos. As distâncias intermediárias
(0.9 / 1.16 / 1.25 m) são a **PROVA** dessa generalização — por decisão do usuário,
**não podem entrar no treino** (densificar o grid destrói a prova). Logo o twin
existe para **extrapolar física condicional** entre âncoras, não para ajustar bem
4 distâncias. Treino = `{0.75, 1.0, 1.35, 1.5}` m; held-out = `{0.9, 1.16, 1.25}` m.

## 2. O muro empírico (por que sabemos que não é ajuste de loss)
Quatro intervenções de treino, mesmo desfecho nas não-vistas:

| experimento | n_pass/63 | G3 nas não-vistas 1.16/1.25m |
|---|---|---|
| E1 — Gaussiano baseline | **30** | falham, ~2x o limiar |
| A1 — loss de erro relativo (mira G1/G3 pela média) | 29 | imóvel |
| A4 — loss de heterocedasticidade (casa Var(δ)~\|X\|) | 29 | imóvel (+seed7 divergiu) |
| B3 — prior de suavidade em μ(d) | **26** (pior) | imóvel (G3 2.2x/1.9x) |

O **G3** (média/dispersão do resíduo rel. à escala) nas distâncias não-vistas é
**imóvel a ~2x**, qualquer que seja a loss. O B3 ainda provou que **μ(d) já é suave**
(`smooth_loss`~5e-6) — o modelo interpola suave, só que para o **valor errado**.
**Conclusão**: o viés da média condicional numa distância nunca vista é
**irredutível por truque de treino** sobre as 4 âncoras.

## 3. Por que o cVAE caixa-preta não resolve — fundamentado
Do próprio diagnóstico (decomposição de variância, held-out, por regime):
- **Intra-regime o canal é trivial**: ~85% ganho **linear** + ~15% ruído
  **Gaussiano** + **0% não-linear** (até em 0.75m/900mA), kurt_resíduo≈0. Ou seja,
  dentro de cada (d,c): `Y ≈ a(d,c)·X + N(0, σ(d,c)²)`.
- **A não-linearidade vive ENTRE regimes**: `a(d,c)` e `σ(d,c)` variam com
  distância/corrente — *esse* é o objeto a modelar.
- **O cVAE aprende `a(d,c)`, `σ(d,c)` como MLP caixa-preta de (d,c)**. Com 4
  distâncias no treino, o MLP acerta as âncoras mas o **interpolante entre elas é
  subdeterminado**: nada na loss informa qual curva `a(d)` seguir nos gaps.

### Descasamento conceitual (o coração de "um cVAE resolveria?")
- A **força** de um cVAE é modelar distribuições **complexas/multimodais/
  cauda-pesada** via latente `z`. O canal OWC intra-regime é **unimodal Gaussiano**
  → essa força é **desperdiçada**.
- A **fraqueza** de um cVAE é o **condicionamento caixa-preta sem prior físico** →
  é **exatamente** onde ele falha (interpolação).
- Logo o cVAE resolve a parte fácil (ruído Gaussiano) e falha na real (interpolação
  física dos momentos). **Para este objetivo, é a ferramenta errada** —
  superdimensionada no que não importa, subequipada no que importa.

## 4. O prior que a física do OWC dá (e que o cVAE joga fora)
Ganho de link VLC/OWC LOS segue lei conhecida: `H(0) ∝ 1/d²` (geometria
Lambertiana) × **transfer não-linear estático do LED** na corrente (Hammerstein;
cards `dang_2022` / `bojarczuk_2023` referenciados no macro Camada 0):
```
a(d,c) ≈ K · h_LED(c) / d²        σ(d,c) ≈ modelo de ruído (∝ potência recebida)
```
Funções **suaves, separáveis, com POUCOS parâmetros**. Ajustados nas 4 âncoras,
`a(d)` em qualquer distância sai **exato**: `a(d) = a(d₀)·(d₀/d)²`. Essa é a
interpolação que o MLP não consegue — porque não sabe que existe um `1/d²`.
> Caveat: a forma paramétrica exata (`1/d²` puro? termo de FOV/ângulo? ruído
> shot+térmico?) deve ser **ajustada e validada contra os dados**, não assumida.

### 4.1 Lastro na literatura do próprio `knowledge/` (não é improviso)
Os cards já no `knowledge/notes/` dão a estrutura paramétrica e — crucialmente — a
**propriedade de extrapolação** que o MLP caixa-preta não tem:
- **`dang_2022` (Hammerstein⊳Wiener)**: a NL e2e do link VLC é **Hammerstein** =
  *NL estática (saturação do LED) → filtro linear*. Confirma `μ = h_LED(c)·(filtro)·X`
  como forma certa; a saturação é **estática na corrente**, separável da distância.
- **`bojarczuk_2023` (modelo comportamental, Volterra 2ª ordem das eqs. de taxa)**:
  3 blocos (filtro→quadrador→filtro) com **só 4 parâmetros físicos** (BW₃dB, corte,
  conversão óptica, fator de NL) e — textual no card — *"parametrização física permite
  **extrapolação entre pontos de operação**"*. É **exatamente** a interpolação que
  falta. Poucos parâmetros suaves ⇒ generaliza por construção.
- **`liu_2024` / `ramirez_2018`**: baselines estatístico/Lambertiano confirmam `1/d²`
  LOS + ruído por regime como ponto de partida físico.

> **Ironia diagnóstica**: os cards `dang_2022`/`bojarczuk_2023` foram arquivados como
> *baseline que "o cVAE supera"* — a narrativa era "paramétrico precisa re-calibrar,
> nosso cVAE **generaliza via condicionamento (d,c)**". O held-out V3 **refuta essa
> segunda metade**: o condicionamento caixa-preta **não** generaliza para distâncias
> não vistas. O gray-box reincorpora a física de Hammerstein/Volterra **onde** o
> condicionamento do cVAE falha — não é retroceder ao baseline, é a **síntese** dos
> dois.

## 5. Possibilidades — losses a rever vs arquitetura

| caminho | o que muda | veredito |
|---|---|---|
| Losses de saída (relativa / het / suavidade / coverage) | pressionam a saída nos pontos vistos | **ESGOTADO** — 4 negativos; nenhuma injeta física |
| Loss de **consistência física** (amarrar `a(d)`∝1/d² entre regimes) | termo penalizando desvio da lei | band-aid: ajuda, mas a arquitetura segue caixa-preta |
| **Gray-box físico (arquitetura)** ⭐ | `μ = a(d,c)·X` com `a,σ` **paramétricos físicos** + parte estocástica só no resíduo benigno | **interpola por construção**; precedente nas branches graybox |
| Repensar a necessidade do cVAE | canal Gaussiano → regressão heterocedástica com momentos físicos JÁ é o espelho | cVAE vira casca fina (opcional) sobre o resíduo |

## 6. Resposta direta: "um cVAE resolveria esse problema?"
- **cVAE caixa-preta puro: NÃO** — estruturalmente incapaz de generalizar os
  momentos condicionais para distâncias não vistas (os 4 experimentos são a prova).
- **Modelo físico-informado / gray-box (que pode ainda *vestir* um cVAE no
  resíduo): SIM** — a lei de propagação dá os momentos certos em qualquer
  distância, e as intermediárias **passam** como prova legítima de generalização.

## 7. Recomendação
**Reviver a linha gray-box.** O core do twin deixa de ser `MLP(d,c)` e passa a ser
um **modelo de canal OWC paramétrico** (`a(d,c)` geométrico×LED, `σ(d,c)` de ruído),
com a parte estocástica/cVAE modelando só o resíduo Gaussiano. Isso (1) injeta a
física que falta → interpola correto nas não-vistas; (2) reaproveita o diagnóstico
(canal = ganho linear + Gaussiano) em vez de lutar contra ele; (3) já tem esqueleto
nas branches `feat/seq-imdd-graybox-mdn` / `feat/imdd-graybox-channel` /
`feat/channel-residual-architecture`.

## 8. Próximo passo decisivo (barato, antes de construir)
**Teste de validação física (CPU, sem treinar nada)**:
1. Ajustar `a(d,c)` (ganho) e `σ(d,c)` (escala de ruído) por regime nas **4 âncoras**
   de treino (já saem do `regime_census` / da equalização `a=<X,Y>/<X,X>`).
2. Ajustar a forma paramétrica `a(d,c) = K·h_LED(c)/d²` (e o modelo de σ) a essas 4.
3. **Prever as 3 não-vistas** com a lei ajustada e comparar `a_pred(d)` vs `a_real(d)`
   medido, e os momentos resultantes contra os gates.

Se a lei física **acerta as não-vistas**, a tese está confirmada → vale construir o
gray-box completo. Se errar, a física não é `1/d²` simples e o problema é mais
profundo (multipath / geometria) — e aí o veredito é honesto antes de gastar GPU.

## 9. Perguntas em aberto / a validar
- Forma exata de `a(d)`: `1/d²` LOS puro, ou há termo de ângulo/FOV/near-field?
- `h_LED(c)`: a saturação do LED é estática (Hammerstein) e separável de `d`?
- `σ(d,c)`: shot (∝√potência) + térmico (constante)? casa com a Camada 2?
- O resíduo após remover `a(d,c)·X` é mesmo i.i.d. Gaussiano (Camada 0 diz que sim
  — kurt≈0, het≈0 no equalizado)? Confirmar nas não-vistas.

---

## 10. Conciliação com a tese (TESE/02 + knowledge) — o gray-box NÃO contradiz o capítulo
A justificativa generativa da tese (`TESE/02_fundamentacao/fundamentos_vlc_imdd.md`)
afirma que *"a distribuição de saída não pode ser resumida adequadamente por uma média
condicional e um erro quadrático"* → logo, modelagem **gerativa** em vez de regressão.
O diagnóstico V3 cria uma **tensão aparente**: intra-regime o resíduo é Gaussiano
benigno (kurt≈0), ou seja, *intra-regime* o canal quase **é** média+σ. Resolução:

1. **O "não-redutível a média+MSE" muda de lugar, não some.** Não é a cauda do
   resíduo (V3 mostra que é leve) — é o fato de que **os momentos condicionais
   `a(d,c)`, `σ(d,c)` são função não-linear de (d,c)** que um **único** regressor MSE
   global não interpola. Essa é uma justificativa **mais forte e mais honesta** para um
   modelo estruturado do que "caudas pesadas": um MSE global falha; só um modelo com a
   **estrutura física entre regimes** acerta as não-vistas. A tese deve **re-enunciar**
   a pergunta de ciência nesses termos (momentos condicionais vs. forma do resíduo).

2. **A parte gerativa continua justificada — em fidelidade, não em cauda.** O valor
   gerativo que sobrevive ao V3 é **calibração/cobertura de σ (G3/coverage)**,
   **estrutura espectral/temporal do resíduo (G4 PSD — resíduo pode ser colorido, não
   branco)** e **estatística conjunta (G5/G6)**. O gray-box **preserva** isso: física
   nos momentos `a(d,c)·X`, `σ(d,c)` (interpola) + componente estocástica/cVAE **só no
   resíduo** (fidelidade que os gates exigem). É híbrido, não regressão pura.

3. **A diferenciabilidade — e portanto o roadmap e2e — sobrevive (`cooke_2022`).**
   "Differentiable Channels Are All You Need": para usar o twin como **canal proxy
   diferenciável** em treino e2e, a *forma* é secundária; basta haver gradiente. Um
   gray-box `a(d,c)·X + N(0,σ(d,c))` (com `a,σ` paramétricos físicos) é **diferenciável
   por construção**. Trocar MLP-caixa-preta por física **não custa** o proxy e2e — o
   benefício que motivava o cVAE permanece.

4. **O knowledge já apontava para cá.** Os "Follow-up Actions" de `dang_2022`/
   `bojarczuk_2023` pedem literalmente *"extrair os parâmetros do LED e comparar com o
   que o decoder aprende por regime"* — o cruzamento físico↔aprendido já era o próximo
   passo previsto. O gray-box apenas **inverte a hierarquia**: a física deixa de ser
   item de comparação e passa a ser o **esqueleto** do twin.

> **Síntese de uma linha**: a tese estava certa em rejeitar regressão MSE pura, mas pela
> razão errada (caudas). A razão certa é **interpolação dos momentos condicionais** — e
> a resposta certa é o **gray-box físico + resíduo gerativo**, que honra os gates de
> fidelidade (G4/G5/G6) E generaliza para as distâncias-prova (G1/G3).
