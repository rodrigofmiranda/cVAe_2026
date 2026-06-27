# Plano de tese — escada experimental de baixo pra cima (digital twin de canal OWC)

> **Status**: princípio de partida (2026-06-27). Reorganiza a NARRATIVA experimental
> a partir de um primeiro princípio, **reaproveitando** infra, diagnóstico e knowledge
> já existentes. Não é recomeçar o projeto — é fundamentar cada decisão por evidência,
> de baixo pra cima, para a defesa de doutorado.
> Fontes-mãe: [CRITICAL_PROJECT_KNOWLEDGE.md](CRITICAL_PROJECT_KNOWLEDGE.md) ·
> [REPROJETO_GRAYBOX_OWC.md](REPROJETO_GRAYBOX_OWC.md).

---

## 0. Por que esta escada
A linha anterior foi (sem querer) **de cabeça pra baixo**: atacou a **generalização de
distância** (o problema mais difícil) com uma **loss complexa** (NLL + MMD + energy +
coverage + axis + quantile + termos experimentais rel/het/smooth) **antes** de ter
provado (a) a **fidelidade de um único regime** e (b) qual é a **loss mínima
suficiente**. Para uma tese, cada peça precisa ser **justificada por uma deficiência
medida**, não herdada. Esta escada inverte: fidelidade primeiro, complexidade só sob
demanda, generalização por último.

### O que NÃO reiniciar (já é resultado consolidado, citar a fonte)
- **Infra**: pipeline `src/protocol/run.py`, gates régua-V3, docker CPU/GPU, dataset V3
  FC/FS, knowledge base.
- **Diagnóstico do canal** (macro_diagnostics, `regime_census.py`, held-out): **intra-
  regime o canal é trivial** — ~85% ganho **linear** + ~15% ruído **Gaussiano** +
  **0% não-linear** (até em 0.75m/900mA), kurt_resíduo≈0. A não-linearidade vive
  **ENTRE** regimes: `a(d,c)` e `σ(d,c)`.
- **Física medida** (`physics_interp_test.py`): `a(d,c) ≈ g(c)·d_m^p` separável
  (Hammerstein), **p medido ≈ −1.82** (≈ Lambertiano), `g(c)` pico ~400–500 mA (curva
  do LED), **σ ≈ const ~0.053** (independente da distância).

> **Implicação direta para a loss**: como o resíduo intra-regime é Gaussiano benigno,
> a **hipótese nula** é que **a NLL Gaussiana basta** para fidelidade. Todo termo extra
> precisa provar que compra uma métrica que a NLL sozinha não entrega.

---

## 1. Regra metodológica transversal (vale para todas as fases)
1. **Loss mínima → ablação aditiva.** Começar com **NLL Gaussiana pura**. Adicionar um
   termo (MMD/energy/coverage/axis/quantile/PSD) **somente** se a métrica-alvo daquele
   termo falhar a régua **sem** ele. Registrar a **tabela de ablação** (termo → métrica
   que ele move → ganho medido). Sem ganho medido ⇒ termo fora.
2. **Régua de validação canônica** (mesma em todas as fases, ver §5): distribuição
   (gates V3) + comunicação (QAM/BER vs real **e** vs AWGN) + espectral/temporal.
3. **AWGN é o modelo nulo.** O twin só se justifica se **reproduz o EXCESSO** do canal
   real sobre o AWGN (penalidade/forma que o AWGN não captura). Se AWGN já basta num
   regime, registrar isso honestamente.
4. **Não-determinismo**: ≥2 seeds por configuração; descartar seeds com bacia ruim
   (val_recon ~−3.5) como divergência, não como veredito. (cf. lição A4/B3/basin.)
5. **Uma variável por vez.** Mudar arquitetura **ou** loss **ou** dados — nunca dois
   juntos no mesmo experimento.

---

## 2. Fase 0 — UM regime (1 distância, 1 corrente)
**Pergunta**: o twin reproduz `p(Y|X)` de um único ponto de operação, e qual é a régua?

- **Setup**: FC, escolher um regime "central" bem-comportado (sugestão: `1.0 m / 400 mA`
  — SNR alto, longe das bordas; documentar a escolha). Treino/val split intra-regime.
- **Modelos a comparar**: (a) **AWGN** ajustado (σ casado ao real) = nulo; (b) twin
  **NLL Gaussiana pura**; (c) só se (b) falhar uma métrica, ligar o termo que a endereça.
- **O que medir** (régua §5 aplicada a 1 regime): match de distribuição (G5/G6),
  EVM/SNR (G1/G2, inferência determinística), PSD (G4), e **curvas de BER vs SNR para
  4/16/64-QAM**: real × twin × AWGN.
- **Critério de sucesso**: twin ≈ real em todas as métricas **E** twin ≈ real melhor que
  AWGN onde houver excesso (ou: AWGN basta e isso fica documentado).
- **Deliverable**: "o twin reproduz um link OWC; a NLL Gaussiana é/não é suficiente;
  cada métrica/loss-term está definido e medido." + a 1ª linha da tabela de ablação.
- **Armadilha**: este regime é **fácil** (canal trivial intra-regime) → é um **portão
  rápido de sanidade e calibração da régua**, NÃO um lugar para morar. Se algo simples
  falhar aqui, é bug de pipeline, não de ciência.

---

## 3. Fase 1 — varredura de CORRENTE (1 distância, todas as correntes)
**Pergunta**: o modelo **condiciona** e aprende a não-linearidade do LED `h_LED(c)`?

- **Por que esta é a 1ª fase substantiva**: as correntes são a parte **fisicamente
  rica** (saturação do LED, Hammerstein) e são **100% observáveis** — não há problema de
  generalização. É onde o cVAE *ganha o salário* sobre um único Gaussiano marginal.
- **Setup**: FC, `1.0 m`, todas as 9 correntes (100–900 mA). Condicionamento em `c`.
- **Ablação central**: (a) o condicionamento em `c` funciona (vs um modelo sem
  condicionamento = baseline pobre)? (b) a loss precisa de mais que NLL para casar a
  variação de `σ(c)`/forma com a corrente? (c) o `g(c)` aprendido bate com o medido
  (pico ~400–500 mA)?
- **O que medir**: régua §5 **por corrente** + a curva `a(c)`/`g(c)` aprendida vs medida
  (sanidade física) + BER por corrente (real/twin/AWGN).
- **Critério para passar de fase**: todas (ou quase) as correntes passam a régua **na
  distância treinada**, e o twin reproduz a tendência `h_LED(c)`.
- **Deliverable**: "o twin condiciona e captura a não-linearidade do LED; aqui está a
  loss mínima que faz isso." (tabela de ablação consolidada.)

---

## 4. Fase 2 — DISTÂNCIA (com held-out = a prova de generalização)
**Pergunta**: o twin **generaliza** para distâncias NÃO vistas {0.9, 1.16, 1.25} m?

- **Aqui — e só aqui — entra o gray-box** (`physical_gain`, `a=g(c)·d_m^p`, p aprendido),
  já implementado e em teste (ver [REPROJETO_GRAYBOX_OWC.md](REPROJETO_GRAYBOX_OWC.md);
  commit 2410ce7). O muro empírico (E1 30 / A1 29 / A4 29 / B3 26) mostrou que loss/prior
  sobre as âncoras não basta; a física do ganho prevê os held-out a 1.4%.
- **Setup**: treino {0.75, 1.0, 1.35, 1.5} m × todas correntes; **held-out NUNCA
  treinado** {0.9, 1.16, 1.25} m.
- **Critério de sucesso**: held-out **1.16/1.25 m passam o G3** (E1 dava 0/9); `n_pass/63`
  ≥ baseline; `p` converge p/ ~−1.82 e `a(d,c)` aprendido bate com o medido; a física
  **estabiliza a bacia** entre seeds (black-box deu 27 vs 2/36).
- **Deliverable**: "o twin generaliza por construção física; as distâncias-prova passam"
  — a contribuição central da tese.

---

## 5. Régua de validação (mesma em todas as fases)
Três camadas, sempre as três:

| camada | métrica | gate/forma | o que prova |
|---|---|---|---|
| **Distribuição** | EVM rel (G1), SNR rel (G2), média/σ rel (G3) | régua-V3 | momentos condicionais certos |
| | skew/kurt (G5), MMD/energy (G6), PSD (G4) | régua-V3 | forma + espectro do resíduo |
| **Comunicação** | BER vs SNR, 4/16/64-QAM | **real × twin × AWGN** | o twin é proxy melhor que AWGN |
| **Física** (sanidade) | `a(d,c)`, `σ(d,c)`, `g(c)`, `p` aprendidos | vs medido (`physics_interp_test`) | o que o modelo aprendeu é físico |

Ferramentas que já existem: gates em `src/protocol`, `comparison_v3/macro_diagnostics`
(census, AWGN, modulações/BER, cross-dist), `physics_interp_test.py`. **Não reescrever**
— reusar e, por fase, restringir ao subconjunto de regimes daquela fase.

---

## 6. Tensão a manter à vista (o coração da tese)
"Com dados suficientes ele aprende o canal" tem um risco: **densificar o grid** vira
**interpolação-por-memorização**, não um twin que *aprende* o canal. A tese forte é
**generalização** — as distâncias intermediárias são a **prova** e por isso **não podem
entrar no treino**. A escada separa limpo: **Fases 0–1 = fidelidade**; **Fase 2 =
generalização**. As duas são necessárias para a defesa.

---

## 7. Critérios de "passar de fase" (não pular etapa)
- **0 → 1**: 1 regime passa a régua completa; régua e ablação inicial documentadas.
- **1 → 2**: correntes na distância treinada passam; `g(c)` aprendido ≈ medido; loss
  mínima fixada por ablação.
- **2 → tese**: held-out passam o G3; `a(d,c)` aprendido ≈ medido; resultado estável
  entre seeds.

> Regra de ouro contra paralisia de recomeço: Fase 0 é **curta e decisiva** (o canal
> intra-regime é trivial). O valor é a **narrativa fundamentada por ablação**, não
> queimar compute refazendo fundações.
