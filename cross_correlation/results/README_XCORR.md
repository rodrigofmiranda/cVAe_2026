# Validação por Cross-Correlation (3º Eixo)

Este diretório contém os resultados da validação do gêmeo digital do canal VLC através do método de **cross-correlation entrada-saída**, correspondendo ao terceiro eixo de validação da tese (ortogonal aos *gates* distribucionais e à calibração e2e).

Para os fundamentos científicos detalhados, consulte a documentação em:
[cross_correlation_fundamento_2026-06-13.md](file:///home/rodrigo/TESE/06_validacao_do_gemeo/cross_correlation_fundamento_2026-06-13.md).

---

## 1. Conteúdo do Diretório

- **`xcorr_table_FC.csv` e `xcorr_table_FS.csv`**: Tabelas contendo as métricas de whiteness de entrada, erro L2 de cross-correlation (`xcorr_l2`) e erro de atraso (`peak_lag_err`) para cada distância (treinada e não-vista).
- **`xcorr_curves_FC.png` e `xcorr_curves_FS.png`**: Gráficos comparativos no formato de duas linhas (rows) por distância.

---

## 2. Estrutura das Imagens (`xcorr_curves_*.png`)

Cada imagem apresenta uma coluna por distância de teste (âncoras treinadas + intermediárias não-vistas), dividida em duas linhas:

* **Linha 1 (Superior) — Resposta Temporal $R_{xy}(\tau)$**:
  - Compara a cross-correlation do canal real (azul, sólida) com a do gêmeo digital (vermelho, tracejada) na escala cheia de $[-0.2, 1.0]$.
  - **O que mostra**: O pico único e agudo em $\tau = 0$ comprova que o canal VLC real é praticamente **sem memória linear (LTI)** sob esta banda de excitação. A sobreposição perfeita mostra que o gêmeo digital modela perfeitamente essa dinâmica instantânea e zera o erro de latência (`peak_lag_err = 0`).
  
* **Linha 2 (Inferior) — Erro Residual ($R_{xy}^{real} - R_{xy}^{twin}$)**:
  - Plota a diferença ponto a ponto (laranja) em uma escala y compartilhada e altamente ampliada ($\approx \pm 0.007$).
  - **O que mostra**: Revela o erro microscópico residual (RMS $< 0.5\%$). A amplitude deste erro cresce de forma contínua e linear à medida que a distância aumenta de $0.75\text{ m}$ a $1.5\text{ m}$.

---

## 3. Conclusão Científica para a Tese (Decomposição Linear vs. Distribucional)

A análise conjunta desses resultados traz duas contribuições centrais para a validação do gêmeo:

1. **Sucesso na Interpolação Linear**: O gêmeo digital aprende e interpola perfeitamente a **resposta linear determinística** (o ganho de primeira ordem e a ausência de memória) ao longo da distância. As distâncias não-vistas no treino (`0.9 m`, `1.16 m` e `1.25 m`) caem exatamente sobre a curva de tendência monotônica das distâncias treinadas, sem sofrer saltos ou explosões de erro.
2. **Isolamento da Falha dos Gates**: Este eixo isola com precisão onde ocorre a falha de não-interpolação do gêmeo digital relatada nos *gates* distribucionais (G1–G6). A falha é de natureza **estocástica/distribucional** (lei do ruído e modelagem da variância residual) e não tem relação com o ganho físico determinístico ou com a resposta dinâmica temporal, que são interpolados com extrema precisão.
