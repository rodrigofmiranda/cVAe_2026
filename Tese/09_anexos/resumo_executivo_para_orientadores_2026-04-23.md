# Resumo Executivo Para Orientadores (2026-04-23)

## Propósito

Oferecer uma visão curta, rastreável e apresentável do estado atual da
pesquisa, destacando as branches ativas, os commits mais recentes, o objetivo de
cada linha e o artefato principal que sustenta a leitura científica atual.

## Escopo

Este resumo cobre apenas o estado consolidado mais recente das três frentes
operacionais diretamente envolvidas na rodada atual:

- `full_square`
- `full_circle`
- camada editorial `Tese/mdn_return`

Ele não substitui os documentos analíticos completos. Funciona como ponto de
entrada para reunião, acompanhamento e prestação de contas da evolução recente.

## Fontes canônicas usadas

- [full_square branch head synthesis](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/full_square_vs_full_circle_master_table_2026-04-23.md)
- [full_square last commit reference](/home/rodrigo/cVAe_2026_full_square/src/models/cvae_sequence.py#L729)
- [full_circle project status](/home/rodrigo/cVAe_2026_shape_fullcircle/PROJECT_STATUS.md)
- [full_circle 16QAM crossline summary](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260422_plus_soft_radial/crossline_summary/README.md)
- [thesis crossline note](/home/rodrigo/cVAe_2026_mdn_return/Tese/06_validacao_do_gemeo/comparacao_crossline_full_square_vs_full_circle_2026-04-23.md)
- [thesis 16QAM note](/home/rodrigo/cVAe_2026_mdn_return/Tese/06_validacao_do_gemeo/validacao_externa_16qam.md)
- [thesis results by line](/home/rodrigo/cVAe_2026_mdn_return/Tese/07_resultados_e_decisoes/resultados_por_linha.md)

## Status do texto

`curado`

## Quadro Executivo

| Frente | Worktree | Branch | Commit mais recente | Objetivo do último ciclo | Leitura estabilizada | Artefato principal |
| --- | --- | --- | --- | --- | --- | --- |
| `full_square` | `/home/rodrigo/cVAe_2026_full_square` | `research/full-square` | `f6623b6` | corrigir o loader do `legacy2025`, fechar o rerun limpo e consolidar a leitura crossline `full_square` versus `full_circle` | `best_compare_large` segue como melhor balizador externo do lado `full_square`; `legacy2025` voltou a ser comparável, mas ficou fraco | [full_square_vs_full_circle_master_table_2026-04-23.md](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/full_square_vs_full_circle_master_table_2026-04-23.md) |
| `full_circle` | `/home/rodrigo/cVAe_2026_shape_fullcircle` | `research/full-circle` | `40a12c8` | endurecer a infraestrutura de avaliação, tornar paths portáveis, separar testes estatísticos auxiliares da GPU principal e consolidar a comparação externa `16QAM` com o braço `soft-radial` | `soft_rinf_local` é hoje o melhor compromisso metodológico dentro de `full_circle`; `disk_geom3` continua como teto operacional; `full_square` ainda vence na leitura externa global | [README.md](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260422_plus_soft_radial/crossline_summary/README.md) |
| `Tese / mdn_return` | `/home/rodrigo/cVAe_2026_mdn_return` | `research/mdn-return-20260416` | `081c1a7` | transformar o estado recente em material apresentável e rastreável para a tese, incluindo validação externa `16QAM`, leitura crossline e regras operacionais de dual-run | a tese já separa com clareza: validação principal do twin, validação externa e linha histórica descartada por desempenho real | [comparacao_crossline_full_square_vs_full_circle_2026-04-23.md](/home/rodrigo/cVAe_2026_mdn_return/Tese/06_validacao_do_gemeo/comparacao_crossline_full_square_vs_full_circle_2026-04-23.md) |

## Cinco Pontos Para Apresentação Oral

1. A pesquisa está organizada em duas linhas principais: `full_square` e
   `full_circle`, com uma terceira frente apenas editorial, dedicada à
   consolidação da tese.
2. No protocolo principal do twin, `full_circle` ainda preserva interesse
   científico real, sobretudo via `soft_rinf_local` e pelo teto operacional
   `disk_geom3`.
3. Na validação externa em `16QAM`, o melhor representante atual de
   `full_square` continua sendo superior na leitura global.
4. O caso `legacy2025` foi fechado metodologicamente: o problema anterior era
   de loader, não de inexistência de modelo; após correção, o desempenho real
   da família se mostrou fraco.
5. A camada `Tese/` já consegue sustentar a narrativa recente sem depender de
   memória oral ou histórico de chat.

## Leitura Integrada Atual

A leitura mais madura que pode ser apresentada hoje é:

- `full_square` lidera quando o critério é generalização externa em `16QAM`
- `full_circle` ainda preserva valor científico quando o critério é o protocolo
  principal do twin
- `soft_rinf_local` é o melhor compromisso metodológico atual dentro da família
  `full_circle`
- `disk_geom3` deve ser tratado como teto operacional, não como baseline neutra
- `legacy2025` pode permanecer como referência histórica, mas não como
  candidata competitiva atual

## Onde Abrir Na Reunião

Se houver pouco tempo, a ordem mais eficiente é:

1. [comparacao_crossline_full_square_vs_full_circle_2026-04-23.md](/home/rodrigo/cVAe_2026_mdn_return/Tese/06_validacao_do_gemeo/comparacao_crossline_full_square_vs_full_circle_2026-04-23.md)
2. [full_square_vs_full_circle_master_table_2026-04-23.md](/home/rodrigo/cVAe_2026_full_square/knowledge/syntheses/full_square_vs_full_circle_master_table_2026-04-23.md)
3. [README.md](/home/rodrigo/cVAe_2026_shape_fullcircle/outputs/analysis/eval_16qam_crossline_20260422_plus_soft_radial/crossline_summary/README.md)
4. [resultados_por_linha.md](/home/rodrigo/cVAe_2026_mdn_return/Tese/07_resultados_e_decisoes/resultados_por_linha.md)

## Implicações

Este resumo torna a conversa com orientadores mais objetiva porque reduz o
estado atual a três perguntas simples:

- em qual branch está cada frente?
- qual foi a última entrega concreta?
- qual artefato deve ser aberto para discutir evidência, e não opinião?
