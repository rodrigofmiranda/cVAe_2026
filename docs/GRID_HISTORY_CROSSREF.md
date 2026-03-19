# Grid History Cross Reference

Artefatos gerados pelo cruzamento:
- `outputs/analysis/grid_history/tested_grid_catalog.csv`
- `outputs/analysis/grid_history/champion_runs_crossref.csv`
- `outputs/analysis/grid_history/config_signature_rollup.csv`
- `outputs/analysis/grid_history/cross_reference_summary.md`

## Estado Atual

- runs com gate summary: `73`
- runs com champion ligado a `gridsearch_results.csv`: `16`
- assinaturas exatas unicas recuperadas: `14`
- nenhuma assinatura passou `G1..G6` ao mesmo tempo nos gates atuais

## Melhor Cobertura De Gates

Historico geral:
- `seq_bigru_residual`
  - `S1seq_W7_h64_lat4_b0p003_fb0p10_lr0p0003_L128-256-512`
  - passou `4/6` gates: `G2 G3 G4 G5`
  - falhou `G1` e `G6`

Linha point-wise residual simplificada:
- `delta_residual`
  - `D1delta_lat4_b0p001_fb0p0_lr0p0003_L128-256-512`
  - passou `3/6` gates: `G3 G4 G5`
  - falhou `G1 G2 G6`

Melhor `score_v2` dentro da linha `delta_residual`:
- `D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512`
- melhor score historico da familia, mas com so `2/6` gates

## Repeticoes Exatas Que Nao Devem Voltar

- `delta_residual | L128-256-512 | latent=4 | beta=0.001 | free_bits=0.0 | lr=3e-4 | batch=16384 | anneal=80`
  - repetido `3x`
- `delta_residual | L128-256-512 | latent=4 | beta=0.001 | free_bits=0.1 | lr=3e-4 | batch=16384 | anneal=80`
  - repetido `2x`
- `seq_bigru_residual | W7 | h64 | latent=4 | beta=0.001 | free_bits=0.1 | batch=8192`
  - repetido `2x`
- `legacy_2025_zero_y | L32-64-128-256 | latent=16 | beta=0.1 | free_bits=0 | lr=1e-4 | batch=4096`
  - repetido `2x`
- `legacy_2025_zero_y | L32-64-128-256 | latent=16 | beta=0.1 | free_bits=0 | lr=1e-4 | batch=8192`
  - repetido `2x`

## Leitura Tecnica

Na familia `delta_residual`, o historico mostrou uma troca clara:
- configuracoes mais proximas do sinal real tendem a melhorar `G1/G2`
- configuracoes mais regulares tendem a melhorar `G3/G4/G5`
- `G6` continua sendo o gargalo em todas as familias testadas

Isso indica que o proximo grid nao deve repetir o centro ja testado. O melhor uso do budget e varrer a fronteira entre:
- `beta` mais baixo / `free_bits` maior para aproximar `EVM/SNR`
- `beta` mais alto / `free_bits` zero para preservar estrutura do residual

## Proposta De Proximo Grid

Lane principal recomendada: `delta_residual`

Objetivo:
- tentar recuperar `G1/G2` sem perder `G3/G4/G5`
- evitar repeticao exata do centro ja explorado

Parametros fixos:
- `arch_variant=delta_residual`
- `activation=leaky_relu`
- `dropout=0.0`
- `layer_sizes=[128,256,512]`
- `lr=3e-4`
- `batch_size=16384`
- `kl_anneal_epochs=80`

Parametros para abrir:
- `latent_dim in {4,5,6}`
- `beta in {0.0007, 0.00085, 0.00115}`
- `free_bits in {0.0, 0.02, 0.05}`

Preset implementado:
- `delta_residual_frontier`

Observacao:
- esse grid evita repetir exatamente os pontos ja testados em `delta_residual`
- ele mira a fronteira entre o melhor `G1/G2` observado e o melhor `G3/G4/G5` observado

Lane secundaria, se quisermos perseguir o melhor gate-count historico:
- `seq_bigru_residual`
- manter `window_size=7`, `seq_hidden_size=64`, `batch_size=8192`
- abrir `beta` em volta de `0.003`
- abrir `latent_dim` em volta de `4`
- nao repetir o ponto exato `beta=0.003, latent=4, free_bits=0.1`
