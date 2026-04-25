# Output Bucket To Architecture Translation - 2026-04-23

Purpose:
- traduzir nomes operacionais de pastas para arquitetura real, hipótese científica e caminho canônico de leitura

Scope:
- candidatos que hoje aparecem na leitura crossline `16QAM` e nas comparações recentes `full_square` vs `full_circle`

Canonical sources used:
- `outputs/architectures/_crossline/16qam/crossline_20260423_best_compare_vs_fullcircle/`
- `outputs/architectures/_crossline/16qam/crossline_20260423_with_legacy_clean/`
- `outputs/architectures/comparative/best_compare_large/full_data_sel4_overnight_20260423_040529/`
- `outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751/`
- `outputs/architectures/mdn_return/S35C_fast_e64_base/`
- `src/training/grid_plan.py`

Status:
- curated

## Translation table

| output bucket / label | what it really is | real architecture family | scientific role | canonical reading path |
| --- | --- | --- | --- | --- |
| `comparative/best_compare_large/full_data_sel4_overnight_20260423_040529` | mixed-family protocol-first study with `12` candidates (`8` seq + `4` delta) | mixed study, winner is `seq_bigru_residual` | study bucket used to decide the strongest external `16QAM` comparator on the `full_square` side | `outputs/architectures/comparative/best_compare_large/full_data_sel4_overnight_20260423_040529/` |
| `full_square_best_compare_large` | shorthand used in crossline tables for the winner imported from the mixed study above | `seq_bigru_residual` | best current `16QAM` comparator on the `full_square` side | `outputs/architectures/seq_bigru_residual/S2seq_W7_h64_lat4_b0p003_lmmd0p5_fb0p10_lr0p0003_L128-256-512/` |
| `S2seq_W7_h64_lat4_b0p003_lmmd0p5_fb0p10_lr0p0003_L128-256-512` | actual winning candidate selected inside `best_compare_large` | `seq_bigru_residual` | best overall `16QAM` model in the current crossline reading | `outputs/architectures/seq_bigru_residual/S2seq_W7_h64_lat4_b0p003_lmmd0p5_fb0p10_lr0p0003_L128-256-512/16qam/sel4_stats_20260423_040529/` |
| `legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751` | rerun limpo da linha histórica `legacy2025` | `legacy_2025_zero_y` | referência histórica válida, mas fraca no `16QAM` | `outputs/architectures/legacy_2025_zero_y/legacy2025_large_sel4_clean_rerun_20260423_114751/16qam/sel4_stats_20260423_114751/` |
| `mdn_return/S35C_fast_e64_base` | candidato da linhagem `mdn_return` fast-stage | `seq_bigru_residual` com decoder `mdn` e `cond_embed_dim=64` | linha histórica intermediária da recuperação `mdn_return`, útil como marco de trajetória | `outputs/architectures/mdn_return/S35C_fast_e64_base/16qam/sel4_stats_20260423_040225_clean/` |
| `probabilistic_shaping/S27_historical` | artefato histórico `16QAM` da linha `S27` anterior | `probabilistic_shaping` / linha `full_square` histórica | baseline histórica preservada para comparação externa | `outputs/architectures/probabilistic_shaping/S27_historical/16qam/all_regimes_s27/` |
| `clean_baseline/S27cov_fc_clean_lc0p25_t0p03_lat10` | baseline científica limpa do `full_circle` | `full_circle` clean baseline | referência neutra do lado `full_circle` | `/home/rodrigo/cVAe_2026_full_circle/outputs/architectures/clean_baseline/S27cov_fc_clean_lc0p25_t0p03_lat10/16qam/crossline_20260420_clean/` |
| `disk_geom3/S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192` | variante geometry-biased mais forte do `full_circle` | `full_circle` disk-geom3 | teto operacional do lado `full_circle` | `/home/rodrigo/cVAe_2026_full_circle/outputs/architectures/disk_geom3/S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192/16qam/crossline_20260420_clean/` |
| `soft_rinf_local/S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0` | variante geometry-light com ponderação local | `full_circle` soft-rinf-local | melhor compromisso científico atual dentro do `full_circle` | `/home/rodrigo/cVAe_2026_full_circle/outputs/architectures/soft_rinf_local/S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0/16qam/crossline_20260422_plus_soft_radial/` |

## Practical reading rule

- when a bucket name starts with `comparative/...`, read it as a study container, not as the architecture itself
- for scientific statements about the winner, cite the winner candidate tag and its architecture family
- for navigation, prefer the architecture-local alias path when it exists
- keep the original mixed-study bucket because it preserves the full competitive context and the defeated candidates

## Current conclusion

The best current `16QAM` result does **not** belong scientifically to the folder name `comparative/best_compare_large`.
That folder is only the study bucket. The winning architecture is:

- `seq_bigru_residual`
- candidate: `S2seq_W7_h64_lat4_b0p003_lmmd0p5_fb0p10_lr0p0003_L128-256-512`

Therefore, for thesis text, summaries and slides, the preferred wording is:

- "the best `16QAM` comparator on the `full_square` side was the `seq_bigru_residual` candidate `S2seq_W7_h64_lat4_b0p003_lmmd0p5_fb0p10_lr0p0003_L128-256-512`, selected inside the mixed comparative study `best_compare_large`."
