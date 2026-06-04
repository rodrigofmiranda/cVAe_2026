# Best-runs inventory — cVAE digital twin (full_square)

Snapshot generated 2026-06-04T03:23:44Z by `scripts/twin_search/inventory_best_runs.py`.
Regenerate any time with:

```bash
python3 scripts/twin_search/inventory_best_runs.py
```

A run is "good basin" when `val_recon_loss <= -4.6` (the bad basin never reaches
below ~-3.98; the gap is clean). **Good basin is necessary but NOT sufficient** for
a good digital twin — `n_pass` (gates passed / 12) varies from 0 to 10 among
good-basin runs. The acceptance metric is `mini_protocol_v1` gates (G1..G6),
target `n_pass >= 10`.

Different-geometry lines (`full_circle`, `shape_fullcircle`) are excluded: their
val_recon scale is not comparable to full_square.

## Current best runs (ranked by n_pass)

```
# Best-runs inventory (full_square) — 15 good-basin runs (val_recon <= -4.6)

n_pass      min   eps    gates G1..G6  tag                        clone / exp
------------------------------------------------------------------------------------------------------------------------
    10   -4.978   492  12/12/12/12/10/12  S39B_edgegap_lowlr_all08_w eduardo/cVAe_2026 / exp_20260427_141943
    10   -4.965   307  10/10/10/12/10/4  S39B_edgegap_lowlr_all08_w rodrigo/cVAe_2026_full_square / exp_20260527_221226
     9   -4.979   500   9/9/9/12/10/8  S38D_smplmmd_cov30_t02_tai rodrigo/cVAe_2026_full_square / exp_20260526_220848
     8   -4.977   500  12/12/10/12/9/8  S41B_s39b_seed123          eduardo/cVAe_2026 / exp_20260429_160736
     8   -4.976   393  12/12/11/12/9/9  S39B_edgegap_lowlr_all08_w rodrigo/cvae_repro_141943 / exp_20260524_225922
     7   -4.977   500  12/12/10/12/10/7  S39B_edgegap_lowlr_all08_w rodrigo/cvae_repro_141943 / exp_20260602_203631
     7   -4.976   500  12/12/12/12/12/7  S35C_fast_e64_base         rodrigo/cVAe_2026_mdn_return / exp_20260416_204135
     6   -4.976   500  12/12/11/12/8/7  S40C_g5_pairfocus_w26_cov3 eduardo/cVAe_2026 / exp_20260428_212906
     6   -4.972   500     8/8/8/8/6/6  S43C_0p8mAnch_w14_lowlr6e5 eduardo/cVAe_2026 / exp_20260504_142103
     3   -4.969   206  12/12/11/12/9/4  S42C_s39b_lowlr6e5_bs8192_ eduardo/cVAe_2026 / exp_20260430_144657
     3   -4.968   248     8/8/7/8/4/5  S44A_0p8mAnchPair_ctrl_low eduardo/cVAe_2026 / exp_20260505_035318
     2   -4.825   366    7/8/5/12/3/2  G3_lat6_b0p001_fb0p10_lr0p eduardo/cVAe_2026 / exp_20260425_180613
     0   -4.979   500  12/12/12/12/8/0  S38D_smplmmd_cov30_t02_tai eduardo/cVAe_2026 / exp_20260423_174212
     0   -4.977   422  12/12/12/12/11/0  S39B_edgegap_lowlr_all08_w eduardo/cVAe_2026 / exp_20260424_122014
     0   -4.924   174     0/0/0/0/1/0  S39B_edgegap_lowlr_all08_w eduardo/cVAe_2026 / exp_20260425_162033
```

## Key facts

- **Champion = 10/12**, reached TWICE, both with config **S39B**:
  eduardo `exp_20260427_141943` and rodrigo `cVAe_2026_full_square/exp_20260527_221226`.
- **S38D** (a different config: sampled-MMD + cov30) reached **9/12** — alternative base.
- The same S39B config yields 10, 8, 7, 0 across runs → the gate result is itself a
  stochastic draw (training basin + Monte-Carlo eval seed), not config alone.
- Both failures in eduardo's 10/12 are **G5** at 0.8 m (300 mA: tail kurtosis
  OVERSHOOT, jb -29 vs real -4.6; 500 mA: marginal). See
  [[REPRODUCIBILITY_DETERMINISM_141943.md]] and `TWIN_SEARCH_PROGRESS.md`.
