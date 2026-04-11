# Support Hyperparameter Scientific Screening Protocol (E2 Line)

## Objective

This note defines the next exploratory grid for the `shape` line in a way that
is scientifically defensible and thesis-friendly.

The intent is **not** to run a naive brute-force search. The intent is to:

1. keep the strongest current family fixed (`E2`, edge-only),
2. perturb one *family of hypotheses* at a time,
3. discard weak hyperparameter directions with evidence,
4. preserve a narrative that can later be written cleanly in a thesis chapter.

## Why This Design Instead of a Giant Cartesian Grid

A giant fully crossed search is expensive and hard to interpret:

- too many interactions at once,
- harder to attribute causality,
- easier to find one lucky winner without learning which family mattered.

A pure one-factor-at-a-time design is also weak:

- it underestimates interactions,
- it tends to miss meaningful local bundles.

The compromise used here is a **blocked local screening design**:

- one robust control,
- blocks of candidates grouped by scientific hypothesis,
- each block changes one family of knobs while freezing the rest.

This is not globally optimal design-of-experiments theory, but it is a strong
practical compromise for deep-learning channel modeling where each full-data run
is expensive.

## Fixed Experimental Conditions

Unless explicitly overridden:

- dataset: `data/dataset_fullsquare_organized`
- regimes: `configs/protocol_default.json` (`12` regimes)
- data volume: full data, no caps
- base family: `S27` / `seq_bigru_residual` / `mdn`
- support mode: `edge_rinf_corner`
- statistical mode for screening: `quick`
- early stopping: `patience=50`
- LR plateau: `reduce_lr_patience=25`

The control configuration is:

- `lambda_coverage=0.25`
- `tail_levels=[0.05, 0.95]`
- `coverage_temperature=0.03`
- `support_weight_alpha=1.50`
- `support_weight_tau=0.75`
- `support_weight_tau_corner=0.35`
- `support_weight_max=3.0`

## Primary and Secondary Endpoints

### Primary endpoints

These should decide whether a family survives:

- `n_pass`
- `gate_g5_pass`
- `gate_g6_pass`

### Secondary endpoints

These arbitrate ties and prevent us from promoting a pathological winner:

- `mean_cvae_rel_evm_error`
- `mean_cvae_rel_snr_error`
- `mean_cvae_mean_rel_sigma`
- `mean_cvae_cov_rel_var`
- `mean_cvae_psd_l2`
- `mean_stat_mmd_qval`
- `mean_stat_energy_qval`

## Decision Rules

### Promote a candidate

Promote a candidate to confirmation if **any** of these hold:

1. `n_pass` improves by at least `+1` over control.
2. `n_pass` ties control, but `gate_g5_pass` or `gate_g6_pass` improves by at
   least `+1` without a severe regression in the secondary metrics.
3. `n_pass`, `G5`, and `G6` tie, but the candidate is clearly better in a
   physically important bundle:
   - lower `rel_evm_error`,
   - lower `cov_rel_var`,
   - lower `psd_l2`,
   - lower `mean_rel_sigma`.

### Discard a family

Discard a whole family when **all** candidates in that block satisfy:

- `n_pass <= control`,
- `gate_g5_pass <= control`,
- `gate_g6_pass <= control`,
- and none shows a convincing secondary-metric advantage.

This is the main thesis value of the design: even “negative” blocks become
useful evidence.

## Candidate Matrix

### Control

| Tag | Purpose | Hypothesis |
| --- | --- | --- |
| `S27cov_sciv1_ctrl_lc0p25_t0p03_a1p50_tau0p75_tc0p35_wmax3p0` | Robust anchor | Keeps the strongest current `E2` full-data configuration as the reference point for all other candidates. |

### Block A: Support-weight localization

Question: is the remaining error because edge pressure is too broad, too local,
or too corner-focused?

| Tag | Changed knobs | Hypothesis |
| --- | --- | --- |
| `S27cov_sciv1_edgebroad_a1p25_tau0p70_tc0p30_wmax2p8` | broader and softer weighting | If the model is over-focusing rare extremes, broader weighting should improve global fidelity without losing G5/G6. |
| `S27cov_sciv1_edgelocal_a1p50_tau0p80_tc0p40_wmax3p0` | more localized edge activation | If the model is still under-attending the true perimeter, tighter localization should help G5. |
| `S27cov_sciv1_cornerhard_a1p75_tau0p82_tc0p45_wmax3p2` | stronger and more corner-specific weighting | If the remaining mismatch is corner-dominated, explicit corner pressure should help more than the baseline. |

### Block B: Coverage and tail calibration

Question: is the bottleneck caused by how strongly we calibrate coverage/tails,
or by *which* tails and central bands we ask the model to match?

| Tag | Changed knobs | Hypothesis |
| --- | --- | --- |
| `S27cov_sciv1_covsoft_lc0p20_t0p035` | softer coverage pressure | If the current control is over-constrained, softer coverage should recover physical fidelity. |
| `S27cov_sciv1_covhard_lc0p30_t0p025` | stronger, sharper coverage pressure | If the model is still too loose in the tails, harder coverage should improve G5/G6. |
| `S27cov_sciv1_tail98_lc0p25_t0p03` | more extreme tail targets | If rare-event underfitting dominates, explicitly targeting 2/98 tails should help. |
| `S27cov_sciv1_covdense_cg50-75-90-95_lc0p25_t0p03` | denser central coverage levels | If the issue is not just tail mass but how the center transitions into the edge, denser coverage levels should help. |

### Block C: Model capacity

Question: is the current `S27/E2` line still under-capacity in the latent,
recurrent, or MLP heads?

| Tag | Changed knobs | Hypothesis |
| --- | --- | --- |
| `S27cov_sciv1_lat10` | `latent_dim=10` | If the latent bottleneck is still compressing edge/tail structure too much, a larger latent should help. |
| `S27cov_sciv1_h96` | `seq_hidden_size=96` | If temporal memory is under-modeled, a wider recurrent state should help. |
| `S27cov_sciv1_L192-384-512` | wider MLP heads | If the seq backbone is fine but the conditional heads are too narrow, wider heads should help. |

### Block D: Optimization and KL regularization

Question: is the current performance limited less by representational capacity
and more by optimization bias or KL pressure?

| Tag | Changed knobs | Hypothesis |
| --- | --- | --- |
| `S27cov_sciv1_lr0p00015` | lower learning rate | If the baseline is slightly too aggressive, a lower LR should improve stability and calibration. |
| `S27cov_sciv1_bs8192` | larger batch size | If gradient noise is hurting distribution calibration, a larger batch may help. |
| `S27cov_sciv1_b0p0015_fb0p05` | looser KL regularization bundle | If the latent pathway is over-regularized, easing KL pressure should improve G5/G6. |

## Why Some Knobs Stay Frozen in V1

The grid does **not** reopen every knob.

The following stay frozen because current evidence says they are either already
well matched or too infrastructural for this stage:

- `activation=leaky_relu`
- `window_size=7`
- `window_stride=1`
- `window_pad_mode=edge`
- `seq_bidirectional=True`
- `seq_gru_unroll=True`

The following are provisionally frozen because they have enough positive
historical evidence that they should not be reopened before this screening:

- `arch_variant=seq_bigru_residual`
- `decoder_distribution=mdn`
- `mdn_components=3`

## Execution Snapshot (2026-04-11)

The blocked screen moved from design into execution in four independent blocks.

Current status:

- `A`: completed clean rerun, scientifically usable
- `B`: completed clean rerun, scientifically usable
- `C`: completed clean rerun, scientifically usable
- `D`: completed clean rerun, scientifically usable

High-level result so far:

- `A` beat `B` on the primary protocol endpoint (`5/12` vs `4/12`)
- the `A` winner was the control itself
- the `B` winner was `tail98`, which helped statistical acceptance but did not
  surpass the control on the main protocol criterion
- `C` did not beat the `A` control; `lat10` won the block at `4/12`
- `D` did not beat the `A` control either, but `lr0p00015` emerged as the
  strongest non-`A` follow-up because it improved `G6`

This is exactly the kind of thesis-useful negative evidence the design was
intended to produce:

- more aggressive support localization has not yet justified promotion
- stronger tail emphasis remains interesting as a secondary direction, but not
  yet as the main reference line
- larger capacity alone did not justify promotion
- optimization remains the strongest remaining follow-up family

The block-by-block master table lives in:

- `knowledge/syntheses/support_scientific_screen_master_table_2026-04-10.md`

## Reproducibility Checklist for Thesis

For each run, record:

- branch name and commit hash
- preset name
- exact control candidate
- seed
- protocol file
- dataset root
- whether full data or capped data was used
- whether `stat_mode` was `quick` or `full`
- total duration
- GPU model
- primary and secondary endpoints
- block-level conclusion:
  - promoted
  - retained for follow-up
  - discarded

## Parallel Execution Rule

If blocks are executed in parallel, each block **must** use a distinct
`OUTPUT_BASE`.

Recommended layout:

- `.../e2_scientific_screen_v1/block_a`
- `.../e2_scientific_screen_v1/block_b`
- `.../e2_scientific_screen_v1/block_c`
- `.../e2_scientific_screen_v1/block_d`

This avoids artifact collisions in:

- `manifest.json`
- `protocol_leaderboard.csv`
- `summary_by_regime.csv`
- `train/tables/gridsearch_results.csv`

Without this separation, logs may still exist, but the final experiment folder
becomes ambiguous and should not be used as a clean scientific record.

## Operational Command

Launcher:

```bash
cd /home/rodrigo/cVAe_2026_shape
scripts/ops/train_support_scientific_screen.sh --seed 42
```

If we want stronger statistics:

```bash
cd /home/rodrigo/cVAe_2026_shape
STAT_MODE=full scripts/ops/train_support_scientific_screen.sh --seed 42
```

## Expected Thesis Contribution

This screening is useful even if most candidates lose.

That is exactly the point:

- it converts hyperparameter folklore into evidence,
- it helps justify why some knobs remain frozen,
- and it creates a clean trail of rejected versus surviving hypotheses for the
  thesis narrative.
