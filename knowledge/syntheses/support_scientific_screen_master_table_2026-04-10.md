# Support Scientific Screen Master Table (2026-04-10)

This note is the block-level master table for the `E2` scientific screening.
It is meant to support both day-to-day decisions and later thesis writing.

## Reading Rule

- `Promote`: keep as a main line for the next confirmation step
- `Retain`: keep as a secondary direction, but not as the main reference
- `Discard for now`: negative evidence is strong enough that the family should
  not be promoted before stronger new evidence appears
- `Open`: execution still running, decision deferred

## Master Table

| Block | Scientific question | Candidates | Status | Champion / current best | Primary result | Provisional decision |
| --- | --- | --- | --- | --- | --- | --- |
| `A` | Is the edge weight too broad, too local, or too corner-focused? | control, `edgebroad`, `edgelocal`, `cornerhard` | completed clean | `S27cov_sciv1_ctrl_lc0p25_t0p03_a1p50_tau0p75_tc0p35_wmax3p0` | `5/12` pass | `Promote` control, `Discard for now` the more aggressive localization variants |
| `B` | Is the bottleneck mainly coverage/tail calibration? | control, `covsoft`, `covhard`, `tail98`, `covdense` | completed clean | `S27cov_sciv1_tail98_lc0p25_t0p03` | `4/12` pass | `Retain` `tail98` as a secondary line, `Discard for now` `covsoft`, `covhard`, `covdense` |
| `C` | Is the model still under-capacity in latent, recurrent, or head width? | control, `lat10`, `h96`, `L192-384-512` | completed clean | `S27cov_sciv1_lat10` | `4/12` pass | `Discard for now` as a mainline promotion; capacity changes did not beat the `A` control |
| `D` | Is performance limited by optimization bias or KL pressure? | control, `lr0p00015`, `bs8192`, `b0p0015_fb0p05` | completed clean | `S27cov_sciv1_lr0p00015` | `4/12` pass | `Retain` `lr0p00015` as the strongest secondary follow-up after the `A` control |

## Block A Summary

Artifacts:

- experiment: `outputs/support_ablation/final_grid/e2_scientific_screen_v1/block_a_clean/exp_20260409_120752`
- log: `outputs/_launch_logs/run_support_scientific_block_a_clean_20260409_010200.log`

Winner:

- `S27cov_sciv1_ctrl_lc0p25_t0p03_a1p50_tau0p75_tc0p35_wmax3p0`

Primary result:

- `5/12` pass
- `gate_g5_pass=10`
- `gate_g6_pass=5`

Pass regimes:

- `1.0 m / 100 mA`
- `1.5 m / 100 mA`
- `1.5 m / 300 mA`
- `1.5 m / 500 mA`
- `1.5 m / 700 mA`

Interpretation:

- The control survived the support-localization challenge.
- This weakens the claim that the remaining gap is best explained by making the
  edge weight more local or more corner-dominant.
- Negative evidence here is scientifically useful: the more aggressive
  localization family is not yet earning promotion.

## Block B Summary

Artifacts:

- experiment: `outputs/support_ablation/final_grid/e2_scientific_screen_v1/block_b_clean/exp_20260409_120752`
- log: `outputs/_launch_logs/run_support_scientific_block_b_clean_20260409_010200.log`

Winner:

- `S27cov_sciv1_tail98_lc0p25_t0p03`

Primary result:

- `4/12` pass
- `gate_g5_pass=5`
- `gate_g6_pass=6`

Pass regimes:

- `1.0 m / 700 mA`
- `1.5 m / 100 mA`
- `1.5 m / 300 mA`
- `1.5 m / 700 mA`

Interpretation:

- `tail98` improved the statistical side of the problem more than the other
  block-`B` candidates.
- However, it still lost to the `A` control on the main protocol endpoint.
- This makes `tail98` a good secondary follow-up if we later decide to pursue a
  `G6`-leaning branch, but not the best mainline candidate today.

## Block C Summary

Artifacts:

- experiment: `outputs/support_ablation/final_grid/e2_scientific_screen_v1/block_c_clean/exp_20260410_113623`
- log: `outputs/_launch_logs/run_support_scientific_block_c_clean_20260410_0001.log`

Winner:

- `S27cov_sciv1_lat10`

Primary result:

- `4/12` pass
- `gate_g5_pass=5`
- `gate_g6_pass=6`

Pass regimes:

- `1.0 m / 700 mA`
- `1.5 m / 100 mA`
- `1.5 m / 300 mA`
- `1.5 m / 700 mA`

Interpretation:

- Increasing latent capacity was the strongest capacity-side move in the block.
- Even so, it did not surpass the `A` control on the primary protocol endpoint.
- This weakens the claim that the main bottleneck is unresolved model capacity.
- Capacity retuning can stay in reserve, but it is not the next promotion target.

## Block D Summary

Artifacts:

- experiment: `outputs/support_ablation/final_grid/e2_scientific_screen_v1/block_d_clean/exp_20260410_113621`
- log: `outputs/_launch_logs/run_support_scientific_block_d_clean_20260410_0001.log`

Winner:

- `S27cov_sciv1_lr0p00015`

Primary result:

- `4/12` pass
- `gate_g5_pass=5`
- `gate_g6_pass=7`

Pass regimes:

- `1.0 m / 700 mA`
- `1.5 m / 100 mA`
- `1.5 m / 300 mA`
- `1.5 m / 700 mA`

Interpretation:

- A lower learning rate was the strongest result outside the main `A` control.
- It still lost to the `A` control on total passes, but it improved `G6`
  relative to the other non-`A` winners.
- This makes `lr0p00015` the strongest secondary follow-up branch from the
  completed screening.

## Cross-Block Conclusion

The current evidence supports this ordering:

1. Main reference: `E2` control
2. Strongest secondary retained branch: `lr0p00015`
3. Secondary retained branch for a more tail-focused follow-up: `tail98`
4. Discarded for immediate promotion:
   - `edgebroad`
   - `edgelocal`
   - `cornerhard`
   - `covsoft`
   - `covhard`
   - `covdense`
5. Capacity-side moves (`lat10`, `h96`, `L192-384-512`) are not the next
   priority for promotion.

## Thesis Framing

For thesis writing, the important scientific message is not only which
candidate wins, but which families fail to justify additional complexity.

The completed screen now supports four defensible statements:

1. The `E2` control remains a hard-to-beat baseline.
2. More aggressive support localization has not yet delivered a better
   full-data protocol outcome.
3. Stronger tail emphasis is interesting, but still looks more like a targeted
   secondary branch than the main reference solution.
4. Capacity expansion was not the main missing ingredient, while a lower
   learning rate remains the strongest optimization-side follow-up.
