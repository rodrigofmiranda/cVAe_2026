# Full Circle Validation Checklist

Date: 2026-04-16

Purpose: operational checklist for deciding whether `full_circle` is a valid
digital-twin line and whether it should replace or complement `full_square`.

This checklist is derived from:

- `knowledge/syntheses/vlc_shaping_experimental_methodology_2026-04-03.md`
- `knowledge/syntheses/vlc_probabilistic_shaping_strategy_2026-04-03.md`
- `knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md`
- `knowledge/syntheses/gate_threshold_calibration_2026-04-11.md`

## Global Rules

- Keep the comparison fair:
  - same average transmit power
  - same symbol rate
  - same `sps`
  - same acquisition duration
  - same current/distance regimes
  - same RF chain settings
  - same preprocessing pipeline
  - same model family and training budget
- Do not conclude from `loss` or `val_loss` alone.
- Treat `G1..G5` as the main engineering validation ladder.
- Treat `G6` as an auxiliary statistical screen.

## Block A. Bench And Dataset Readiness

Goal: prove the `full_circle` dataset is physically fair and technically usable
before any twin claim.

### A1. Fair acquisition pair

- [ ] Acquire `full_square` and `full_circle` under the same physical settings.
- [ ] Keep the same regime set for both geometries.
- [ ] Prioritize at least these regimes first:
  - [ ] `0.8 m / 100 mA`
  - [ ] `0.8 m / 300 mA`

### A2. Preprocessing parity

- [ ] Apply the same path to both datasets:
  - [ ] startup skip
  - [ ] lag estimation / synchronization
  - [ ] global phase correction
  - [ ] anchor normalization
  - [ ] supervised array generation into `X.npy` and `Y.npy`
- [ ] Do not introduce geometry-specific preprocessing before the first fair comparison.

### A3. Bench diagnostics

- [ ] Measure peak positive excursion.
- [ ] Measure peak negative excursion.
- [ ] Measure asymmetry ratio.
- [ ] Measure crest factor / peak-stress indicator.
- [ ] Measure occupied bandwidth.
- [ ] Measure baseband reconstruction correlation.
- [ ] Measure baseband reconstruction NMSE.
- [ ] Inspect visible harmonic content near `2f` and higher when available.

### A4. Dataset diagnostics

- [ ] Compute input and output radial statistics.
- [ ] Compute residual mean and covariance.
- [ ] Compute residual skewness and kurtosis.
- [ ] Compute Jarque-Bera or equivalent normality-distance metric.
- [ ] Compute center-vs-edge mismatch.
- [ ] Compute amplitude-conditioned diagnostics.
- [ ] Compute radius-conditioned diagnostics.

### A Exit Criteria

- [ ] `full_circle` data is valid, synchronized, normalized, and comparable to `full_square`.
- [ ] Bench evidence suggests `full_circle` is not winning only because of a broken or easier acquisition path.

## Block B. Development Screening On 12 Regimes

Goal: determine whether `full_circle` improves digital-twin behavior under the
reduced scientific protocol.

### B1. Main screening run

Run the canonical protocol entrypoint on the 12-regime protocol.

```bash
python -m src.protocol.run \
  --dataset_root data/FULL_CIRCLE_2026 \
  --output_base outputs/full_circle \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --no_data_reduction \
  --stat_tests --stat_mode quick
```

### B2. Required readout

- [ ] Read regime-by-regime results, not only global averages.
- [ ] Inspect edge-focused diagnostics.
- [ ] Inspect whether the center-vs-edge gap is reduced.
- [ ] Inspect `G1..G5` separately from `G6`.

### B3. Comparison discipline

- [ ] Compare against a matched `full_square` run under the same budget.
- [ ] Compare matched families, not just the global champion.
- [ ] If claiming a real effect, run at least 3 seeds for the same comparison.

### B Exit Criteria

- [ ] `full_circle` shows a meaningful improvement in residual-shape fidelity and/or edge behavior on the 12-regime screen.
- [ ] If the gain appears only in training loss and not in validation metrics, do not promote it.

## Block C. Final Confirmation On 27 Regimes

Goal: verify that the effect survives the full operating space and is not a
screening artifact.

### C1. Full confirmation run

```bash
python -m src.protocol.run \
  --dataset_root data/FULL_CIRCLE_2026 \
  --output_base outputs/full_circle \
  --protocol configs/all_regimes_full_dataset.json \
  --train_once_eval_all \
  --no_data_reduction \
  --stat_tests --stat_mode quick
```

### C2. Required readout

- [ ] Check all 27 regimes.
- [ ] Identify whether gains remain at hard regimes, not only easy ones.
- [ ] Check whether any benefit disappears when moving from 12 to 27 regimes.

### C3. Validation split to report

The preferred reporting split is:

- [ ] `validation_status_twin = G1..G5`
- [ ] `stat_screen_pass = G6`
- [ ] optional `validation_status_full`

### C Exit Criteria

- [ ] `full_circle` survives 27-regime confirmation.
- [ ] Improvement is still visible in engineering fidelity, not only in the statistical screen.

## Block D. Credibility Hardening For Thesis-Level Validation

Goal: close the main gaps explicitly called out in the validation notes.

### D1. Threshold calibration

- [ ] Write a calibration note for `G1..G5`.
- [ ] Update the canonical gate table if calibrated thresholds are accepted.
- [ ] Recompute validation gates on key finalists after calibration.

### D2. Uncertainty layer

- [ ] Add confidence intervals or bootstrap uncertainty for the main validation metrics.
- [ ] Report uncertainty for final claims, not only point estimates.

### D3. Temporal validation

- [ ] Promote an explicit temporal/memory gate, such as `G7` or `delta_acf_l2`.
- [ ] Verify that the twin captures temporal structure, not only pointwise residual shape.

### D4. External validation

Use the trained shared model as an external generalization check on `16QAM`.

```bash
python -m src.protocol.run \
  --dataset_root data/16qam \
  --output_base outputs/full_circle_external \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --reuse_model_run_dir outputs/full_circle/exp_YYYYMMDD_HHMMSS/train \
  --stat_tests --stat_mode quick \
  --no_baseline
```

- [ ] Formalize whether external validation is advisory, tie-breaking, or mandatory.

### D5. Statistical interpretation cleanup

- [ ] Report `G6` as "no detected residual-distribution mismatch under MMD/Energy + BH".
- [ ] Do not describe `G6` as proof of equivalence or indistinguishability.

### D Exit Criteria

- [ ] Thresholds are calibrated.
- [ ] Main metrics have uncertainty bounds.
- [ ] Temporal validation exists.
- [ ] External validation role is explicit.
- [ ] Final claims do not overstate `G6`.

## Block E. Downstream Decision Experiment

Goal: decide whether `full_circle` should become the preferred support for the
future learned-communication line.

### E1. Support decision

- [ ] Promote `full_circle` only if it improves:
  - [ ] bench diagnostics
  - [ ] digital-twin residual-shape fidelity
  - [ ] downstream learned-signaling behavior under fair constraints

### E2. Learned-signaling follow-up

- [ ] Train the next learned constellation / modulation system on the stronger twin.
- [ ] Re-inject the learned signaling into GNU Radio.
- [ ] Measure whether the expected gain survives on the physical bench.

### E Decision Rule

- [ ] If `full_circle` wins in bench + twin + downstream layers, promote it.
- [ ] If it only improves model fit while excluding relevant operating regions, keep it as a complementary dataset.
- [ ] If no consistent gain appears, refocus on architecture, conditioning, loss design, and temporal modeling.

## Minimal Claim Ladder

Use this claim ladder when reporting results:

- Level 1: dataset and bench comparison completed
- Level 2: 12-regime screening supports `full_circle`
- Level 3: 27-regime confirmation supports `full_circle`
- Level 4: calibrated and uncertainty-aware twin validation supports `full_circle`
- Level 5: downstream learned signaling also benefits on the physical bench

Do not claim a thesis-grade validated twin before Level 4.