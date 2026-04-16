# Full Circle Execution Plan

Date: 2026-04-16

Purpose: immediate execution plan for the `full_circle` line starting from
zero scientific progress.

Important assumption for this document:

- no validation experiment is considered completed yet
- nothing is marked as done
- this file is the operational starting point
- the master reference remains `docs/active/FULL_CIRCLE_VALIDATION_CHECKLIST.md`

## Current Starting Point

Status:

- scientific validation of `full_circle`: not started
- controlled twin comparison against `full_square`: not started
- 12-regime screening: not started
- 27-regime confirmation: not started
- threshold calibration for this line: not started
- uncertainty layer: not started
- temporal validation gate: not started
- external validation on `16QAM`: not started
- downstream learned-signaling validation: not started

## Execution Order

## Phase 0. Setup Freeze

Goal: freeze the initial comparison contract before running any experiment.

Tasks:

- [ ] Confirm dataset root for `full_circle` experiments.
- [ ] Confirm baseline dataset root for `full_square` comparison.
- [ ] Confirm protocol to use for first screening:
  - [ ] `configs/all_regimes_sel4curr.json`
- [ ] Confirm model family for the first controlled run.
- [ ] Confirm training budget for the first controlled run.
- [ ] Confirm output directory convention for the new line.

Minimum decision before moving on:

- one fixed model family
- one fixed training budget
- one fixed 12-regime protocol
- one fixed output path convention

## Phase 1. Dataset Readiness Gate

Goal: verify that the `full_circle` dataset is usable before training.

Tasks:

- [ ] Verify dataset structure is complete across intended regimes.
- [ ] Verify `_report/REPORT.md` exists and is readable.
- [ ] Verify `_report/summary_by_experiment.csv` exists.
- [ ] Verify `_report/summary_by_regime.csv` exists.
- [ ] Verify that preprocessing is the same as the baseline comparison path.
- [ ] Verify the anchor and normalization settings that generated the dataset.
- [ ] Verify there is no geometry-specific preprocessing deviation.

Deliverable:

- a short readiness note saying the dataset is accepted for training

## Phase 2. First Scientific Screening Run

Goal: run the first `full_circle` digital-twin experiment on the reduced
12-regime protocol.

Canonical command template:

```bash
python -m src.protocol.run \
  --dataset_root data/FULL_CIRCLE_2026 \
  --output_base outputs/full_circle \
  --protocol configs/all_regimes_sel4curr.json \
  --train_once_eval_all \
  --no_data_reduction \
  --stat_tests --stat_mode quick
```

Tasks:

- [ ] Launch first 12-regime run.
- [ ] Record exact command used.
- [ ] Record run directory.
- [ ] Summarize the resulting experiment after completion.

Deliverable:

- one baseline `full_circle` screening run with a written summary

## Phase 3. First Reading Of Results

Goal: determine whether the first `full_circle` run is scientifically usable.

Tasks:

- [ ] Read regime-by-regime outcomes.
- [ ] Separate `G1..G5` from `G6` in the interpretation.
- [ ] Inspect edge-focused diagnostics.
- [ ] Inspect whether the center-vs-edge gap looks improved or unchanged.
- [ ] Decide whether the result is:
  - [ ] unusable
  - [ ] promising but incomplete
  - [ ] ready for matched comparison against `full_square`

Deliverable:

- a short reading note with the decision above

## Phase 4. Controlled Matched Comparison

Goal: compare `full_circle` and `full_square` under the same model family and
same training budget.

Tasks:

- [ ] Run matched baseline on `full_square` if no fair comparator exists.
- [ ] Keep model family identical.
- [ ] Keep protocol identical.
- [ ] Keep training budget identical.
- [ ] Compare physical evidence, twin evidence, and regime-specific outcomes.

Deliverable:

- one matched comparison note: `full_square` vs `full_circle`

## Phase 5. Replication And Stability

Goal: test whether any apparent gain is real or just seed variance.

Tasks:

- [ ] Repeat the same comparison with at least 3 seeds if an effect is claimed.
- [ ] Record pass/fail variation by seed.
- [ ] Record whether the conclusion is stable.

Deliverable:

- seed-stability note for the first claimed result

## Phase 6. Full Confirmation

Goal: move from screening to full validation across 27 regimes.

Canonical command template:

```bash
python -m src.protocol.run \
  --dataset_root data/FULL_CIRCLE_2026 \
  --output_base outputs/full_circle \
  --protocol configs/all_regimes_full_dataset.json \
  --train_once_eval_all \
  --no_data_reduction \
  --stat_tests --stat_mode quick
```

Tasks:

- [ ] Run the 27-regime confirmation only after a promising 12-regime result.
- [ ] Read all 27 regimes.
- [ ] Check whether the effect survives hard regimes.
- [ ] Report:
  - [ ] `validation_status_twin = G1..G5`
  - [ ] `stat_screen_pass = G6`
  - [ ] optional `validation_status_full`

Deliverable:

- one full confirmation note for the `full_circle` line

## Phase 7. Credibility Hardening

Goal: convert promising results into a thesis-defensible validation claim.

Tasks:

- [ ] write threshold-calibration note for `G1..G5`
- [ ] add uncertainty / bootstrap reporting
- [ ] add explicit temporal validation gate
- [ ] define the role of `16QAM` external validation
- [ ] clean reporting language for `G6`

Deliverable:

- a defensible validation package, not only a promising run

## Phase 8. Downstream Decision

Goal: decide whether `full_circle` becomes the preferred support geometry.

Tasks:

- [ ] confirm physical bench benefit
- [ ] confirm digital-twin fidelity benefit
- [ ] confirm downstream learned-signaling benefit
- [ ] decide:
  - [ ] promote `full_circle`
  - [ ] keep `full_circle` as complementary only
  - [ ] reject geometry change and refocus on modeling

## Immediate Next Action

Start here:

1. freeze the exact first-run configuration for the 12-regime screen
2. verify dataset readiness for `data/FULL_CIRCLE_2026`
3. launch the first canonical `src.protocol.run` experiment

## What Not To Do Yet

- do not claim `full_circle` is better before the first matched run
- do not jump directly to 27 regimes before the first 12-regime screen
- do not treat `G6` alone as proof of a valid twin
- do not use training loss alone as acceptance evidence