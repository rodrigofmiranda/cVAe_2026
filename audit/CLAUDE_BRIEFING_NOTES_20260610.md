# Claude Briefing Notes - V3 FULLSQUARE G6, Organization, And Run Anchors

Date: 2026-06-10

Purpose: notes to explain the current V3 result and the G6/status semantics to
Claude later.

## Current V3 Runs

Primary V3 global model re-eval with stat tests:

- run: `/home/rodrigo/cVAe_2026_full_square_v3det/outputs/v3_fullsquare_s39b_eval_stat_20260610/exp_20260610_154921`
- timestamp: `2026-06-10T15:49:21` to `2026-06-10T15:58:18`
- commit: `40162c300d375022a0276df494457dff2bf6e2ad`
- reused model from: `outputs/v3_fullsquare_s39b_det_20260610/exp_20260610_032448/train`
- stat config: `--stat_tests`, `stat_mode=quick`, `stat_max_n=5000`, `n_perm=200`, seed 42
- current code status: `1 pass / 0 partial / 11 fail`
- gates: `G1=9`, `G2=12`, `G3=9`, `G4=12`, `G5=9`, `G6=1`

If read as main twin gates only (`G1..G5`), the same V3 global result is:

- `9/12 pass`
- `3/12 fail`
- structural `G1..G5` failures are all near-field:
  - `dist_0p75m__curr_300mA`: `G1,G3,G5`
  - `dist_0p75m__curr_500mA`: `G1,G3,G5`
  - `dist_0p75m__curr_700mA`: `G1,G3,G5`

The remaining `8` current-code failures are G6-only failures. That means the
model is often passing the engineering twin ladder while failing the formal
statistical residual screen.

## G6 Measurement Check

G6 is being measured in the V3 re-eval. It is not empty.

Current V3 code path:

- `src/protocol/run.py` computes residuals `Y - X`
- it uses MC predictions, capped at `stat_max_n=5000`
- it runs MMD RBF and Energy distance permutation tests
- `src/evaluation/validation_summary.py` applies BH/FDR q-values
- `gate_g6 = stat_mmd_qval > 0.05 and stat_energy_qval > 0.05`

Manual audit result:

- G6 pass count: `1/12`
- MMD q pass count: `2/12`
- Energy q pass count: `1/12`
- per-family BH instead of joint MMD+Energy BH does not change pass/fail for this run

So the V3 G6 collapse is not a simple missing-column bug.

## Important Semantic Issue

Rodrigo remembered correctly: the older/canonical FULLSQUARE methodology split
G6 out of the main validation status.

Evidence in `/home/rodrigo/cVAe_2026_full_square`:

- `src/evaluation/validation_summary.py`
  - `_validation_status(...)` uses `G1..G5`
  - `stat_screen_pass = gate_g6`
  - `validation_status_twin = G1..G5`
  - `validation_status_full = G1..G6`
  - legacy `validation_status = validation_status_twin`
- `docs/reference/PROTOCOL.md`
  - says the twin heatmap/status uses `G1..G5`
  - says `G6` is auxiliary statistical-screen summary
- `knowledge/syntheses/gate_validation_audit_2026-04-11.md`
  - says G6 is legitimate but should not be phrased as proof of equivalence
- `knowledge/syntheses/gate_threshold_calibration_2026-04-11.md`
  - recommends `validation_status_twin`, `stat_screen_pass`, and `validation_status_full`

The current V3 branch lost that semantic split. It currently lets G6 veto
`validation_status`, which turns a `9/12` twin-status reading into `1/12`.

## Recommended Framing For Claude

Do not remove G6.

Restore the earlier semantics:

- keep `gate_g6`
- keep MMD/Energy q-values
- keep `validation_status_full` for the strict/conservative read
- make the main `validation_status` or new `validation_status_twin` use `G1..G5`
- expose/report `stat_screen_pass` separately

Suggested language:

- Bad: "G6 proves distributional indistinguishability."
- Good: "G6 is a conservative residual-distribution screen; under the configured
  MMD/Energy + BH test budget, it did or did not detect mismatch."

## Isolation 0.75m Result

Exp 2 isolation run completed:

- run: `/home/rodrigo/cVAe_2026_full_square_v3det/outputs/v3_iso_0p75m_s39b_det_20260610/exp_20260610_185747`
- timestamp: `2026-06-10T18:57:47` to `2026-06-10T20:23:53`
- commit: `40162c300d375022a0276df494457dff2bf6e2ad`
- seed: 33
- best epoch: 252
- best val recon: `-4.75850248336792`
- current full status: `0/4 pass`
- gates: `G1=4`, `G2=4`, `G3=3`, `G4=4`, `G5=1`, `G6=0`

If read as `G1..G5` only:

- `1/4 pass`
- `3/4 fail`
- `0.75m/300mA`: `G3,G5`
- `0.75m/500mA`: `G5`
- `0.75m/700mA`: `G5`

Interpretation: the near-field V3 issue is not only global inter-regime conflict.
Even the 0.75m specialist does not recover the higher-current near-field shape
gates. G6 is harsh, but G5/G3 still show real structure issues.

## Exp 1 Status

The G6-aligned loss experiment is running:

- container: `cvae_v3_g6a`
- output: `/home/rodrigo/cVAe_2026_full_square_v3det/outputs/v3g6_aligned_det_20260610`
- run: `exp_20260610_202554`
- commit: `53e4195e75b841f47526ce88a7a57adb8da4680f`
- started: `2026-06-10T20:25:54`
- status at latest Codex note: still running, no leaderboard yet
- manifest status: `run_status=running`
- container status: `cvae_v3_g6a` up
- run log had reached about epoch `72/500`
- training loss includes the new differentiable terms:
  - `mmd_loss`
  - `energy_loss`

Important caveat for Claude:

- Do not call Exp 1 a result until `tables/protocol_leaderboard.csv` exists.
- Its role is to test whether directly aligning training pressure with G6
  improves the G6-only failures without regressing the G1-G5 fingerprint.

## Repository Organization Update

Canonical local repos after cleanup:

- `/home/rodrigo/cVAe_2026_full_square`
  - FULLSQUARE methodology/data/artifact workspace.
  - Use for G1-G5/G6 split, older V2 methodology, MDN-return consolidation, and
    architecture buckets.
- `/home/rodrigo/cVAe_2026_full_circle`
  - FULLCIRCLE support-geometry/16QAM architecture comparison workspace.
  - The old `cVAe_2026_shape_fullcircle` checkout was consolidated here and
    removed.
- `/home/rodrigo/cVAe_2026_full_square_v3det`
  - current V3 deterministic/full-square experiment workspace.
- `/home/rodrigo/eduardo_cVAe_2026_audit`
  - canonical Eduardo/V2 audit checkout.
  - as of 2026-06-11, it also contains the former
    `cvae_repro_141943` audit and reproduction outputs.
  - the old top-level `/home/rodrigo/cvae_repro_141943` symlink was removed, so
    this is the only folder for that line.
- `/home/rodrigo/TESE`
  - thesis-writing workspace. The old `/home/rodrigo/Tese` name was normalized
    to uppercase `TESE`.
- `/home/rodrigo/1-Data`
  - official shared dataset workspace.
  - the temporary `/home/rodrigo/data` root was removed on 2026-06-11 after
    comparison against this official tree.
  - official processed datasets live under `Dataset/`: `FULLSQUARE_2026`,
    `FULL_CIRCLE_2026`, `16QAM_DATASET_2026`, and `V3`.
  - repo-local `data` paths in full-square, full-circle, V3det, and Eduardo
    audit point to `/home/rodrigo/1-Data/compat/cvae_data`.
  - that compatibility layer preserves old names such as
    `dataset_fullsquare_organized` and `16qam`, while resolving to the official
    dataset names.
- `/home/rodrigo/knowledge`
  - canonical shared literature/notes/syntheses workspace.
  - consolidated from duplicated `knowledge/` trees in `cVAe_2026_full_square`
    and `cVAe_2026_full_circle`.
  - full-square, full-circle, and Eduardo audit now keep `knowledge` as a
    symlink to this root base, not as independent copies.

Cleanups already done:

- `cVAe_2026_full_square`
  - dirty worktree was stashed before reorganization:
    `codex clean cVAe_2026_full_square worktree 2026-06-10`
  - current expected status includes only the tracked deletion:
    `D docs/revisao_tecnica_cvae_vlc.docx`
  - that file was moved to `/home/rodrigo/TESE/09_anexos/revisao_tecnica_cvae_vlc.docx`
  - expected after knowledge dedupe:
    - tracked `knowledge/...` files appear deleted
    - `knowledge` appears as an untracked symlink to `/home/rodrigo/knowledge`
  - expected after data dedupe:
    - tracked `data/...` files appear deleted
    - `data` appears as an untracked symlink to
      `/home/rodrigo/1-Data/compat/cvae_data`
- `cVAe_2026_full_circle`
  - dirty worktree was stashed before migration:
    `codex clean cVAe_2026_full_circle before shape_fullcircle migration 2026-06-10`
  - current expected status includes only:
    `D docs/revisao_tecnica_cvae_vlc.docx`
  - that same docx was identical to the full-square copy and was deduplicated
    into `TESE`
  - expected after knowledge dedupe:
    - tracked `knowledge/...` files appear deleted
    - `knowledge` appears as an untracked symlink to `/home/rodrigo/knowledge`
  - expected after data dedupe:
    - tracked `data/...` files appear deleted
    - `data` appears as an untracked symlink to
      `/home/rodrigo/1-Data/compat/cvae_data`
- `cVAe_2026_shape_fullcircle`
  - outputs and analysis were moved into `/home/rodrigo/cVAe_2026_full_circle`
  - old checkout was removed
  - old docs may still mention this path; translate mentally to
    `/home/rodrigo/cVAe_2026_full_circle`
- `cVAe_2026_mdn_return`
  - was not a third geometry; it was a detached FULLSQUARE MDN-return line
  - useful outputs were moved into `/home/rodrigo/cVAe_2026_full_square`
  - old checkout was removed
  - branch/history preserved in full-square as:
    `archive/mdn-return-20260416` at `dc1cb53`
  - branch plus dirty-worktree stash preserved as:
    `/home/rodrigo/cVAe_2026_full_square/outputs/architectures/mdn_return/_repo_archive/mdn_return_research_branch_and_stash_20260610.bundle`
- `cvae_repro_141943`
  - was consolidated into `/home/rodrigo/eduardo_cVAe_2026_audit` on
    2026-06-11.
  - the old top-level path was removed after consolidation; only
    `/home/rodrigo/eduardo_cVAe_2026_audit` remains.
  - former `audit/` lives at
    `/home/rodrigo/eduardo_cVAe_2026_audit/audit/repro_141943/repro_workspace_audit/`.
  - former repro outputs were moved into
    `/home/rodrigo/eduardo_cVAe_2026_audit/outputs/`.
  - source/config/docs snapshot from the non-Git repro workspace is preserved at
    `/home/rodrigo/eduardo_cVAe_2026_audit/audit/repro_141943/_repo_archive/cvae_repro_141943_source_config_snapshot_20260611.tar.gz`.
  - expected Eduardo audit `git status` is now noisy: it includes pre-existing
    modified/unknown code/docs plus tracked deletions under `data/`, `knowledge/`,
    and `docs/revisao_tecnica_cvae_vlc.docx`; those deletions correspond to the
    new root data/knowledge/TESE organization and should not be blindly reverted.

Do not look for active work in the removed folders:

- `/home/rodrigo/cVAe_2026_shape_fullcircle`
- `/home/rodrigo/cVAe_2026_mdn_return`
- `/home/rodrigo/Tese`

Do not look for `/home/rodrigo/cvae_repro_141943`; it was removed after its
content was consolidated into the Eduardo audit checkout.

Do not treat the old `knowledge/` trees in `full_square` and `full_circle` as
separate sources anymore. Use `/home/rodrigo/knowledge` for shared literature,
notes, and syntheses.

Do not treat old repo-local `data/` trees as separate dataset copies anymore.
Use `/home/rodrigo/1-Data`; the repo-local `data` paths are symlinks to
`/home/rodrigo/1-Data/compat/cvae_data` for compatibility.

For the Eduardo/141943 line, `data/16qam` now resolves to the official
`/home/rodrigo/1-Data/Dataset/16QAM_DATASET_2026` through that compatibility
layer.

## Architecture Buckets After Reorganization

In `/home/rodrigo/cVAe_2026_full_square/outputs/architectures`:

- `mdn_return/S35C_fast_e64_base/`
  - `fullsquare/`: original S35 FULLSQUARE training/eval artifacts
  - `16qam/`: downstream 16QAM evaluations
- `seq_bigru_residual/.../16qam/`
- `probabilistic_shaping/S27_historical/16qam/`
- `legacy_2025_zero_y/...`
- `comparative/best_compare_large/...`
- `_crossline/16qam/...`

In `/home/rodrigo/cVAe_2026_full_circle/outputs/architectures`:

- `clean_baseline/S27cov_fc_clean_lc0p25_t0p03_lat10/16qam/`
- `disk_geom3/S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192/16qam/`
- `soft_rinf_local/S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0/16qam/`
- `seq_bigru_residual/S2seq_W7_h64_lat4_b0p003_lmmd0p5_fb0p10_lr0p0003_L128-256-512/`
- `_crossline/16qam/...`

## Run And Architecture Anchors

Always consult `/home/rodrigo/ai_workflow/EXPERIMENT_RESULTS.csv` and
`EXPERIMENT_HISTORY.md` before proposing a new run. The local skill
`vlc-cvae-research` was updated with the current project map.

### FULLSQUARE / Eduardo / V2 Anchors

Canonical Eduardo S39B anchor:

- run: `/home/rodrigo/eduardo_cVAe_2026_audit/outputs/exp_20260427_141943`
- tag: `S39B_edgegap_lowlr_all08_w18_p120`
- architecture: `seq_bigru_residual`, MDN
- result: `10/12`
- gates: `G1=12`, `G2=12`, `G3=12`, `G4=12`, `G5=10`, `G6=12`
- failures:
  - `0.8m/300mA`: `G5`
  - `0.8m/500mA`: `G5`
- reading: best historical V2/Eduardo anchor. It shows the ML family can be
  strong on that data line, but it is not a direct numeric baseline for V3.

Local full-square S39B reproduction:

- run: `/home/rodrigo/cVAe_2026_full_square/outputs/parent_s39b_repro_141943_20260527_train/exp_20260527_221226`
- result: `10/12`
- gates: `G1=10`, `G2=10`, `G3=10`, `G4=12`, `G5=10`, `G6=4`
- failures:
  - `0.8m/100mA`: `G1,G2,G3,G5,G6`
  - `0.8m/300mA`: `G1,G2,G3,G5,G6`
- reading: local reproduction reaches `10/12` but with much weaker G6 than
  Eduardo's original copy; again, report fingerprint, not only pass count.

S38D edge-gap recovery:

- run: `/home/rodrigo/cVAe_2026_full_square/outputs/parent_s38d_repro_20260526_train/exp_20260526_220848`
- result: `9/12`
- gates: `G1=9`, `G2=9`, `G3=9`, `G4=12`, `G5=10`, `G6=8`
- failures:
  - `0.8m/100mA`: `G1,G2,G3,G6`
  - `0.8m/300mA`: `G1,G2,G3,G5,G6`
  - `0.8m/500mA`: `G1,G2,G3,G5,G6`

MDN-return S35C after consolidation:

- run: `/home/rodrigo/cVAe_2026_full_square/outputs/architectures/mdn_return/S35C_fast_e64_base/fullsquare/20260416_204133_seq_cond_embed_fast_stage1_100k/exp_20260416_204135`
- tag: `S35C_fast_e64_base`
- result under conservative current status: `7/12`
- gates: `G1=12`, `G2=12`, `G3=12`, `G4=12`, `G5=12`, `G6=7`
- if read as main twin status (`G1..G5`): `12/12`
- failures are G6-only:
  - `0.8m/100mA`
  - `0.8m/300mA`
  - `0.8m/500mA`
  - `1.0m/500mA`
  - `1.5m/300mA`
- reading: important example for Claude. It demonstrates why the G1-G5 versus
  G6 semantic split matters. It is not a reason to delete G6.

### FULLCIRCLE Anchors

Clean Full Circle line:

- clean line is weak overall: historical reads include `1/12`, `2/12`, and
  `5/12`.
- example run:
  `/home/rodrigo/cVAe_2026_full_circle/outputs/full_circle/20260417_115140_clean_bs8192_lat10_100k_split_a/exp_20260417_115142`
- result: `2/12`
- gates: `G1=7`, `G2=2`, `G3=7`, `G4=10`, `G5=8`, `G6=8`
- reading: this is the honest clean baseline. Do not promote Full Circle just
  because geometry-biased runs are easier.

Disk/geometry-biased Full Circle:

- run:
  `/home/rodrigo/cVAe_2026_full_circle/outputs/full_circle/g2_shortlist_100k/exp_20260416_120619`
- tag: `S27cov_lc0p25_tail95_t0p03_disk_geom3`
- result: `7/12`
- gates: `G1=9`, `G2=8`, `G3=9`, `G4=11`, `G5=8`, `G6=10`
- reading: useful but geometry-biased.

Disk/geometry-biased bs8192/lat10:

- run:
  `/home/rodrigo/cVAe_2026_full_circle/outputs/full_circle/disk_bs8192_lat10_100k_split_a/exp_20260416_165643`
- tag: `S27cov_lc0p25_tail95_t0p03_disk_geom3_bs8192`
- result: `8/12`
- gates: `G1=8`, `G2=8`, `G3=8`, `G4=11`, `G5=10`, `G6=9`
- failures all at `0.8m`
- reading: operationally strong for this line, but still geometry-biased.

Soft radial compromise:

- run:
  `/home/rodrigo/cVAe_2026_full_circle/outputs/full_circle/20260420_233254_soft_radial_block_a_100k/exp_20260420_233256`
- tag: `S27cov_fc_soft_rinf_local_lat10_a1p50_tau0p80_wmax3p0`
- result: `6/12`
- gates: `G1=7`, `G2=6`, `G3=8`, `G4=11`, `G5=8`, `G6=8`
- reading: current compromise line; not as high as the hard geometry-biased
  runs, but more defensible than hard support tricks.

Full Circle seq baseline / architecture comparison:

- run:
  `/home/rodrigo/cVAe_2026_full_circle/outputs/architectures/seq_bigru_residual/S2seq_W7_h64_lat4_b0p003_lmmd0p5_fb0p10_lr0p0003_L128-256-512/full_circle_dataset/mirror_fullsquare_16qam_best_20260425_1835/exp_20260425_183444`
- result: `8/12`
- gates: `G1=9`, `G2=9`, `G3=9`, `G4=12`, `G5=8`, `G6=9`
- failures:
  - `0.8m/100mA`: `G1,G2,G3,G5,G6`
  - `0.8m/300mA`: `G1,G2,G3,G5,G6`
  - `0.8m/500mA`: `G1,G2,G3,G5,G6`
  - `0.8m/700mA`: `G5`
- reading: strong architecture-comparison artifact, but still dominated by
  `0.8m` failure structure.

### V3 Anchors

V3 global S39B deterministic training and G6 re-eval:

- training run:
  `/home/rodrigo/cVAe_2026_full_square_v3det/outputs/v3_fullsquare_s39b_det_20260610/exp_20260610_032448`
- stat re-eval:
  `/home/rodrigo/cVAe_2026_full_square_v3det/outputs/v3_fullsquare_s39b_eval_stat_20260610/exp_20260610_154921`
- conservative current status: `1/12`
- main twin read (`G1..G5`): `9/12`
- structural `G1..G5` failures are `0.75m/300`, `0.75m/500`, `0.75m/700`
- G6-only failures dominate the conservative full-status collapse.

V3 0.75m specialist:

- run:
  `/home/rodrigo/cVAe_2026_full_square_v3det/outputs/v3_iso_0p75m_s39b_det_20260610/exp_20260610_185747`
- conservative current status: `0/4`
- main twin read (`G1..G5`): `1/4`
- reading: near-field problem is not only inter-regime conflict; G5/G3 remain
  real within the specialist.

## Full-Square Knowledge To Use

Use `/home/rodrigo/cVAe_2026_full_square` as the methodology/code reference and
`/home/rodrigo/knowledge` as the shared literature/synthesis reference, not as a
direct numeric baseline for V3.

Read order:

1. `docs/agents/CONTEXT_CAPSULE.md`
2. `docs/active/WORKING_STATE.md`
3. `docs/reference/PROTOCOL.md`
4. `/home/rodrigo/knowledge/syntheses/gate_validation_audit_2026-04-11.md`
5. `/home/rodrigo/knowledge/syntheses/gate_threshold_calibration_2026-04-11.md`
6. `/home/rodrigo/knowledge/syntheses/digital_twin_validation_foundation_table_2026-04-11.md`

Important V2 lesson:

- V2 FULLSQUARE had strong results with the same broad ML family, but the
  persistent bottleneck was residual shape (`G5`) around short distance/low
  current.
- V3 changed the dataset/hardware geometry. More LEDs or more data does not
  automatically preserve the same residual distribution or make G6 easier.
- Do not compare `val_recon` directly across V2/V3 lines.

## Next Practical Step

When Claude is ready to change code, the least invasive patch is to port the
FULLSQUARE split from:

- `/home/rodrigo/cVAe_2026_full_square/src/evaluation/validation_summary.py`

into:

- `/home/rodrigo/cVAe_2026_full_square_v3det/src/evaluation/validation_summary.py`

Then regenerate/backfill the V3 summary/leaderboard without retraining, so we
can report both:

- main twin status: `G1..G5`
- stat screen: `G6`
- conservative full status: `G1..G6`

## Suggested Message To Claude

Short version to hand him:

1. Organization is now canonicalized:
   - two geometry repos: `cVAe_2026_full_square`, `cVAe_2026_full_circle`
   - current V3 repo: `cVAe_2026_full_square_v3det`
   - Eduardo/V2 audit: `/home/rodrigo/eduardo_cVAe_2026_audit`
   - `/home/rodrigo/cvae_repro_141943` was consolidated and removed; use the
     Eduardo audit path.
   - thesis docs: `/home/rodrigo/TESE`
   - shared data: `/home/rodrigo/1-Data`
   - shared knowledge: `/home/rodrigo/knowledge`
   - old detached folders `shape_fullcircle`, `mdn_return`, and the standalone
     `cvae_repro_141943` workspace were consolidated.
2. Do not interpret V3 as `1/12` without the split:
   - current conservative V3 global status is `1/12`
   - main twin status is `9/12`
   - G6 is real, but should be reported as a separate statistical screen.
3. The S35C MDN-return artifact is a key semantics example:
   - conservative `7/12`
   - `G1..G5 = 12/12`
   - all failures are G6-only.
4. Historical strong results remain relevant as methodology/architecture
   anchors, not direct V3 numeric baselines:
   - Eduardo S39B `10/12`, failures only `0.8m/300` and `0.8m/500` on `G5`
   - local full-square S39B `10/12`, but weaker G6
   - S38D `9/12`
   - Full Circle clean weak, geometry-biased lines up to `8/12`, soft radial
     around `6/12`
5. Exp 1 G6-aligned is still running; wait for leaderboard before drawing any
   conclusion.
