# V3 FULLSQUARE S39B Audit - 2026-06-10

## Verdict

The V3 FULLSQUARE transfer run is **not functionally reproduced/passing** under
the current protocol. The strongest artifact is the statistical re-evaluation:

- Run: `outputs/v3_fullsquare_s39b_eval_stat_20260610/exp_20260610_154921`
- Result: `1 pass / 0 partial / 11 fail`
- Gate totals: `G1=9/12`, `G2=12/12`, `G3=9/12`, `G4=12/12`, `G5=9/12`, `G6=1/12`
- Only full pass: `dist_1m__curr_100mA`

The earlier train+eval artifact did not run statistical tests, so its
`0 pass / 9 partial / 3 fail` result is incomplete for G6.

## Evidence Paths

- Training run:
  `outputs/v3_fullsquare_s39b_det_20260610/exp_20260610_032448`
- Statistical re-evaluation:
  `outputs/v3_fullsquare_s39b_eval_stat_20260610/exp_20260610_154921`
- Main tables:
  - `tables/protocol_leaderboard.csv`
  - `tables/summary_by_regime.csv`
  - `tables/stat_fidelity_by_regime.csv`
- Main plots:
  - `plots/best_model/heatmap_gate_metrics_by_regime.png`
  - `plots/best_model/residual_signature_overview.png`

## Provenance

- Git commit: `40162c300d375022a0276df494457dff2bf6e2ad`
- Git branch: `exp/v3-fullsquare-s39b-det`
- Python/TensorFlow/NumPy: `3.12.3 / 2.17.0 / 1.26.4`
- Protocol: `configs/protocol_v3_fullsquare.json`
- Dataset root recorded by artifacts:
  `/data/Dataset/V3/FULLSQUARE_2026_V3_ORGANIZED`
- Protocol geometry: 12 regimes, `0.75/1.0/1.5 m x 100/300/500/700 mA`
- Split: `per_experiment`
- Seed: `33`
- Inference: `deterministic_inference=false`, `rank_mode=mc`, `mc_samples=8`
- Decoder logvar clamp: `[-6.61, -0.54]`, empirical V3 clamp
- Best model hash:
  `68d04dc747306e8ea688510cf664bff8644c868a7f63fb973cf37a2c65d96465`

Host caveat: `/data/Dataset/V3/FULLSQUARE_2026_V3_ORGANIZED` is not mounted on
the host at audit time, so this audit cannot verify a byte-level dataset hash.

Git caveat: the working tree contains untracked files used by this line:

- `configs/protocol_v3_fullsquare.json`
- `scripts/ntfy_watch_v3det.sh`

## Training Basin

The model trained successfully, but this is not the V2 champion basin on the V3
scale.

- Preset: `S39B_edgegap_lowlr_all08_w18_p120`
- Architecture: `seq_bigru_residual`, MDN, 3 components
- Epochs run: `268`
- Best epoch: `208`
- Best val recon loss: `-4.748651504516602`
- Last val recon loss: `-4.726807594299316`
- Active dims: `8/8`
- KL mean total: `39.53750991821289`
- Final LR: `2e-05`, with 2 LR drops

Operational finding: despite the `det` label, the run log says
`determinism: setseed=0 opdet=0 seed=33`. This is seeded, but not the opt-in
deterministic training mode.

## Functional Fingerprint

Statistical re-evaluation (`stat_mode=quick`, `n=5000`, `n_perm=200`,
`q_alpha=0.05`) produced:

| Regime | Status | Failed gates | Key metric |
|---|---:|---|---|
| `dist_0p75m__curr_100mA` | fail | G6 | Energy q=0.0398 |
| `dist_0p75m__curr_300mA` | fail | G1,G3,G5,G6 | rel EVM=0.1078, mean/cov=0.403/0.431 |
| `dist_0p75m__curr_500mA` | fail | G1,G3,G5,G6 | rel EVM=0.2270, JB rel=3.37 |
| `dist_0p75m__curr_700mA` | fail | G1,G3,G5,G6 | mean/cov=0.442/0.421, JB rel=21.30 |
| `dist_1m__curr_100mA` | pass | none | MMD q=0.5522, Energy q=0.4101 |
| `dist_1m__curr_300mA` | fail | G6 | MMD/Energy q=0.0085 |
| `dist_1m__curr_500mA` | fail | G6 | MMD q=0.0265, Energy q=0.0140 |
| `dist_1m__curr_700mA` | fail | G6 | MMD/Energy q=0.0140 |
| `dist_1p5m__curr_100mA` | fail | G6 | MMD/Energy q=0.0085 |
| `dist_1p5m__curr_300mA` | fail | G6 | MMD q=0.0377, Energy q=0.0398 |
| `dist_1p5m__curr_500mA` | fail | G6 | MMD/Energy q=0.0085 |
| `dist_1p5m__curr_700mA` | fail | G6 | MMD/Energy q=0.0085 |

Interpretation:

- The hard structural failure cluster is `0.75m / 300-700mA`.
- `G2` and `G4` are healthy across all 12 regimes.
- `G6` is the dominant global rejection: only `1m/100mA` survives both MMD and
  Energy q-values.
- The `0.75m` failures are not only statistical; they include direct EVM,
  mean/cov residual fidelity, and JB/tail mismatch.

## Protocol/Config Issues To Fix Before Next Run

1. The S39B resampling weights still target literal `dist_0p8m__curr_*`.
   In this V3 protocol the near-field analog is `dist_0p75m__curr_*`, and the
   training log shows `regime resampling: changed=False`.

2. The run is named deterministic, but deterministic training was not enabled.
   If exact reproducibility is required, rerun with the opt-in deterministic
   environment used by the reproducibility patch.

3. The protocol config is untracked. Commit or archive it with the run before
   treating the result as a fully reproducible V3 baseline.

## Recommended Next Experiment

Do not retry a blind S39B seed. First run a V3-retargeted S39B control:

- map resampling weights from `0p8m` to `0p75m`;
- keep the same V3 empirical clamp `[-6.61, -0.54]`;
- keep `mc_samples=8` and evaluate with stat tests;
- decide whether deterministic mode is part of the claim before launch.

If that still fails as above, the next evidence-based change is a V3 near-field
specialization around `0.75m/300-700mA`, not a generic MDN component increase
or kurtosis-only penalty; those routes were already negative on the V2 history.
