# Legacy 2025 Batch-Size Results

Historical note:

- this document is archived on purpose
- some links point to local outputs from a deleted historical worktree
- keep it only as scientific history, not as the active runbook

Current status of the `legacy_2025_zero_y` batch-size sweep under the strict
2026 pivot-regime protocol.

Protocol basis:

- config: [one_regime_1p0m_300mA_sel4curr.json](/workspace/2026/feat_seq_bigru_residual_cvae/configs/one_regime_1p0m_300mA_sel4curr.json)
- preset family: `legacy2025_batch_sweep`
- fixed model config: `layer_sizes=[32,64,128,256]`, `latent_dim=16`,
  `beta=0.1`, `lr=1e-4`, `free_bits=0.0`, `kl_anneal_epochs=50`

## Reference

- batch size: `4096`
- run: [exp_20260318_193036](/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_193036)
- best val loss: `-1.8306`
- train time: `591.9 s`
- pivot metrics:
  - `ΔEVM = -1.8946 pp`
  - `ΔSNR = +0.5557 dB`
  - `Δmean_l2 = 0.02266`
  - `PSD_L2 = 0.25219`
  - `MMD² = 0.004884`

## Sweep outcome

### `bs8192` — PASS

- run: [exp_20260318_195010](/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_195010)
- best val loss: `-1.6959`
- train time: `291.3 s`
- pivot metrics:
  - `ΔEVM = -1.4411 pp`
  - `ΔSNR = +0.4194 dB`
  - `Δmean_l2 = 0.01993`
  - `PSD_L2 = 0.27514`
  - `MMD² = 0.005237`

Interpretation:

- roughly half the training time of `bs4096`
- moderate degradation relative to the reference, but still within the
  intended acceptance band for the batch-size protocol
- no OOM, no NaN, no obvious instability

Overlay paths:

- [overlay_constellation.png](/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_195010/studies/within_regime/regimes/dist_1m__curr_300mA/plots/overlay_constellation.png)
- [overlay_residual_delta.png](/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_195010/studies/within_regime/regimes/dist_1m__curr_300mA/plots/overlay_residual_delta.png)

### `bs16384` — FAIL

- run: [exp_20260318_195709](/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_195709)
- best val loss: `-1.3882`
- train time: `145.6 s`
- pivot metrics:
  - `ΔEVM = +0.9666 pp`
  - `ΔSNR = -0.2704 dB`
  - `Δmean_l2 = 0.02167`
  - `PSD_L2 = 0.51943`
  - `MMD² = 0.016263`

Interpretation:

- much faster, but quality dropped sharply
- `ΔEVM` crossed to the wrong side of zero
- `PSD_L2` roughly doubled versus the reference
- `MMD²` worsened by more than `3×`

Overlay paths:

- [overlay_constellation.png](/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_195709/studies/within_regime/regimes/dist_1m__curr_300mA/plots/overlay_constellation.png)
- [overlay_residual_delta.png](/workspace/2026/feat_delta_residual_adv/outputs/exp_20260318_195709/studies/within_regime/regimes/dist_1m__curr_300mA/plots/overlay_residual_delta.png)

## Current conclusion

Stop condition reached at `bs16384`.

Current operational ceiling for this legacy variant on the tested protocol:

- `batch_size = 8192`

Reason:

- `8192` preserves acceptable quality while cutting training time by about half
- `16384` is clearly too aggressive for this setup and causes a meaningful
  fidelity collapse

No further escalation to `32768` or `65536` is recommended unless the goal is
explicitly to characterize failure modes rather than choose an operational
batch size.
