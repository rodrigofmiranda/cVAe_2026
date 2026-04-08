# Support Hyperparameters Guide (Shape Line)

## Context

This note summarizes the support-aware hyperparameters we have been using in the
`shape` branch and what they appear to do in practice.

The focus is the family that actually survived the recent screening:

- `E2` = edge-weight only
- `E3c` = geom3 + edge + softer coverage
- `E2 full overnight` = local retune around the `E2 full` finalist

The intent is practical: explain what each knob means, what changing it should
do, and what our own runs suggest so far.

## Where These Knobs Act

- `support_feature_mode` changes the model inputs/conditioning:
  - in the sequence model it injects support geometry features into the prior,
    encoder and decoder.
- `support_weight_mode`, `support_weight_alpha`, `support_weight_tau`,
  `support_weight_tau_corner`, `support_weight_max` change the per-sample
  training weights.
- `lambda_coverage`, `tail_levels`, `coverage_temperature` change the auxiliary
  coverage/tail calibration loss.
- `support_filter_mode` changes which samples are kept in train/eval.

Code anchors:

- support geometry / weights / filter:
  - `src/data/support_geometry.py`
- support-aware weighting + coverage loss:
  - `src/models/losses.py`
- geom3 injection into the seq model:
  - `src/models/cvae_sequence.py`

## Hyperparameters

| Hyperparameter | What it means in code | If you increase it | Main risk | Values already explored | Trend we observed | Confidence | Change now? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `support_feature_mode` | Adds explicit support descriptors. `geom3` = `[r_l2_norm, r_inf_norm, cornerness_norm]` from the current center symbol. | Gives the model more geometric context about center/edge/corner location. | Can help geometry but also make optimization more brittle or over-specialized. | `none`, `geom3` | `geom3` alone (`E1`) did not solve the problem. It became useful only when paired with edge pressure, and even then it did **not** hold up best in full data. | medium | no |
| `support_weight_mode` | Chooses the sample-weighting rule. `edge_rinf_corner` upweights samples near the square boundary and even more in corners. | Makes the model spend more gradient budget on edge/corner samples. | Can hurt global fidelity if the border gets overweighted. | `none`, `edge_rinf_corner` | This was the first clearly successful idea. `E2` beat `E0/E1`, and the winning full-data family is still the edge-only one. | high | no |
| `support_weight_alpha` | Global intensity of support-aware upweighting. In code, weight starts as `1 + alpha * edge_term`; corners get an extra multiplicative factor. | Stronger emphasis on edge/corner samples. | Too high can distort the fit toward rare regions and hurt center/global reconstruction. | `1.25`, `1.50`, `1.75` | In quick runs, moderate values worked best. In full-data overnight, the best winner kept the baseline-strength weighting rather than softer or more aggressive variants. | medium | maybe later |
| `support_weight_tau` | Edge threshold on normalized `r_inf`. Above this, the edge term ramps from `0` to `1`. Lower = start weighting earlier; higher = only near the outer boundary. | Higher values localize weighting closer to the perimeter. Lower values spread weighting over more of the support. | Too low makes almost everything “edge”; too high makes the signal too sparse. | `0.75`, `0.80`, `0.82`, `0.84` | Full-data winner used the baseline `0.75`, which is less localized. More localized variants (`0.80+`) did not win the overnight grid. | medium-high | no |
| `support_weight_tau_corner` | Corner threshold on `cornerness_norm`. Above this, extra corner emphasis turns on. Lower = corners are emphasized earlier; higher = only very corner-like points get extra pressure. | Higher values isolate true corners; lower values spread corner pressure toward edge-near-corner regions. | Too aggressive corner isolation may make training noisy. | `0.35`, `0.40`, `0.45`, `0.50` | Full-data winner again stayed at the baseline `0.35`. Stronger corner focus looked interesting in training but did not beat the control in final protocol score. | medium | no |
| `support_weight_max` | Hard cap on the support-aware sample weights. | Allows edge/corner examples to dominate more strongly. | Too high can destabilize optimization or overfit rare support regions. | `2.4`, `2.5`, `3.0` | The overnight full-data winner used the highest cap (`3.0`). Softer caps (`2.4/2.5`) did not improve final results. | medium | maybe later |
| `lambda_coverage` | Multiplier for the auxiliary coverage/tail calibration loss. This loss compares real vs generated residual coverage at central levels and tail mass at selected quantiles. | Pushes the model harder to match central coverage and tail occupancy. | If too strong, it can fight reconstruction and destabilize the statistical shape. | `0.20`, `0.22`, `0.25` in the recent support line; older historical support sweeps also touched `0.0`, `0.02`, `0.05`, higher values elsewhere in the repo. | In `E3c` quick, softer coverage (`0.20`) helped. In **full-data E2 overnight**, the winner reverted to the stronger baseline `0.25`. So full data currently seems to prefer stronger coverage pressure than the quick retunes did. | high | no |
| `coverage_temperature` | Smoothness of the sigmoid approximation inside the coverage/tail loss. Lower temperature makes the coverage test sharper/harder; higher makes it softer. | Lower values sharpen the loss around the quantile threshold. | Too low can create harsh gradients and training brittleness; too high can make the loss too weak/blurry. | `0.025`, `0.03`, `0.035`, `0.04` | `0.03` is still the safest value. `0.04` was too soft in earlier retunes, and `0.025` did not win the full-data overnight sweep. | medium-high | no |
| `tail_levels` | Which tail quantiles the coverage loss explicitly targets. For example `[0.05, 0.95]` means lower/upper 5% tails; `[0.02, 0.98]` is more extreme. | Tighter tails force the model to care about more extreme mass. | Can over-focus on very rare events and hurt overall fit. | `[0.05, 0.95]`, `[0.02, 0.98]` | Tighter tails looked scientifically appealing, but in the full-data overnight sweep they did not beat the standard `[0.05, 0.95]`. | medium-high | no |
| `support_filter_mode` | Hard support filter. `disk_l2` keeps only samples inside a power-matched disk; `none` keeps the full square support. | Restricts the effective support seen during train/eval. | Can improve an easier subproblem while moving us away from the true full-square channel. | `none`, `disk_l2` | `disk_l2` was useful diagnostically, but it was a negative direction for the main objective. We should treat it as a probe, not the main line. | high | no |
| `support_feature_scale` | Scale used to normalize support geometry (`a_train`). In practice it is the train-derived max-abs support scale. | Rescales all geometry features. | Wrong scaling would make the geometry terms inconsistent across runs. | effectively `1.0` externally, resolved from `a_train` internally | We did not sweep this. It is infrastructure, not a current scientific knob. | high | no |

## Core Model / Training Hyperparameters

These are the "basic" hyperparameters that stayed fixed in the recent support
overnight sweep, but they are still important because they define the underlying
`S27`/`E2` family we are now trusting most.

| Hyperparameter | What it controls | Values already explored historically | Current value in the active full-data winner line | Confidence | Change now? | Reading from our evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `activation` | Nonlinearity in the MLP blocks | effectively `leaky_relu` in the winning seq line; no serious recent A/B in `shape` | `leaky_relu` | medium-low | no | It is not currently a bottleneck signal. I would not reopen it before more central knobs. |
| `arch_variant` | Overall family of the model | `delta_residual`, `gaussian`, `seq_bigru_residual`, other historical families | `seq_bigru_residual` | high | no | This is one of the most validated choices we have. The seq family clearly outlived the point-wise and Gaussian references in the strong MDN line. |
| `layer_sizes` | Width/depth of the MLP heads around the seq backbone | smaller `[64,128]`, production `[128,256,512]`, and some deeper decoder explorations in older branches | `[128,256,512]` | medium-high | maybe later | The production-scale stack has been the stable backbone of the strong S27 line. I would treat it as good enough unless we later do a capacity-only study. |
| `latent_dim` | Latent bottleneck capacity | `4`, `6`, `8` historically; `8` appears in `S26full_lat8` and `S27`-family winners | `8` | high | no | `latent_dim=8` has repeated positive evidence in the MDN line and remains the most defensible setting. |
| `beta` | KL pressure weight | `0.001`, `0.002`, `0.003` in seq history; other values in other families | `0.002` | high | no | Historically, `0.002` is the center of gravity of the best seq/MDN results. I would call it a good default, though still not sacred. |
| `free_bits` | KL floor / posterior-collapse guard | `0.0`, `0.05`, `0.10`, larger values in broader repo history | `0.10` | high | no | This has been part of the stable winning seq bundle. Not the first knob I would change. |
| `lr` | Optimizer step size | `3e-4`, `2e-4`, `4e-4` in surrounding history | `2e-4` | medium-high | maybe later | In the later seq/shape work, `2e-4` looks safer. We already saw cases where larger LR made variants brittle. |
| `batch_size` | Optimization noise vs throughput | `8192`, `6144`, larger in other families | `6144` | medium | maybe later | It is working, but I see it as a throughput/stability compromise more than a proven scientific optimum. |
| `kl_anneal_epochs` | How slowly KL pressure turns on | `3` in tiny smoke runs, `80` in production, `100` in some old sweeps | `80` | medium-high | no | This has enough historical support that I would not move it unless we specifically chase optimization stability. |
| `window_size` | Temporal context length seen by the seq model | strong evidence around `7`; other sizes were not the active winning branch focus | `7` | high | no | `W=7` is part of the seq line identity and has been repeatedly validated as a reasonable temporal context. |
| `window_stride` | How densely windows are sampled | effectively `1` in the active line | `1` | medium-low | no | This is mostly an infrastructure choice right now; no strong reason to reopen before more meaningful knobs. |
| `window_pad_mode` | How border windows are padded | effectively `edge` in the active line | `edge` | medium-low | no | Logical and stable, but not something we really optimized scientifically. |
| `seq_hidden_size` | BiGRU hidden width | `64`, `128` in the seq exploratory history | `64` | high | no | We did test `64` vs `128` historically, and `64` survived as the practical winner. |
| `seq_num_layers` | Number of stacked recurrent layers | mostly `1` in the winning line | `1` | medium | maybe later | It may simply be the right bias-variance point, but we have not re-opened it seriously in the current support line. |
| `seq_bidirectional` | Whether context is read in both directions | effectively `True` in the winning seq family | `True` | medium-high | no | This is part of the established seq family and not currently the weak link. |
| `seq_gru_unroll` | Runtime/kernel style of the GRU implementation | `True` in the main line, `False` in some fast-stage operational runs | `True` | low (scientific), high (operational) | no | I do not read this as a science knob. It affects runtime/backend behavior more than the target hypothesis. |

## Practical Answer: Are the Basic Hyperparameters Well-Matched?

Not all to the same degree.

### Strongly matched today

- `arch_variant=seq_bigru_residual`
- `latent_dim=8`
- `beta=0.002`
- `free_bits=0.10`
- `window_size=7`
- `seq_hidden_size=64`

These have repeated positive evidence from the historical `S27`/MDN line and
still make sense in the current support-aware branch.

### Good enough / production-stable, but less conclusively optimized

- `layer_sizes=[128,256,512]`
- `lr=2e-4`
- `batch_size=6144`
- `kl_anneal_epochs=80`
- `seq_num_layers=1`
- `seq_bidirectional=True`

I trust this bundle, but I would call it "well matched operationally" more than
"proven globally optimal".

### Mostly inherited / not where I would spend the next scientific budget

- `activation=leaky_relu`
- `window_stride=1`
- `window_pad_mode=edge`
- `seq_gru_unroll=True`

These are fine, but not the place where current evidence says the main gain is
hiding.

## Practical Reading of the Knobs

### `support_weight_*`: "where do we spend gradient?"

These knobs do **not** change the target distribution directly. They change how
much each sample matters while fitting the same model.

- `alpha` controls *how much*
- `tau` controls *how close to the edge*
- `tau_corner` controls *how specifically corner-like*
- `weight_max` controls *how far the weighting is allowed to go*

Our full-data evidence suggests:

- the model wants **edge pressure**
- but not an over-localized or over-fancy version of it
- the simple baseline edge weighting is still the most robust setting

### `lambda_coverage`, `tail_levels`, `coverage_temperature`: "how much do we force tail calibration?"

These knobs act through the auxiliary calibration loss:

- central coverage targets at `coverage_levels`
- tail-mass targets at `tail_levels`
- smoothness controlled by `coverage_temperature`
- global strength controlled by `lambda_coverage`

Our runs suggest:

- too-soft coverage can leave G5 on the table
- too-aggressive tail sharpening did not win in full data
- the standard setting `lambda_coverage=0.25`, `tail_levels=[0.05, 0.95]`,
  `coverage_temperature=0.03` remains the strongest robust baseline in full
  data for the `E2` family

## What We Actually Learned from the Full-Data Overnight Sweep

### Sweep

Recent overnight full-data `E2` retune:

- control:
  - `lambda_coverage=0.25`
  - `tail_levels=[0.05, 0.95]`
  - `coverage_temperature=0.03`
  - `alpha=1.5`, `tau=0.75`, `tau_corner=0.35`, `weight_max=3.0`
- softlocal:
  - `lambda_coverage=0.20`
  - `coverage_temperature=0.035`
  - `alpha=1.25`, `tau=0.80`, `tau_corner=0.40`, `weight_max=2.4`
- tailfocus:
  - `lambda_coverage=0.22`
  - `tail_levels=[0.02, 0.98]`
  - `coverage_temperature=0.025`
  - `alpha=1.25`, `tau=0.80`, `tau_corner=0.40`, `weight_max=2.4`
- hybrid:
  - `lambda_coverage=0.22`
  - `tail_levels=[0.02, 0.98]`
  - `coverage_temperature=0.03`
  - `alpha=1.50`, `tau=0.82`, `tau_corner=0.45`, `weight_max=2.5`
- cornerfocus:
  - `lambda_coverage=0.20`
  - `coverage_temperature=0.03`
  - `alpha=1.75`, `tau=0.84`, `tau_corner=0.50`, `weight_max=2.5`

### Outcome

The winner was the **control** again.

That means:

- we improved from `4/12` to `5/12`
- but the improvement came from a **rerun of the robust baseline**, not from a
  more exotic local retune
- the line is currently telling us:
  - keep the `E2` family
  - trust the baseline edge-only setup
  - be cautious about over-localizing edge/corner pressure

## Current Best Interpretation

If the goal is to improve the support-aware model without drifting away from the
physical channel:

1. `support_weight_mode=edge_rinf_corner` is worth keeping.
2. `support_feature_mode=geom3` is not the safest full-data direction right now.
3. The strongest full-data baseline remains close to:
   - `lambda_coverage=0.25`
   - `tail_levels=[0.05, 0.95]`
   - `coverage_temperature=0.03`
   - `support_weight_alpha=1.5`
   - `support_weight_tau=0.75`
   - `support_weight_tau_corner=0.35`
   - `support_weight_max=3.0`
4. The remaining bottleneck looks more like:
   - regime-specific tail fidelity (`G5`)
   - not “missing geometry features” by itself
