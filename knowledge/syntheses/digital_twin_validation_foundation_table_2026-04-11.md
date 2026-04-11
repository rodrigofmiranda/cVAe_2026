# Digital Twin Validation Foundation Table

Date: 2026-04-11

## Purpose

This table summarizes what is already well grounded in the current
digital-twin validation stack, what is still missing, and what should be
prioritized next if the goal is a thesis-defensible validation protocol.

## Table

| Area | What We Already Have | What Is Still Missing | Priority | Practical Decision |
|---|---|---|---|---|
| Measured-channel basis | Training and evaluation are driven by measured bench data across multiple physical regimes. This is well aligned with digital-twin channel-modeling literature. | A short, explicit "validation dataset governance" note: which regimes are screening-only, which are final-validation regimes, and which external sets count as generalization checks. | High | Keep current approach; document dataset roles more formally. |
| Multi-regime validation | The protocol evaluates across distance/current regimes instead of a single operating point. This is a strong methodological choice. | A final thesis-level rule for promotion from 12-regime screening to 27-regime confirmation. | High | Keep the 12-regime protocol for development, add a final 27-regime confirmation step. |
| Multi-metric validation | `G1..G5` already cover direct signal fidelity, residual scale, covariance, PSD, and higher-order shape. This is much stronger than validating only with MSE/EVM. | A concise justification note explaining why each metric is needed and what failure mode it covers. | High | Keep `G1..G5` as the main twin-validation ladder. |
| Direct communication metrics | `G1` and `G2` use EVM and SNR relative error, which are physically interpretable and communication-relevant. | Formal empirical calibration of the current `10%` thresholds for this bench. | High | Keep them, but label them as project-calibrated engineering thresholds until calibration is documented. |
| Residual-structure metrics | `G3` and `G4` are well designed for a twin: they validate mean/dispersion relative to channel scale and spectral residual structure. | Threshold-calibration memo for `mean_rel_sigma`, `cov_rel_var`, and `PSD_L2`. | High | Keep them as core validation gates. |
| Tail / shape fidelity | `G5` explicitly checks skew, kurtosis, and JB-relative mismatch, which is well aligned with the shaping/nonlinearity literature. | Better documentation that `G5` is a project-specific engineering gate, not a universal textbook standard. Possibly recalibration under larger-`N` conditions. | High | Keep `G5`, but present it as a domain-calibrated tail/shape acceptance rule. |
| Formal distribution screen | `G6` uses valid tools: MMD, Energy distance, and BH/FDR. This is a legitimate formal complement to the engineering gates. | Reframing: `G6` should not be described as proving equivalence or full indistinguishability. It also needs standardized reporting of `stat_mode`, `stat_max_n`, and `stat_n_perm`. | High | Keep `G6`, but demote it from main veto to auxiliary statistical screen. |
| Threshold calibration | Thresholds exist and are operationally useful today. | A dedicated calibration protocol showing how `G1..G5` thresholds were chosen from accepted runs, expert judgment, or bench tolerances. | Very high | This is one of the most important missing pieces. |
| Uncertainty quantification | Some statistical pieces exist already, and PSD includes a bootstrap CI. | Confidence intervals or bootstrap uncertainty for the main acceptance metrics, not just point estimates. | Very high | Add a final uncertainty layer before thesis claims. |
| Temporal dependence / memory | The architecture is sequence-aware, and the literature strongly supports sequence effects. | A stronger explicit temporal validation gate. The code already hints at future `G7` / `delta_acf_l2`, but it is not yet gate-driving. | Very high | This is the main technical gap in the current validation stack. |
| External validation | We already used `16QAM` as an external generalization check, which is a very strong move. | A formal place for external validation in the protocol: whether it is advisory, tie-breaking, or mandatory for final model acceptance. | High | Keep `16QAM` external validation and formalize its role. |
| Statistical interpretation | We already use sound building blocks for NHST-style distribution checks. | Explicit separation between "no detected mismatch" and "equivalence." The current docs still overstate `G6` in places. | Very high | Update wording across docs and reports. |
| Final credibility protocol | We already have enough pieces for an internal development pipeline. | A thesis-ready credibility workflow: screening, finalist reruns, uncertainty, external validation, and final acceptance. | Very high | This should become a dedicated synthesis / chapter outline. |

## What Is Best Grounded Today

The strongest current foundation is:

1. measured bench data
2. multi-regime evaluation
3. multi-metric twin validation through `G1..G5`
4. residual-oriented evaluation instead of only raw-output MSE
5. external generalization checks such as `16QAM`

If we had to defend the current pipeline today, the most defensible statement
would be:

- the project already has a strong engineering validation stack for a
  data-driven digital twin, but it still needs threshold calibration,
  uncertainty quantification, and a cleaner statistical-interpretation layer
  before final thesis claims.

## Biggest Missing Pieces

If we focus on the most important gaps only, they are:

1. threshold calibration for `G1..G5`
2. uncertainty / confidence intervals on the main metrics
3. explicit temporal-memory validation
4. reframing `G6` as an auxiliary statistical screen rather than proof of equivalence
5. a final credibility workflow tying screening, external validation, and final confirmation together

## Recommended Order Of Work

1. Reframe `G6` in the protocol language.
2. Introduce `validation_status_twin = G1..G5`.
3. Keep `G6` as `stat_screen_pass`.
4. Write a threshold-calibration memo for `G1..G5`.
5. Add uncertainty reporting.
6. Promote a temporal gate (`G7` / `delta_acf_l2` or equivalent).
7. Define the final thesis-grade credibility protocol.
