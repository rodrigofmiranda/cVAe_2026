# Conditional Density Decoder Guide

This note is the short guide for decoder-family decisions in the dedicated
flow worktree.

## Current Reading

- The scientific problem is conditional density estimation:
  - learn `p(noise | signal window, distance, current, context)`
- Gaussian decoders are too regular for the residual-shape gap we keep seeing.
- MDN improves shape, but still leaves systematic constellation under-structure.
- The old flow branch that was already tried was narrow:
  - per-axis `sinh-arcsinh`
  - useful as plumbing proof, negative as a scientific direction
- That negative result does **not** rule out richer conditional flows.

## What This Worktree Adds

- dedicated branch:
  - `feat/seq-bigru-residual-spline-flow-v2`
- dedicated worktree:
  - `/workspace/2026/feat_seq_bigru_residual_spline_flow_v2`
- active seq decoder families in this lane:
  - `flow_family="spline_2d"`:
    - new default here
    - decoder-conditioned rational-quadratic spline flow per residual axis
    - intended to move beyond both the old axis-separable flow and the first
      coupling-flow retry
  - `flow_family="coupling_2d"`:
    - kept only as compatibility / reference from flow v1
  - `flow_family="sinh_arcsinh"`:
    - kept only as legacy compatibility / control

## Presets

- `seq_flow_spline_smoke`
  - single-candidate structural smoke
  - proves build, training, save/load, inference, and protocol wiring
- `seq_flow_spline_guided_quick`
  - small guided quick grid
  - starts from the strongest seq residual region we know
  - changes decoder family first, not the whole architecture

## Results In This Lane

- run:
  - `outputs/exp_20260329_015508`
- preset:
  - `seq_flow_spline_smoke`
- result:
  - `0/12`
- reading:
  - structural success, scientific failure
  - the route is integrated end-to-end
  - the smoke is too short to judge the family on merit
  - training diagnostics flagged:
    - `flag_undertrained=True`
    - `flag_posterior_collapse=True`
    - `active_dim_ratio=0.0`

- run:
  - `outputs/exp_20260329_015815`
- preset:
  - `seq_flow_spline_guided_quick`
- result:
  - `0/12`
- reading:
  - this is the decisive result for the current family
  - all four grid candidates also stayed at `0/12` in the mini protocol
  - `gate_g5_pass=0`
  - diagnostics no longer support the idea that the failure is just budget:
    - `flag_undertrained=False`
    - `flag_posterior_collapse=False`
    - `active_dim_ratio=1.0`
    - `flag_unstable=True`
  - the present `flow_family="spline_2d"` is therefore a branch-local
    negative result

## Practical Next Step

- Do **not** open another sweep of the current `spline_2d` line.
- Keep this branch as the formal record that:
  - the route was implemented correctly
  - the route was tested beyond smoke
  - the current formulation failed scientifically
- If flow is revisited later, the next escalation is:
  - a materially different conditional flow family
  - or conditional diffusion
