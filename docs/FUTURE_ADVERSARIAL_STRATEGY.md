# Future Adversarial Strategy

This document is the only active note for a future return of the archived
`delta_residual_adv` idea.

## Why It Was Removed

- the active worktree now prioritizes scientific momentum on
  `seq_bigru_residual` and `delta_residual`
- the adversarial path added too much code and operational complexity for the
  current phase
- we want a cleaner main branch while keeping the idea documented

The old local worktree for this idea was deleted to save disk space.
If this strategy comes back, recover it from the branch name
`feat/delta-residual-adv` or from git history.

## Target Idea

If we revisit the strategy, the goal is still:

- keep the `delta_residual` backbone as the generator
- add a conditional discriminator over `(x, d, c, Δ)`
- preserve the same protocol-first evaluation used by the non-adversarial lines

## Minimal Reintroduction Plan

Implement the comeback in this order:

1. Recreate `arch_variant="delta_residual_adv"` in `src/models/cvae.py`.
2. Add a compact conditional discriminator module in `src/models/discriminator.py`.
3. Add a wrapper model in `src/models/adversarial.py` that:
   - owns `encoder`, `prior_net`, `decoder`, and `discriminator`
   - performs one discriminator step and one generator step
   - keeps checkpoint layer names compatible with `create_inference_model_from_full`
4. Restore adversarial hyperparameters in `src/training/grid_plan.py`:
   - `lambda_adv`
   - `disc_layer_sizes`
   - start with a single smoke preset
5. Restore save/load coverage in:
   - `src/models/cvae_sequence.py`
   - `tests/test_cvae_architecture_variants.py`
   - `tests/test_pipeline_dry_run.py`

## Training Rules If It Comes Back

- start with a single smoke preset, not a wide grid
- compare against the current best `delta_residual` anchor
- compare against the current best `seq_bigru_residual` reference
- keep protocol evaluation identical to the non-adversarial lines
- do not promote the line without fresh post-reimplementation reruns

## Validation Checklist

Before treating a new adversarial rerun as meaningful, verify:

- the discriminator consumes sampled residuals, not only residual means
- save/load roundtrips keep the full trainable wrapper intact
- `dry_run` works without special-case breakage
- the reduced multi-regime protocol finishes end to end
- the new run improves residual-distribution metrics without destroying gate stability

## Decision Rule

Do not reopen this strategy unless one of these is true:

- `delta_residual` plateaus scientifically and still misses the needed residual structure
- `seq_bigru_residual` remains too expensive or too brittle for the target deployment path
- a focused adversarial rerun is needed to test a concrete hypothesis, not just curiosity
