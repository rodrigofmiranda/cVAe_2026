# Working State

This is the single active working note for the current worktree.

## Current Worktree

- active worktree:
  - `/workspace/2026/feat_seq_bigru_residual_mdn_route`
- active branch:
  - `feat/seq-bigru-residual-mdn-route`
- sister worktree:
  - `/workspace/2026/feat_seq_bigru_residual_cvae`
- git worktree count:
  - `2`

## What This Branch Is For

- isolate the `seq_bigru_residual + MDN` family from gray-box work
- reproduce the strongest historical MDN line under the current stack
- determine whether the immediate problem is scientific or operational

## Branch Status

- dedicated rerun completed:
  - `outputs/exp_20260328_041729`
  - result: `4/12`
- historical anchors still better:
  - `outputs/exp_20260325_230938`
  - `outputs/exp_20260327_161311`
  - both at `9/12`

## Current Reading

- this branch did not reproduce the old MDN ceiling
- therefore the next problem on this branch is reproducibility drift
- this is not the moment for a broader MDN hyperparameter sweep
- the branch should now be used to compare:
  - environment behavior
  - effective defaults
  - ranking and evaluation path differences

## Most Relevant Runs

- historical MDN anchor:
  - `outputs/exp_20260325_230938`
  - `9/12`
- previous-branch MDN benchmark:
  - `outputs/exp_20260327_161311`
  - `9/12`
- dedicated rerun on this branch:
  - `outputs/exp_20260328_041729`
  - `4/12`

## Immediate Next Actions

1. explain why the old MDN line no longer reproduces `9/12`
2. only after that, reopen local G5/G6 tuning on `seq_bigru_residual + MDN`
3. keep the gray-box + MDN branch available as a separate implemented route,
   but do not mix that code path into this reproducibility lane

## Do Not Reopen Blindly

- another broad `seq_bigru_residual + MDN` sweep before explaining the rerun gap
- `sample-aware MMD`
- the current `sinh-arcsinh` flow line
- pure regime-resampling as the main intervention

## Where The Full Matrix Lives

For the consolidated list of:

- branches
- worktrees
- tested route families
- recent results
- current queue

read:

- [ROUTES_AND_RESULTS.md](/workspace/2026/feat_seq_bigru_residual_mdn_route/docs/active/ROUTES_AND_RESULTS.md)

## Minimal Read Order

1. [README.md](/workspace/2026/feat_seq_bigru_residual_mdn_route/README.md)
2. [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_mdn_route/PROJECT_STATUS.md)
3. [ROUTES_AND_RESULTS.md](/workspace/2026/feat_seq_bigru_residual_mdn_route/docs/active/ROUTES_AND_RESULTS.md)
4. [PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_mdn_route/docs/reference/PROTOCOL.md)
