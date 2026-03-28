# Working State

This is the single active working note for the current worktree.

## Current Worktree

- active worktree:
  - `/workspace/2026/feat_seq_bigru_residual_cvae`
- active branch:
  - `feat/seq-imdd-graybox-mdn`
- additional registered worktree:
  - `/workspace/2026/feat_seq_bigru_residual_mdn_route`
- git worktree count:
  - `2`

## What This Branch Is For

- implement `seq_imdd_graybox + MDN`
- keep the IM/DD gray-box inductive bias
- test whether a non-Gaussian decoder fixes the residual-shape problem that the
  Gaussian gray-box route could not solve

## Branch Status

- implementation status:
  - complete
- key code change:
  - `src/models/cvae_sequence.py` now supports `decoder_distribution="mdn"`
    for `seq_imdd_graybox`
- new presets:
  - `seq_imdd_graybox_mdn_smoke`
  - `seq_imdd_graybox_mdn_guided_quick`
- test coverage:
  - build
  - save/load
  - inference
  - fit smoke
  - preset selection

## Most Relevant Runs

- gray-box Gaussian anchor:
  - `outputs/exp_20260327_172148`
  - `6/12`
- gray-box Gaussian guided large:
  - `outputs/exp_20260327_183153`
  - `5/12`
- gray-box + MDN smoke:
  - `outputs/exp_20260328_003030`
  - `0/12`
  - purpose: plumbing only
- gray-box + MDN guided quick:
  - `outputs/exp_20260328_023302`
  - `5/12`

## Current Reading

- `seq_imdd_graybox + MDN` is a valid implemented route.
- It did not beat the best gray-box Gaussian anchor.
- It also did not justify staying as the main long-grid lane.
- The immediate next lane has already moved to the dedicated
  `seq_bigru_residual + MDN` worktree because that family still owns the best
  historical MDN results.

## Recent Decision

- keep this branch as the implementation and audit trail for the gray-box + MDN
  route
- do not open another long gray-box + MDN sweep yet
- use the dedicated worktree `feat/seq-bigru-residual-mdn-route` for the next
  reproducibility investigation

## Route Queue

1. `feat/seq-bigru-residual-mdn-route`
   - explain why the historical `9/12` MDN line is not reproducing under the
     current stack
2. only after that, decide whether the next work should be:
   - a local `seq_bigru_residual + MDN` follow-up
   - or a return to gray-box work with a much tighter hypothesis

## Do Not Reopen Blindly

- `sample-aware MMD`
- the current `sinh-arcsinh` flow line
- pure regime-resampling as the main intervention
- another broad gray-box sweep without first explaining why the historical MDN
  anchors are not reproducing

## Where The Full Matrix Lives

For the consolidated list of:

- branches
- worktrees
- route families
- tested variations
- recent results
- current queue

read:

- [ROUTES_AND_RESULTS.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/ROUTES_AND_RESULTS.md)

## Minimal Read Order

1. [README.md](/workspace/2026/feat_seq_bigru_residual_cvae/README.md)
2. [PROJECT_STATUS.md](/workspace/2026/feat_seq_bigru_residual_cvae/PROJECT_STATUS.md)
3. [ROUTES_AND_RESULTS.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/active/ROUTES_AND_RESULTS.md)
4. [PROTOCOL.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/PROTOCOL.md)
