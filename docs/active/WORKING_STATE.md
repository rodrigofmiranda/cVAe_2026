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

## Execution Provenance

- remote lane:
  - `5090` host
  - long runs usually executed via `scripts/ops/run_tf25_gpu.sh` under host
    `tmux`
- local lane:
  - `A6000` workstation backing the current workspace session
  - current reruns and audits executed locally
- artifact scope:
  - only a subset of remote `5090` experiments is copied into this local
    workspace
  - some historical anchors therefore exist here as imported artifacts, not as
    locally generated runs
- branch scope:
  - the remote `5090` host may be working on a different branch in parallel
    from the local `A6000` session
  - reported remote snapshot on `2026-03-28`:
    - repo path `~/RODRIGO/cVAe_2026`
    - branch `feat/mdn-g5-recovery`
    - clean working tree

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
- The dedicated MDN reproducibility lane now has a controlled isolation run:
  - `outputs/exp_20260328_181213`
  - exact historical candidate tag
  - `2/12`
- The current reproducibility blocker is now understood more sharply:
  - historical `9/12` MDN anchors ran on `Python 3.12.3 / TensorFlow 2.17.0 /
    NumPy 1.26.4`
  - current reruns are on `Python 3.8.10 / TensorFlow 2.8.0 / NumPy 1.21.1`
  - host provenance also changed between the remote `5090` lane and the local
    `A6000` lane
  - branch provenance may also differ between those two lanes
  - data caps matched, so data quantity is not the main explanation
  - seed is set once at pipeline start, so isolated single-candidate reruns are
    runtime probes, not exact reproductions of a late grid position
- decoder-family note:
  - the old `feat/conditional-flow-decoder` branch already tested a narrow
    conditional flow family and discarded it
  - this does **not** discard richer conditional density decoders
  - guide:
    - [reference/CONDITIONAL_DENSITY_DECODER_GUIDE.md](/workspace/2026/feat_seq_bigru_residual_cvae/docs/reference/CONDITIONAL_DENSITY_DECODER_GUIDE.md)

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
   - a `seq_bigru_residual + MDN` rerun inside the documented `tf25_gpu`
     container path
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
