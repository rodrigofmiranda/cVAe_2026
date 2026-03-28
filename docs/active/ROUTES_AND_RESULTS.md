# Routes And Results

This document is the compact registry of the branches, worktrees, routes, and
recent experimental outcomes that matter right now.

## Worktree Inventory

| Worktree | Branch | Role | Current status |
|---|---|---|---|
| `/workspace/2026/feat_seq_bigru_residual_cvae` | `feat/seq-imdd-graybox-mdn` | Implement and test `seq_imdd_graybox + MDN` | Implemented, tested, not promoted |
| `/workspace/2026/feat_seq_bigru_residual_mdn_route` | `feat/seq-bigru-residual-mdn-route` | Dedicated rerun lane for `seq_bigru_residual + MDN` | Dedicated rerun completed, reproducibility gap open |

## Canonical Anchors

| Run | Family / reading | Pass | G3 | G5 | G6 | Score |
|---|---|---:|---:|---:|---:|---:|
| `exp_20260324_023558` | Stable Gaussian reference | 10/12 | 10 | 11 | 10 | 0.7528 |
| `exp_20260325_230938` | Best historical MDN | 9/12 | 12 | 9 | 12 | 0.3406 |
| `exp_20260327_161311` | Previous-branch MDN benchmark | 9/12 | 11 | 10 | 12 | 0.3179 |
| `exp_20260327_172148` | Gray-box Gaussian anchor | 6/12 | 9 | 11 | 6 | 0.7798 |
| `exp_20260327_183153` | Gray-box Gaussian guided large | 5/12 | 10 | 9 | 6 | 0.5822 |
| `exp_20260328_023302` | Gray-box + MDN guided quick | 5/12 | 9 | 9 | 5 | 0.9632 |
| `exp_20260328_041729` | Dedicated seq MDN rerun | 4/12 | 9 | 6 | 5 | 0.9617 |

## Variations Tested

| Route family | Typical presets / runs | Result | Reading |
|---|---|---|---|
| `seq_bigru_residual` Gaussian | `exp_20260324_023558` | Strongest verified overall reference | Still the strongest stable baseline in this repo snapshot |
| `seq_bigru_residual + MDN` historical | `exp_20260325_230938`, `exp_20260327_161311` | `9/12` | Confirms MDN can model the channel well when the stack is right |
| `seq_bigru_residual + MDN v2` fastbase / local follow-ups | `exp_20260327_021632`, `exp_20260327_032019` | `5/12`, `6/12` | Coverage and local G5 tuning helped, but did not recover the old `9/12` ceiling |
| `seq_bigru_residual + MDN` tail explore | `exp_20260327_050422` | `5/12` | Tail-level sweep alone did not unlock `0.8 m` |
| `seq_bigru_residual + MDN` dedicated rerun branch | `exp_20260328_041729` | `4/12` | Current reproducibility is worse than the historical MDN anchors |
| `seq_imdd_graybox` Gaussian | `exp_20260327_172148`, `exp_20260327_183153` | `6/12`, `5/12` | Viable but below the best MDN family |
| `seq_imdd_graybox + MDN` smoke | `exp_20260328_003030` | `0/12` | Plumbing only; not a scientific result |
| `seq_imdd_graybox + MDN` guided quick | `exp_20260328_023302` | `5/12` | Route is implemented and valid, but not better than gray-box Gaussian |
| `sample-aware MMD` | historical branch results | Negative | Do not reopen as the main lane |
| Current `sinh-arcsinh` flow line | historical branch results | Negative | Discarded for this cycle |
| Pure regime-resampling | historical branch results | Negative | Did not solve the remaining gap |

## Branch-Specific Notes

### `feat/seq-imdd-graybox-mdn`

- Main code change:
  - `seq_imdd_graybox` now supports `decoder_distribution="mdn"`
- New presets:
  - `seq_imdd_graybox_mdn_smoke`
  - `seq_imdd_graybox_mdn_guided_quick`
- Status:
  - route implemented correctly
  - route tested end-to-end
  - route not promoted as main search lane

### `feat/seq-bigru-residual-mdn-route`

- Purpose:
  - isolate the `seq_bigru_residual + MDN` family from gray-box work
  - check whether the historical MDN line still reproduces under the current
    stack
- Current dedicated rerun:
  - `exp_20260328_041729`
  - `4/12`
- Status:
  - branch is useful as a reproducibility / regression lane
  - current result does not justify a blind new sweep yet

## Current Reading

- `seq_imdd_graybox + MDN` solved the implementation problem, not the modeling
  problem.
- The best gray-box result is still the Gaussian anchor `exp_20260327_172148`
  at `6/12`.
- The dedicated `seq_bigru_residual + MDN` rerun did not reproduce the old
  `9/12` anchors.
- The immediate blocker is now reproducibility drift:
  - environment
  - defaults
  - ranking path
  - evaluation stack
- The unresolved scientific zone is still concentrated in `0.8 m`, with G5/G6
  failures dominating low-current regimes.

## Current Queue

1. Explain why the historical `seq_bigru_residual + MDN` line no longer
   reproduces `9/12` under the current stack.
2. Only after reproducing that baseline, reopen local G5/G6 tuning.
3. Keep `seq_imdd_graybox + MDN` available as an implemented branch, but do
   not prioritize another long gray-box grid yet.

## Do Not Reopen

- `sample-aware MMD` as the main route
- the current `sinh-arcsinh` flow line
- pure regime-resampling as the main intervention
