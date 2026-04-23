# Data Layout

`cVAe_2026` is the integration/documentation worktree. Do not place new runtime
benchmark datasets here.

Operational datasets must live in the active line worktree that uses them, for
example:

- `/home/rodrigo/cVAe_2026_full_square/data/...`
- `/home/rodrigo/cVAe_2026_full_circle/data/...`

External modulation datasets such as `16QAM` and `4QAM` are shared benchmark
inputs, but their evaluated artifacts must be attached to the architecture/run
that consumed them.
