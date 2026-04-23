# Data Layout

This tree separates line-specific datasets from transversal benchmark datasets.

- `lines/full_square/`: datasets owned by the `full_square` line
- `lines/full_circle/`: datasets owned by the `full_circle` line
- `benchmarks/modulations/<mod>/`: external modulation benchmarks such as `16qam` and `4qam`

Large binary datasets remain outside git or under LFS when explicitly allowed.
