# Outputs Layout

`cVAe_2026` is not the canonical location for new runtime outputs.

New benchmark artifacts must stay next to the architecture and candidate that
were evaluated, in the corresponding active worktree, for example:

```text
/home/rodrigo/cVAe_2026_full_square/outputs/architectures/<architecture>/<candidate>/benchmarks/16qam/
/home/rodrigo/cVAe_2026_full_circle/outputs/architectures/<architecture>/<candidate>/benchmarks/16qam/
```

This directory may contain historical leftovers during migration, but it should
not be used as a new benchmark sink.
