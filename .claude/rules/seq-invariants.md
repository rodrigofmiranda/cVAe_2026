---
paths:
  - "src/**/*"
  - "tests/**/*"
  - "configs/**/*"
  - "docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md"
---

# Sequence Architecture Invariants

- This branch is implementing `seq_bigru_residual`, not replacing the whole pipeline ad hoc.
- Preserve existing `concat` and `channel_residual` flows unless the user explicitly asks otherwise.
- Split per experiment before windowing.
- Windows must never cross experiment boundaries or train/validation boundaries.
- Keep one output per original raw sample so protocol metrics stay comparable.
- Deterministic inference must remain compatible with `EVM/SNR`.
- Stochastic `mc_concat` inference must remain compatible with distribution metrics.
- Keep saved-model layer names compatible: `encoder`, `prior_net`, `decoder`.
- Prefer small phase-by-phase changes aligned with `docs/CLAUDE_SEQ_BIGRU_RESIDUAL_PLAN.md`.
- Ignore `knowledge/`, `data/`, and generated `outputs/` unless the user explicitly asks for them.
