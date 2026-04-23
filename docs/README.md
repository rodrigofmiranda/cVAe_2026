# Documentation Map

This folder is organized by function so the public entrypoints, active working
notes, reference material, and archive do not all compete at the same level.

## Public Entry Points

Start with these if you are arriving from GitHub or onboarding into the repo:

- [../README.md](../README.md) - public repository overview
- [BRANCH_GUIDE.md](BRANCH_GUIDE.md) - what each public branch is for
- [../PROJECT_STATUS.md](../PROJECT_STATUS.md) - current codebase and scientific status

## Internal Operations

These are especially relevant for the shared lab server:

- [active/FULL_SQUARE_LINEAGE_TO_SHAPE.md](active/FULL_SQUARE_LINEAGE_TO_SHAPE.md) - single-document history of the `full_square` research line from traditional cVAE to `shape`, with a recommended simple-to-complex restart order for `full_circle`
- [active/FULL_CIRCLE_NEXT_STEP.md](active/FULL_CIRCLE_NEXT_STEP.md) - why `shape` was a proxy, what `full_circle` now needs to prove, and the concrete next validation step
- [active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md](active/FULL_CIRCLE_CLEAN_RUN_CHECKLIST.md) - short operational checklist for the next clean `full_circle` validation run
- [active/INFRA_GUIDE.md](active/INFRA_GUIDE.md) - SSH, users, Docker, tmux, and the local-first dataset policy
- [active/MULTI_PC_WORKFLOW.md](active/MULTI_PC_WORKFLOW.md) - worktree layout, two hot slots, and how to operate across PCs
- [operations/DUAL_RUN_MANDATORY_RULES.md](operations/DUAL_RUN_MANDATORY_RULES.md) - short mandatory memo for what is and is not canonically allowed when two runs share one machine/GPU
- [operations/DUAL_RUN_HEAVY_LOAD_POSTMORTEM_2026-04-22.md](operations/DUAL_RUN_HEAVY_LOAD_POSTMORTEM_2026-04-22.md) - audited comparison between the capped dual runs that worked and the full-data dual run that failed during auxiliary inference
- [operations/CANONICAL_DUAL_RUN_STANDARD.md](operations/CANONICAL_DUAL_RUN_STANDARD.md) - canonical `algo1`/`algo2` sessions, default directories, launch wrapper, and naming rules
- [operations/DATASET_LFS_UPLOAD.md](operations/DATASET_LFS_UPLOAD.md) - local-first dataset placement policy; Git LFS is exceptional only

## Reference Docs

- [PROTOCOL.md](PROTOCOL.md) - protocol runner, CLI, artifacts
- [MODELING_ASSUMPTIONS.md](MODELING_ASSUMPTIONS.md) - modeling rationale
- [DIAGNOSTIC_CHECKLIST.md](DIAGNOSTIC_CHECKLIST.md) - diagnostic workflow
- [reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt](reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt) - acquisition-to-training path

## Archive

- [archive/REFACTOR_PLAN_legacy.md](archive/REFACTOR_PLAN_legacy.md) - historical refactor context
