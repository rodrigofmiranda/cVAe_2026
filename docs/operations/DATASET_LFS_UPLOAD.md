# Dataset Placement Policy (Local Path First, Git LFS Only By Exception)

This guide standardizes how to place acquisition datasets under `data/` using
the real local filesystem path on the active machine.

The filename is kept for backward compatibility, but the operational policy is
now explicit:

- default flow: local copy into `data/`
- default prohibition: do not publish full acquisition datasets via Git LFS
- Git LFS is exceptional only, and requires explicit approval because it can
  exhaust the shared data quota quickly

## Mandatory Rule

On the lab server and on active worktrees:

- copy the dataset directly into the real local path under `data/`
- do not treat `git add data/...` as the standard dataset workflow
- do not use Git LFS for `full_square`, `full_circle`, `16QAM`, or similar full
  acquisition trees unless there is an explicit archival/sharing decision

## Canonical Local Dataset Roots

Use these names inside the repository:

- `data/dataset_fullsquare_organized`
- `data/FULL_CIRCLE_2026`
- `data/16qam`

## Current Host Examples

Current real-path copies already organized on this host include:

- `/home/rodrigo/cVAe_2026_full_square/data/dataset_fullsquare_organized`
- `/home/rodrigo/cVAe_2026_full_square/data/FULL_CIRCLE_2026`
- `/home/rodrigo/cVAe_2026_full_square/data/16qam`
- `/home/rodrigo/cVAe_2026_shape_fullcircle/data/FULL_CIRCLE_2026`
- `/home/rodrigo/cVAe_2026_shape_fullcircle/data/16qam`

## Recommended Local Copy Flow

From repo root, copy by real path with `rsync` or `cp -a`.

Examples:

```bash
rsync -a --info=progress2 \
  /home/rodrigo/cVAe_2026_full_square/data/dataset_fullsquare_organized/ \
  data/dataset_fullsquare_organized/

rsync -a --info=progress2 \
  /home/rodrigo/cVAe_2026_full_square/data/FULL_CIRCLE_2026/ \
  data/FULL_CIRCLE_2026/

rsync -a --info=progress2 \
  /home/rodrigo/cVAe_2026_full_square/data/16qam/ \
  data/16qam/
```

Equivalent copy with `cp -a` is also acceptable when the source and target are
on the same machine.

## What Not To Do By Default

Do not use this as the standard operational path:

```bash
git add data/...
git commit
git push
git lfs push
```

For the active datasets, this wastes quota and couples routine experimentation
to repository storage limits.

## Sanity-check For Training/Evaluation

Run with explicit dataset root when needed:

```bash
python -m src.protocol.run \
  --dataset_root data/16qam \
  --output_base outputs \
  --protocol configs/all_regimes_full_dataset.json
```

Or with wrapper:

```bash
DATASET_ROOT="$PWD/data/16qam" bash scripts/ops/train.sh
```

## Exceptional Git LFS Flow

Use Git LFS only if there is an explicit decision to publish or archive a
curated dataset snapshot through the repository.

Only in that exceptional case should you use:

```bash
scripts/ops/import_dataset_lfs.sh "<source_dir>" "<dataset_dir_name>"
git add .gitattributes .gitignore data/<dataset_dir_name>
git lfs ls-files | grep -E "<dataset_dir_name>|\\.npy|\\.png"
git commit -m "data: add <dataset_dir_name> via Git LFS"
git push
```

## Notes

- Keep `IQ_data/`, `metadata.json`, and `report.json` inside each experiment leaf.
- Keep `dist_*` and `curr_*` folder names for automatic regime parsing.
- If `selected_experiments` is hardcoded in a protocol JSON for another dataset
  root, remove or adjust that list for portability.
