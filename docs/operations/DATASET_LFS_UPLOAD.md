# Dataset Upload to Git LFS

This guide standardizes how to import a new acquisition dataset into the
repository under `data/` and publish it with Git LFS.

## Naming Convention

Use:

- `data/dataset_<name>_organized/`

Examples:

- `data/dataset_fullsquare_organized/`
- `data/dataset_16qam_organized/`

## Prerequisites

From repo root:

```bash
git lfs install --local
git lfs version
```

If your dataset is on Windows drive `H:`, in WSL the path is usually:

```bash
/mnt/h/...
```

## One-command Import (recommended)

Use the helper script:

```bash
scripts/ops/import_dataset_lfs.sh "<source_dir>" "<dataset_dir_name>"
```

Example for the 16QAM dataset:

```bash
scripts/ops/import_dataset_lfs.sh \
  "/mnt/h/Meu Drive/Doutorado/IA/1-Data/Dataset/16QAM_DATASET_2026" \
  "dataset_16qam_organized"
```

What the script does:

1. Validates source has `IQ_data` folders and expected sent/received `.npy`.
2. Copies source into `data/<dataset_dir_name>/`.
3. Prints summary and next git commands.

## Commit and Push

After import:

```bash
git add .gitattributes .gitignore data/dataset_16qam_organized
git status --short
git lfs ls-files | grep -E "dataset_16qam_organized|\\.npy|\\.png"
git commit -m "data: add dataset_16qam_organized via Git LFS"
git push
```

## Sanity-check For Training/Evaluation

Run with explicit dataset root:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_16qam_organized \
  --output_base outputs \
  --protocol configs/all_regimes_full_dataset.json
```

Or with wrapper:

```bash
DATASET_ROOT="$PWD/data/dataset_16qam_organized" bash scripts/ops/train.sh
```

## Notes

- Keep `X.npy`/`Y.npy` (or legacy aliases) inside each `IQ_data/` folder.
- Keep `dist_*` and `curr_*` folder names for automatic regime parsing.
- If `selected_experiments` is hardcoded in a protocol JSON for another
  dataset root, remove/adjust that list for portability.
