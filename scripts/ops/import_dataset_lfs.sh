#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/ops/import_dataset_lfs.sh "<source_dir>" "<dataset_dir_name>"

Example:
  scripts/ops/import_dataset_lfs.sh \
    "/mnt/h/Meu Drive/Doutorado/IA/1-Data/Dataset/16QAM_DATASET_2026" \
    "dataset_16qam_organized"

Behavior:
  1) Validates source tree has IQ_data and expected .npy files.
  2) Copies source into data/<dataset_dir_name>/.
  3) Prints summary and next git/LFS commands.
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
    exit 1
fi

SOURCE_DIR="$1"
DATASET_DIR_NAME="$2"

if [[ "$DATASET_DIR_NAME" == *"/"* ]]; then
    fail "dataset_dir_name must be only a directory name (no '/'): got '$DATASET_DIR_NAME'"
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    fail "Source directory does not exist: $SOURCE_DIR"
fi

if ! command -v rsync >/dev/null 2>&1; then
    fail "rsync not found. Please install rsync and retry."
fi

if ! git lfs version >/dev/null 2>&1; then
    fail "git-lfs not available. Install git-lfs first."
fi

if ! REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    fail "Run this command inside the repository."
fi
cd "$REPO_ROOT"

SOURCE_ABS="$(cd "$SOURCE_DIR" && pwd -P)"
DEST_REL="data/$DATASET_DIR_NAME"
DEST_ABS="$REPO_ROOT/$DEST_REL"

if [[ -e "$DEST_ABS" ]]; then
    fail "Destination already exists: $DEST_REL"
fi

iq_count="$(find "$SOURCE_ABS" -type d -name 'IQ_data' | wc -l | tr -d '[:space:]')"
sent_count="$(find "$SOURCE_ABS" -type f \( -name 'X.npy' -o -name 'x_sent.npy' -o -name 'sent_data_tuple.npy' \) | wc -l | tr -d '[:space:]')"
recv_count="$(find "$SOURCE_ABS" -type f \( -name 'Y.npy' -o -name 'y_recv.npy' -o -name 'received_data_tuple_sync-phase.npy' -o -name 'received_data_tuple_sync_phase.npy' -o -name 'received_data_tuple_sync.npy' -o -name 'received_data_tuple.npy' \) | wc -l | tr -d '[:space:]')"

if [[ "$iq_count" -eq 0 ]]; then
    fail "No IQ_data directories found in source."
fi
if [[ "$sent_count" -eq 0 || "$recv_count" -eq 0 ]]; then
    fail "Source is missing sent/received .npy files (X/x_sent and Y/y_recv families)."
fi

git lfs install --local >/dev/null

echo "Importing dataset..."
echo "  source: $SOURCE_ABS"
echo "  target: $DEST_REL"
mkdir -p "$(dirname "$DEST_ABS")"
rsync -a --human-readable --info=progress2 "$SOURCE_ABS"/ "$DEST_ABS"/

dist_count="$(find "$DEST_ABS" -maxdepth 1 -type d -name 'dist_*' | wc -l | tr -d '[:space:]')"
curr_count="$(find "$DEST_ABS" -type d -name 'curr_*' | wc -l | tr -d '[:space:]')"
exp_count="$(find "$DEST_ABS" -type d -name 'IQ_data' | wc -l | tr -d '[:space:]')"
npy_count="$(find "$DEST_ABS" -type f -name '*.npy' | wc -l | tr -d '[:space:]')"
png_count="$(find "$DEST_ABS" -type f -name '*.png' | wc -l | tr -d '[:space:]')"
json_count="$(find "$DEST_ABS" -type f -name '*.json' | wc -l | tr -d '[:space:]')"
total_size="$(du -sh "$DEST_ABS" | awk '{print $1}')"

sample_npy="$(find "$DEST_ABS" -type f -name '*.npy' | head -n 1 || true)"
sample_filter_attr="n/a"
if [[ -n "$sample_npy" ]]; then
    sample_filter_attr="$(git check-attr filter -- "$sample_npy" | awk '{print $3}')"
fi

echo
echo "Dataset summary:"
echo "  dist_* folders : $dist_count"
echo "  curr_* folders : $curr_count"
echo "  experiments    : $exp_count (IQ_data folders)"
echo "  npy files      : $npy_count"
echo "  png files      : $png_count"
echo "  json files     : $json_count"
echo "  total size     : $total_size"
echo "  LFS filter attr on sample .npy: $sample_filter_attr"

if [[ "$sample_filter_attr" != "lfs" ]]; then
    echo
    echo "WARNING: sample .npy is not marked as LFS by git attributes."
    echo "Check .gitattributes before commit."
fi

echo
echo "Next commands:"
echo "  git add .gitattributes .gitignore \"$DEST_REL\""
echo "  git status --short"
echo "  git lfs ls-files | grep -E \"$DATASET_DIR_NAME|$(basename "$sample_npy")\""
echo "  git commit -m \"data: add $DATASET_DIR_NAME via Git LFS\""
echo "  git push"
