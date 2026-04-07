#!/usr/bin/env python3
"""Normalize a flat Markdown cache into the local parsed-paper layout.

This is meant for already-extracted corpora such as Docling-generated `.md`
archives. Each source Markdown becomes:

  <output_dir>/<paper_id>/document.md
  <output_dir>/<paper_id>/document.txt
  <output_dir>/<paper_id>/metadata.yaml

The output is intentionally compatible with `index_knowledge_chroma.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a flat Markdown cache into the parsed-paper layout."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source .md files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where normalized parsed outputs will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized outputs.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    return slug or "paper"


def list_markdown_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*.md") if p.is_file())


def split_title_and_suffix(stem: str) -> tuple[str, str]:
    if "__" in stem:
        title, suffix = stem.rsplit("__", 1)
        suffix = suffix.strip()
        if suffix:
            return title.strip(), suffix
    digest = hashlib.sha1(stem.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return stem.strip(), digest


def build_paper_id(title: str, suffix: str) -> str:
    slug = slugify(title)
    if len(slug) > 160:
        slug = slug[:160].rstrip("_")
    return f"{slug}__{suffix.lower()}"


def build_metadata_yaml(*, paper_id: str, title_hint: str, source_md: Path, out_dir: Path) -> str:
    lines = [
        f"paper_id: {paper_id}",
        f"title_hint: {title_hint}",
        f"source_markdown: {source_md.as_posix()}",
        "source_type: imported_markdown_cache",
        f"parsed_dir: {out_dir.as_posix()}",
        f"markdown_path: {(out_dir / 'document.md').as_posix()}",
        f"text_path: {(out_dir / 'document.txt').as_posix()}",
    ]
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    source_files = list_markdown_files(input_dir)
    if not source_files:
        print(f"No Markdown files found under: {input_dir}")
        return 0

    print("=== Markdown Cache Import ===")
    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Markdowns:   {len(source_files)}")

    imported = 0
    skipped = 0

    for index, source_md in enumerate(source_files, start=1):
        title_hint, suffix = split_title_and_suffix(source_md.stem)
        paper_id = build_paper_id(title_hint, suffix)
        paper_dir = output_dir / paper_id
        md_path = paper_dir / "document.md"
        txt_path = paper_dir / "document.txt"
        meta_path = paper_dir / "metadata.yaml"

        if md_path.exists() and meta_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(source_files)}] skip {paper_id}")
            continue

        content = source_md.read_text(encoding="utf-8", errors="ignore")
        metadata_yaml = build_metadata_yaml(
            paper_id=paper_id,
            title_hint=title_hint,
            source_md=source_md,
            out_dir=paper_dir,
        )

        write_text(md_path, content)
        write_text(txt_path, content)
        write_text(meta_path, metadata_yaml)
        imported += 1
        print(f"[{index}/{len(source_files)}] import {paper_id}")

    print("\n=== Summary ===")
    print(f"Imported: {imported}")
    print(f"Skipped:  {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
