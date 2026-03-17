#!/usr/bin/env python3
"""Batch-ingest research PDFs with Docling.

Reads PDFs from ``knowledge/papers/raw`` and writes parsed artifacts under
``knowledge/papers/parsed/<paper_slug>/``:

- ``document.md``
- ``document.json``
- ``document.txt``
- ``metadata.yaml``

This script is intentionally conservative:
- it preserves the raw PDFs untouched
- it does not try to guess citation metadata
- it writes simple local metadata for downstream notes/retrieval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs with Docling.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("knowledge/papers/raw"),
        help="Directory containing source PDFs (default: knowledge/papers/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("knowledge/papers/parsed"),
        help="Directory for parsed outputs (default: knowledge/papers/parsed).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parsed outputs.",
    )
    return parser.parse_args()


def list_pdfs(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.rglob("*.pdf") if p.is_file())


def safe_write_text(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text(content, encoding="utf-8")


def build_metadata_yaml(*, paper_id: str, pdf_path: Path, out_dir: Path) -> str:
    lines = [
        f"paper_id: {paper_id}",
        f"source_pdf: {pdf_path.as_posix()}",
        f"parsed_dir: {out_dir.as_posix()}",
        f"markdown_path: {(out_dir / 'document.md').as_posix()}",
        f"json_path: {(out_dir / 'document.json').as_posix()}",
        f"text_path: {(out_dir / 'document.txt').as_posix()}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    pdfs = list(list_pdfs(input_dir))
    if not pdfs:
        print(f"No PDFs found under: {input_dir}")
        return 0

    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:
        print(
            "Docling is not available. Install it first, for example:\n"
            "  pip install docling\n\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 3

    converter = DocumentConverter()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdfs)} PDF(s) under {input_dir}")

    for pdf_path in pdfs:
        paper_id = pdf_path.stem
        paper_out_dir = output_dir / paper_id
        paper_out_dir.mkdir(parents=True, exist_ok=True)

        md_path = paper_out_dir / "document.md"
        json_path = paper_out_dir / "document.json"
        txt_path = paper_out_dir / "document.txt"
        meta_path = paper_out_dir / "metadata.yaml"

        if (
            not args.overwrite
            and md_path.exists()
            and json_path.exists()
            and txt_path.exists()
            and meta_path.exists()
        ):
            print(f"Skipping {pdf_path.name}: parsed outputs already exist")
            continue

        print(f"Converting {pdf_path.name} ...")
        result = converter.convert(pdf_path)
        doc = result.document

        markdown = doc.export_to_markdown()
        text = doc.export_to_text()
        doc_json = json.dumps(doc.export_to_dict(), ensure_ascii=False, indent=2)
        metadata_yaml = build_metadata_yaml(
            paper_id=paper_id,
            pdf_path=pdf_path,
            out_dir=paper_out_dir,
        )

        safe_write_text(md_path, markdown, overwrite=args.overwrite)
        safe_write_text(txt_path, text, overwrite=args.overwrite)
        safe_write_text(json_path, doc_json, overwrite=args.overwrite)
        safe_write_text(meta_path, metadata_yaml, overwrite=args.overwrite)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
