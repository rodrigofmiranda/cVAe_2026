#!/usr/bin/env python3
"""Index parsed research papers into a local Chroma vector database.

This stage reads the canonical Docling outputs under:

  knowledge/papers/parsed/<paper_id>/document.md

and builds a retrieval layer under:

  knowledge/index/chroma_db/
  knowledge/index/index_state.json

The script is incremental and keyed by the Markdown content hash, so it does
not need to re-embed unchanged papers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ParsedPaper:
    paper_id: str
    paper_dir: Path
    markdown_path: Path
    metadata_path: Path | None
    json_path: Path | None
    source_pdf: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index parsed paper Markdown into a local Chroma database."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("knowledge/papers/parsed"),
        help="Directory containing Docling parsed outputs.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("knowledge/index/chroma_db"),
        help="Directory for the Chroma database.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("knowledge/index/index_state.json"),
        help="Path for incremental indexing state.",
    )
    parser.add_argument(
        "--collection-name",
        default="knowledge_papers",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_MODEL,
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for Markdown splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for Markdown splitting.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=256,
        help="Batch size for Chroma insertions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N parsed papers.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing DB/state and rebuild everything.",
    )
    return parser.parse_args()


def fmt_s(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds / 60:.2f}min"
    return f"{seconds / 3600:.2f}h"


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def chunk_id(paper_id: str, chunk_index: int) -> str:
    return f"{paper_id}::chunk::{chunk_index:05d}"


def read_simple_metadata(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def iter_parsed_papers(parsed_dir: Path) -> list[ParsedPaper]:
    papers: list[ParsedPaper] = []
    for child in sorted(parsed_dir.iterdir()):
        if not child.is_dir():
            continue
        markdown_path = child / "document.md"
        if not markdown_path.exists():
            continue
        metadata_path = child / "metadata.yaml"
        json_path = child / "document.json"
        metadata = read_simple_metadata(metadata_path)
        papers.append(
            ParsedPaper(
                paper_id=child.name,
                paper_dir=child,
                markdown_path=markdown_path,
                metadata_path=metadata_path if metadata_path.exists() else None,
                json_path=json_path if json_path.exists() else None,
                source_pdf=metadata.get("source_pdf"),
            )
        )
    return papers


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"version": 1, "config": {}, "papers": {}}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def state_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "embed_model": args.embed_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "collection_name": args.collection_name,
    }


def ensure_dependencies() -> tuple[Any, Any, Any]:
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:
        print(
            "Missing retrieval dependencies. Install them in the indexing environment, for example:\n"
            "  pip install langchain-huggingface langchain-chroma "
            "langchain-text-splitters chromadb sentence-transformers\n\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return Chroma, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter


def delete_previous_chunks(vectordb: Any, paper_id: str, chunk_count: int) -> None:
    if chunk_count <= 0:
        return
    ids = [chunk_id(paper_id, idx) for idx in range(chunk_count)]
    vectordb.delete(ids=ids)


def main() -> int:
    args = parse_args()
    parsed_dir = args.parsed_dir.resolve()
    db_dir = args.db_dir.resolve()
    state_path = args.state_path.resolve()

    if not parsed_dir.exists():
        print(f"Parsed directory not found: {parsed_dir}", file=sys.stderr)
        return 1

    if args.rebuild:
        shutil.rmtree(db_dir, ignore_errors=True)
        if state_path.exists():
            state_path.unlink()

    Chroma, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter = ensure_dependencies()

    papers = iter_parsed_papers(parsed_dir)
    if args.limit is not None:
        papers = papers[: args.limit]

    if not papers:
        print(f"No parsed papers found under: {parsed_dir}")
        return 0

    state = load_state(state_path)
    current_config = state_config(args)
    previous_config = state.get("config", {})
    if previous_config and previous_config != current_config and not args.rebuild:
        print(
            "Index configuration changed since the last run.\n"
            "Rerun with --rebuild to avoid mixing incompatible chunking/embedding settings.",
            file=sys.stderr,
        )
        return 3

    state["config"] = current_config
    state.setdefault("papers", {})

    print("=== Knowledge Indexer ===")
    print(f"Parsed dir:       {parsed_dir}")
    print(f"DB dir:           {db_dir}")
    print(f"State path:       {state_path}")
    print(f"Collection:       {args.collection_name}")
    print(f"Embedding model:  {args.embed_model}")
    print(f"Chunking:         size={args.chunk_size} overlap={args.chunk_overlap}")
    print(f"Parsed papers:    {len(papers)}")

    t0 = time.perf_counter()

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embed_model,
        encode_kwargs={
            "batch_size": args.embed_batch_size,
            "normalize_embeddings": True,
        },
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_start_index=True,
    )
    vectordb = Chroma(
        collection_name=args.collection_name,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )

    indexed = 0
    skipped = 0
    updated = 0
    total_chunks = 0

    for index, paper in enumerate(papers, start=1):
        t_paper0 = time.perf_counter()
        markdown = paper.markdown_path.read_text(encoding="utf-8", errors="ignore")
        content_hash = sha1_text(markdown)
        previous = state["papers"].get(paper.paper_id)

        if previous and previous.get("content_hash") == content_hash:
            skipped += 1
            print(f"[{index}/{len(papers)}] skip {paper.paper_id} (unchanged)")
            continue

        if previous:
            delete_previous_chunks(vectordb, paper.paper_id, int(previous.get("chunk_count", 0)))
            updated += 1

        base_metadata = {
            "paper_id": paper.paper_id,
            "title": paper.paper_id,
            "source_pdf": paper.source_pdf or "",
            "paper_dir": str(paper.paper_dir),
            "markdown_path": str(paper.markdown_path),
            "json_path": str(paper.json_path) if paper.json_path else "",
            "content_hash": content_hash,
        }
        documents = splitter.create_documents([markdown], metadatas=[base_metadata])

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        for chunk_index, document in enumerate(documents):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = chunk_index
            texts.append(document.page_content)
            metadatas.append(metadata)
            ids.append(chunk_id(paper.paper_id, chunk_index))

        for batch_start in range(0, len(texts), args.add_batch_size):
            batch_end = batch_start + args.add_batch_size
            vectordb.add_texts(
                texts=texts[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
                ids=ids[batch_start:batch_end],
            )

        state["papers"][paper.paper_id] = {
            "paper_dir": str(paper.paper_dir),
            "markdown_path": str(paper.markdown_path),
            "source_pdf": paper.source_pdf or "",
            "content_hash": content_hash,
            "chunk_count": len(texts),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_state(state_path, state)

        indexed += 1
        total_chunks += len(texts)
        print(
            f"[{index}/{len(papers)}] indexed {paper.paper_id} | "
            f"chunks={len(texts)} | time={fmt_s(time.perf_counter() - t_paper0)}"
        )

    if hasattr(vectordb, "persist"):
        try:
            vectordb.persist()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    print("\n=== Summary ===")
    print(f"Indexed now:   {indexed}")
    print(f"Updated:       {updated}")
    print(f"Skipped:       {skipped}")
    print(f"Chunks added:  {total_chunks}")
    print(f"Total time:    {fmt_s(elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
