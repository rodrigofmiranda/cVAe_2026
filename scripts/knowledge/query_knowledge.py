#!/usr/bin/env python3
"""Query local Chroma knowledge indexes built from parsed paper Markdown."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CANONICAL_DB = Path("knowledge/index/chroma_db")
DEFAULT_CANONICAL_STATE = Path("knowledge/index/index_state.json")
DEFAULT_IMPORTED_DB = Path("knowledge/index/chroma_db_imported_docling_cache_md")
DEFAULT_IMPORTED_STATE = Path("knowledge/index/index_state_imported_docling_cache_md.json")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class IndexSpec:
    name: str
    db_dir: Path
    state_path: Path


@dataclass(frozen=True)
class SearchHit:
    source: str
    score: float
    paper_id: str
    title: str
    chunk_index: int
    markdown_path: str
    source_pdf: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local knowledge Chroma indexes."
    )
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument(
        "--source",
        choices=["canonical", "imported", "both"],
        default="both",
        help="Which index set to query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Maximum hits to print after merging all selected indexes.",
    )
    parser.add_argument(
        "--per-source-top-k",
        type=int,
        default=8,
        help="Number of hits to fetch from each selected index before merging.",
    )
    parser.add_argument(
        "--paper-pattern",
        default=None,
        help="Optional case-insensitive regex filter over paper_id/title/path.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Maximum preview length per hit.",
    )
    parser.add_argument(
        "--dedupe-paper",
        action="store_true",
        help="Keep only the best hit per paper after merging results.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print Markdown/PDF paths for each hit.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Do not truncate chunk previews.",
    )
    parser.add_argument(
        "--canonical-db-dir",
        type=Path,
        default=DEFAULT_CANONICAL_DB,
        help="Canonical Chroma DB directory.",
    )
    parser.add_argument(
        "--canonical-state-path",
        type=Path,
        default=DEFAULT_CANONICAL_STATE,
        help="Canonical index state JSON.",
    )
    parser.add_argument(
        "--imported-db-dir",
        type=Path,
        default=DEFAULT_IMPORTED_DB,
        help="Imported-corpus Chroma DB directory.",
    )
    parser.add_argument(
        "--imported-state-path",
        type=Path,
        default=DEFAULT_IMPORTED_STATE,
        help="Imported-corpus index state JSON.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_MODEL,
        help="Embedding model name. Must match the one used to build the DB.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device for query-time retrieval (default: cpu).",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow model resolution from the network if it is not already cached locally.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    return parser.parse_args()


def ensure_dependencies() -> tuple[Any, Any]:
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception as exc:
        print(
            "Missing retrieval dependencies. Activate the knowledge environment or install:\n"
            "  pip install langchain-huggingface langchain-chroma chromadb sentence-transformers\n\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return Chroma, HuggingFaceEmbeddings


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Index state not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def selected_indexes(args: argparse.Namespace) -> list[IndexSpec]:
    specs: list[IndexSpec] = []
    if args.source in {"canonical", "both"}:
        specs.append(
            IndexSpec(
                name="canonical",
                db_dir=args.canonical_db_dir.resolve(),
                state_path=args.canonical_state_path.resolve(),
            )
        )
    if args.source in {"imported", "both"}:
        specs.append(
            IndexSpec(
                name="imported",
                db_dir=args.imported_db_dir.resolve(),
                state_path=args.imported_state_path.resolve(),
            )
        )
    return specs


def build_vectordb(spec: IndexSpec, args: argparse.Namespace, Chroma: Any, HuggingFaceEmbeddings: Any) -> tuple[Any, dict[str, Any]]:
    if not spec.db_dir.exists():
        raise FileNotFoundError(f"Index DB not found: {spec.db_dir}")
    state = load_state(spec.state_path)
    config = dict(state.get("config", {}))
    collection_name = str(config.get("collection_name", "")).strip()
    if not collection_name:
        raise ValueError(f"collection_name missing from state: {spec.state_path}")
    embed_model = str(config.get("embed_model", args.embed_model)).strip() or args.embed_model
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={
            "device": args.device,
            "local_files_only": not args.allow_network,
        },
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=str(spec.db_dir),
        embedding_function=embeddings,
    )
    return vectordb, state


def preview_text(text: str, max_chars: int, show_all: bool) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if show_all or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 1)].rstrip() + "…"


def hit_matches_pattern(hit: SearchHit, pattern: re.Pattern[str] | None) -> bool:
    if pattern is None:
        return True
    haystack = " | ".join(
        [
            hit.paper_id,
            hit.title,
            hit.markdown_path,
            hit.source_pdf,
        ]
    )
    return bool(pattern.search(haystack))


def query_index(
    spec: IndexSpec,
    args: argparse.Namespace,
    Chroma: Any,
    HuggingFaceEmbeddings: Any,
) -> list[SearchHit]:
    vectordb, state = build_vectordb(spec, args, Chroma, HuggingFaceEmbeddings)
    papers_state = dict(state.get("papers", {}))
    pairs = vectordb.similarity_search_with_score(args.query, k=max(1, args.per_source_top_k))
    hits: list[SearchHit] = []
    for document, score in pairs:
        metadata = dict(document.metadata or {})
        paper_id = str(metadata.get("paper_id", "")).strip() or "unknown"
        paper_state = dict(papers_state.get(paper_id, {}))
        title = str(metadata.get("title", "")).strip() or paper_id
        hits.append(
            SearchHit(
                source=spec.name,
                score=float(score),
                paper_id=paper_id,
                title=title,
                chunk_index=int(metadata.get("chunk_index", -1)),
                markdown_path=str(metadata.get("markdown_path") or paper_state.get("markdown_path") or ""),
                source_pdf=str(metadata.get("source_pdf") or paper_state.get("source_pdf") or ""),
                text=str(document.page_content or ""),
            )
        )
    return hits


def dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
    best: dict[str, SearchHit] = {}
    for hit in hits:
        previous = best.get(hit.paper_id)
        if previous is None or hit.score < previous.score:
            best[hit.paper_id] = hit
    return sorted(best.values(), key=lambda item: item.score)


def render_text(hits: list[SearchHit], args: argparse.Namespace) -> str:
    if not hits:
        return "No matching chunks found."
    lines = [
        f"Query: {args.query}",
        f"Hits: {len(hits)}",
        "",
    ]
    for idx, hit in enumerate(hits, start=1):
        lines.append(
            f"[{idx}] source={hit.source} score={hit.score:.4f} "
            f"paper_id={hit.paper_id} chunk={hit.chunk_index}"
        )
        lines.append(f"    title: {hit.title}")
        if args.show_paths:
            if hit.markdown_path:
                lines.append(f"    markdown: {hit.markdown_path}")
            if hit.source_pdf:
                lines.append(f"    source_pdf: {hit.source_pdf}")
        lines.append(f"    text: {preview_text(hit.text, args.max_chars, args.show_all)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    Chroma, HuggingFaceEmbeddings = ensure_dependencies()
    pattern = re.compile(args.paper_pattern, flags=re.IGNORECASE) if args.paper_pattern else None

    hits: list[SearchHit] = []
    missing_indexes: list[str] = []
    for spec in selected_indexes(args):
        try:
            hits.extend(query_index(spec, args, Chroma, HuggingFaceEmbeddings))
        except FileNotFoundError:
            missing_indexes.append(spec.name)

    if not hits and missing_indexes:
        print(
            "Selected index(es) not available: " + ", ".join(missing_indexes),
            file=sys.stderr,
        )
        return 1

    hits = [hit for hit in hits if hit_matches_pattern(hit, pattern)]
    hits.sort(key=lambda item: item.score)
    if args.dedupe_paper:
        hits = dedupe_hits(hits)
    hits = hits[: max(1, args.top_k)]

    if args.json:
        payload = [
            {
                "source": hit.source,
                "score": hit.score,
                "paper_id": hit.paper_id,
                "title": hit.title,
                "chunk_index": hit.chunk_index,
                "markdown_path": hit.markdown_path,
                "source_pdf": hit.source_pdf,
                "text": hit.text if args.show_all else preview_text(hit.text, args.max_chars, False),
            }
            for hit in hits
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_text(hits, args), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
