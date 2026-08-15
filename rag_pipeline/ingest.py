#!/usr/bin/env python3
"""
Chunks book_analyzer's chapter JSON, embeds each chunk, and upserts it into
the Supabase `chunks` table (see schema.sql).

One chunk per vocabulary word, one per grammar point, one per reading
passage -- never merged. Re-running is safe: rows are upserted on
(chapter_id, chunk_type, content), so identical content won't duplicate.

Usage:
    python ingest.py \
        --chapter-files ../book_analyzer/chapter-01.json \
        --book-structure ../book_analyzer/book-structure.json

Requires SUPABASE_URL, SUPABASE_SERVICE_KEY, and VOYAGE_API_KEY in the
environment (see .env.example).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from chunking import chunk_book_structure, chunk_strict_chapter
from db import get_client, upsert_chunks
from embeddings import embed_texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--chapter-files", nargs="*", type=Path,
        default=[Path("../book_analyzer/chapter-01.json")],
        help="chapter-NN.json files (strict, source-quoted shape)",
    )
    parser.add_argument(
        "--book-structure", type=Path,
        default=None,
        help=(
            "book-structure.json from book_analyzer (opt-in). Its vocabulary definitions are "
            "the model's own gloss, not quoted from the book -- pass this explicitly only if "
            "that's acceptable for your use case."
        ),
    )
    args = parser.parse_args()

    chunks = []
    for p in args.chapter_files:
        if not p.is_file():
            print(f"error: {p} not found", file=sys.stderr)
            return 1
        chapter_chunks = chunk_strict_chapter(p)
        chunks += chapter_chunks
        print(f"{p}: {len(chapter_chunks)} chunks", file=sys.stderr)

    if args.book_structure is None:
        pass
    elif args.book_structure.is_file():
        bs_chunks = chunk_book_structure(args.book_structure)
        chunks += bs_chunks
        print(f"{args.book_structure}: {len(bs_chunks)} chunks", file=sys.stderr)
    else:
        print(f"warning: {args.book_structure} not found, skipping", file=sys.stderr)

    if not chunks:
        print("error: no chunks produced", file=sys.stderr)
        return 1

    print(f"Embedding {len(chunks)} chunks...", file=sys.stderr)
    vectors = embed_texts([c["content"] for c in chunks])

    rows = [
        {
            "chapter_id": c["chapter_id"],
            "chunk_type": c["chunk_type"],
            "content": c["content"],
            "metadata": c["metadata"],
            "embedding": vec,
        }
        for c, vec in zip(chunks, vectors)
    ]

    client = get_client()
    print(f"Upserting {len(rows)} rows into Supabase...", file=sys.stderr)
    upsert_chunks(client, rows)

    counts: dict[str, dict[str, int]] = {}
    for c in chunks:
        counts.setdefault(c["chapter_id"], {}).setdefault(c["chunk_type"], 0)
        counts[c["chapter_id"]][c["chunk_type"]] += 1
    print("Done. Chunk counts:", file=sys.stderr)
    for cid, by_type in counts.items():
        print(f"  {cid}: {by_type}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
