#!/usr/bin/env python3
"""
Supabase/pgvector access for the RAG pipeline.

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) in the
environment. Use a service-role key for ingest.py (it writes); a lower-
privilege key is fine for quiz.py's reads if you set up RLS for that later.
"""

from __future__ import annotations

import os

TABLE = "chunks"



def get_client():
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) must be set in the environment."
        )
    return create_client(url, key)


def upsert_chunks(client, rows: list[dict]) -> None:
    client.table(TABLE).upsert(rows, on_conflict="chapter_id,chunk_type,content").execute()


def get_available_chapter_ids(client) -> set[str]:
    resp = client.table(TABLE).select("chapter_id").execute()
    return {row["chapter_id"] for row in resp.data}


def fetch_chunks(client, chapter_ids: list[str], chunk_types: list[str]) -> list[dict]:
    resp = (
        client.table(TABLE)
        .select("chapter_id,chunk_type,content,metadata")
        .in_("chapter_id", chapter_ids)
        .in_("chunk_type", chunk_types)
        .execute()
    )
    return resp.data


def _parse_embedding(value) -> list[float]:
    """PostgREST serializes pgvector columns as their text form, e.g. "[0.1,0.2]"."""
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(x) for x in value.strip("[]").split(",")]


def fetch_chunks_with_embeddings(client, chapter_ids: list[str]) -> list[dict]:
    """Every chunk (any chunk_type) for the given chapters, embedding included -- for similarity search."""
    resp = (
        client.table(TABLE)
        .select("chapter_id,chunk_type,content,metadata,embedding")
        .in_("chapter_id", chapter_ids)
        .execute()
    )
    rows = resp.data
    for row in rows:
        row["embedding"] = _parse_embedding(row["embedding"])
    return rows
