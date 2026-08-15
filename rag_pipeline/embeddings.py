#!/usr/bin/env python3
"""
Embedding backend for the RAG pipeline.

Uses Voyage AI (Anthropic's recommended embeddings partner) since the rest
of this repo already standardizes on the Anthropic ecosystem. voyage-3-lite
produces 512-dim vectors -- matches the `vector(512)` column in schema.sql.
If you swap models/providers, update schema.sql's dimension to match.

Requires VOYAGE_API_KEY in the environment.
"""

from __future__ import annotations

import os

MODEL = "voyage-3-lite"


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY must be set in the environment to embed chunks.")

    client = voyageai.Client(api_key=api_key)
    result = client.embed(texts, model=MODEL, input_type=input_type)
    return result.embeddings
