#!/usr/bin/env python3
"""
ask(question) -> dict

This is the semantic-search use of the `embedding` column that quiz.py's
docstring points at ("a future 'ask the book a question' feature can do real
semantic search"). Unlike generate_quiz's exact chapter/chunk_type filter,
there's no metadata that can answer a free-text question -- so this embeds
the question and ranks every chunk belonging to a chapter that's actually in
the knowledge base by cosine similarity against it.

If the best match doesn't clear SIMILARITY_THRESHOLD -- or the knowledge
base is empty -- ask() refuses rather than guessing.

The answer is built by directly quoting the matched chunk(s)' stored
`content`, never summarized or rephrased by an LLM. That's the same
guarantee quiz.py relies on: there's no way for the answer to contain
anything not already in the source chunk.

Requires SUPABASE_URL, SUPABASE_SERVICE_KEY (or SUPABASE_KEY), and
VOYAGE_API_KEY in the environment.
"""

from __future__ import annotations

import math

from db import fetch_chunks_with_embeddings, get_available_chapter_ids, get_client
from embeddings import embed_texts

TOP_K = 3
SIMILARITY_THRESHOLD = 0.4  # cosine similarity; see test_ask.py for observed scores on real questions

NOT_COVERED_MESSAGE = "I don't know — this isn't covered in the material I have."


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _sources(scored: list[tuple[float, dict]]) -> list[dict]:
    return [
        {"chapter_id": c["chapter_id"], "chunk_type": c["chunk_type"], "similarity": round(sim, 4)}
        for sim, c in scored
    ]


def _compose_answer(matches: list[tuple[float, dict]]) -> str:
    parts = [f"From Chapter {c['chapter_id']} ({c['chunk_type']}):\n{c['content']}" for _, c in matches]
    return "\n\n".join(parts)


def ask(question: str, top_k: int = TOP_K, similarity_threshold: float = SIMILARITY_THRESHOLD) -> dict:
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    client = get_client()
    available = get_available_chapter_ids(client)
    if not available:
        return {"status": "not_found", "question": question, "answer": NOT_COVERED_MESSAGE, "sources": []}

    chunks = fetch_chunks_with_embeddings(client, sorted(available))
    if not chunks:
        return {"status": "not_found", "question": question, "answer": NOT_COVERED_MESSAGE, "sources": []}

    query_embedding = embed_texts([question], input_type="query")[0]

    scored = sorted(
        ((_cosine_similarity(query_embedding, c["embedding"]), c) for c in chunks),
        key=lambda pair: pair[0],
        reverse=True,
    )
    top = scored[:top_k]
    matches = [(sim, c) for sim, c in top if sim >= similarity_threshold]

    if not matches:
        return {
            "status": "not_found",
            "question": question,
            "answer": NOT_COVERED_MESSAGE,
            "sources": _sources(top),
        }

    return {
        "status": "ok",
        "question": question,
        "answer": _compose_answer(matches),
        "sources": _sources(matches),
    }
