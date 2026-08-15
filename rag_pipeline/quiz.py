#!/usr/bin/env python3
"""
generate_quiz(chapter_ids, topic_type, num_questions) -> dict

Retrieval is a plain metadata filter (chapter_id + chunk_type), not a
similarity search -- there's no free-text query to embed here, we just want
"every vocabulary/grammar chunk belonging to these chapters", and a filter
gives exact recall where a nearest-neighbor search could miss or pull in
unrelated chunks. The embeddings stored by ingest.py exist so a future
"ask the book a question" feature can do real semantic search; quizzes
don't need it.

There is no LLM call anywhere in this module. Questions are built by
template from the exact word/sentence/rule text already stored in each
chunk's metadata, so there is no way for a question (or its answer) to
contain anything not already in the source chunk. That's a stronger
guarantee than instructing an LLM not to invent content.

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) in the
environment.
"""

from __future__ import annotations

import random
import re

from db import fetch_chunks, get_available_chapter_ids, get_client

VALID_TOPIC_TYPES = ("vocabulary", "grammar", "mixed")

BE_CONTRACTIONS = ["isn't", "aren't", "am not", "'re", "'m", "'s"]
BE_BARE = ["is", "are", "am"]


def _find_span(sentence: str, needle: str, whole_word: bool) -> re.Match | None:
    if whole_word:
        return re.search(rf"\b{re.escape(needle)}\b", sentence, re.IGNORECASE)
    return re.search(re.escape(needle), sentence)


def _find_vocab_span(sentence: str, word: str) -> re.Match | None:
    return _find_span(sentence, word, whole_word=True)


def _find_be_span(sentence: str) -> re.Match | None:
    for form in BE_CONTRACTIONS:
        m = _find_span(sentence, form, whole_word=False)
        if m:
            return m
    for form in BE_BARE:
        m = _find_span(sentence, form, whole_word=True)
        if m:
            return m
    return None


def _blank(sentence: str, match: re.Match) -> str:
    return sentence[: match.start()] + "_____" + sentence[match.end() :]


def _make_vocab_fill_in_blank(chunk: dict, qid: str) -> dict | None:
    word = chunk["metadata"]["word"]
    for ex in chunk["metadata"].get("example_sentences") or []:
        m = _find_vocab_span(ex, word)
        if m:
            return {
                "id": qid,
                "type": "fill_in_blank",
                "topic": "vocabulary",
                "chapter_id": chunk["chapter_id"],
                "prompt": f'Fill in the blank (book sentence, Chapter {chunk["chapter_id"]}): "{_blank(ex, m)}"',
                "choices": [],
                "answer": word,
            }
    return None


def _make_vocab_short_answer(chunk: dict, qid: str) -> dict | None:
    word = chunk["metadata"]["word"]
    examples = chunk["metadata"].get("example_sentences") or []
    if not examples:
        return None
    return {
        "id": qid,
        "type": "short_answer",
        "topic": "vocabulary",
        "chapter_id": chunk["chapter_id"],
        "prompt": (
            f'Short answer: Chapter {chunk["chapter_id"]} uses the word "{word}" in one of its '
            "example sentences. Write that sentence."
        ),
        "choices": [],
        "answer": examples[0],
    }


def _make_grammar_fill_in_blank(chunk: dict, qid: str) -> dict | None:
    for ex in chunk["metadata"].get("example_sentences") or []:
        m = _find_be_span(ex)
        if m:
            return {
                "id": qid,
                "type": "fill_in_blank",
                "topic": "grammar",
                "chapter_id": chunk["chapter_id"],
                "prompt": (
                    f'Fill in the blank with the correct form of \'be\' '
                    f'(book sentence, Chapter {chunk["chapter_id"]}): "{_blank(ex, m)}"'
                ),
                "choices": [],
                "answer": ex[m.start() : m.end()],
            }
    return None


def _make_grammar_short_answer(chunk: dict, qid: str) -> dict | None:
    examples = chunk["metadata"].get("example_sentences") or []
    if not examples:
        return None
    return {
        "id": qid,
        "type": "short_answer",
        "topic": "grammar",
        "chapter_id": chunk["chapter_id"],
        "prompt": (
            f'Short answer: What grammar point from Chapter {chunk["chapter_id"]} does this book '
            f'sentence practice — "{examples[0]}"?'
        ),
        "choices": [],
        "answer": chunk["metadata"]["topic"],
    }


def _pair_for_matching(chunk: dict, used_sentences: set[str]) -> tuple[str, str] | None:
    examples = chunk["metadata"].get("example_sentences") or []
    label = chunk["metadata"]["word"] if chunk["chunk_type"] == "vocabulary" else chunk["metadata"]["topic"]
    for ex in examples:
        if ex not in used_sentences:
            return label, ex
    return None


def _make_matching_question(chunks_subset: list[dict], qid: str, rng: random.Random) -> dict | None:
    used_sentences: set[str] = set()
    pairs = []
    for c in chunks_subset:
        p = _pair_for_matching(c, used_sentences)
        if p:
            pairs.append(p)
            used_sentences.add(p[1])
    if len(pairs) < 3:
        return None
    left = [p[0] for p in pairs]
    right = [p[1] for p in pairs]
    shuffled_right = right[:]
    rng.shuffle(shuffled_right)
    chapter_ids = sorted({c["chapter_id"] for c in chunks_subset})
    return {
        "id": qid,
        "type": "matching",
        "topic": "mixed" if len({c["chunk_type"] for c in chunks_subset}) > 1 else chunks_subset[0]["chunk_type"],
        "chapter_id": ", ".join(chapter_ids),
        "prompt": (
            "Match each item to its example sentence from the book:\n"
            + "\n".join(f"{i + 1}. {item}" for i, item in enumerate(left))
        ),
        "choices": shuffled_right,
        "answer": "; ".join(f"{l} -> {r}" for l, r in pairs),
    }


def _build_questions(chunks: list[dict], num_questions: int, rng: random.Random) -> list[dict]:
    chunks = chunks[:]
    rng.shuffle(chunks)
    vocab_chunks = [c for c in chunks if c["chunk_type"] == "vocabulary"]
    grammar_chunks = [c for c in chunks if c["chunk_type"] == "grammar"]

    questions: list[dict] = []
    qid = 1

    if num_questions >= 3:
        pool = vocab_chunks if len(vocab_chunks) >= 3 else grammar_chunks
        if len(pool) >= 3:
            subset = rng.sample(pool, min(4, len(pool)))
            mq = _make_matching_question(subset, str(qid), rng)
            if mq:
                questions.append(mq)
                qid += 1

    all_chunks = vocab_chunks + grammar_chunks
    rng.shuffle(all_chunks)
    idx = 0
    attempts = 0
    max_attempts = max(len(all_chunks) * 3, 1)
    while len(questions) < num_questions and all_chunks and attempts < max_attempts:
        chunk = all_chunks[idx % len(all_chunks)]
        idx += 1
        attempts += 1
        generators = (
            [_make_vocab_fill_in_blank, _make_vocab_short_answer]
            if chunk["chunk_type"] == "vocabulary"
            else [_make_grammar_fill_in_blank, _make_grammar_short_answer]
        )
        gen = rng.choice(generators)
        q = gen(chunk, str(qid))
        if q:
            questions.append(q)
            qid += 1

    return questions[:num_questions]


def generate_quiz(
    chapter_ids: list[str],
    topic_type: str,
    num_questions: int,
    seed: int | None = None,
) -> dict:
    if topic_type not in VALID_TOPIC_TYPES:
        raise ValueError(f"topic_type must be one of {VALID_TOPIC_TYPES}, got {topic_type!r}")
    if num_questions < 1:
        raise ValueError("num_questions must be >= 1")

    client = get_client()
    available = get_available_chapter_ids(client)
    missing = [c for c in chapter_ids if c not in available]
    if missing:
        return {
            "status": "not_available",
            "message": (
                f"not available — chapter(s) {', '.join(missing)} haven't been added to the "
                "knowledge base yet."
            ),
            "requested_chapter_ids": chapter_ids,
            "available_chapter_ids": sorted(available),
        }

    chunk_types = ["vocabulary", "grammar"] if topic_type == "mixed" else [topic_type]
    chunks = fetch_chunks(client, chapter_ids, chunk_types)
    if not chunks:
        return {
            "status": "not_available",
            "message": (
                f"not available — no {topic_type} content found for chapter(s) "
                f"{', '.join(chapter_ids)} in the knowledge base."
            ),
            "requested_chapter_ids": chapter_ids,
            "available_chapter_ids": sorted(available),
        }

    rng = random.Random(seed)
    questions = _build_questions(chunks, num_questions, rng)

    return {
        "status": "ok",
        "chapter_ids": chapter_ids,
        "topic_type": topic_type,
        "num_requested": num_questions,
        "num_generated": len(questions),
        "questions": questions,
    }
