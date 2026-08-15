#!/usr/bin/env python3
"""
Turns book_analyzer's chapter JSON into embeddable chunks.

Two source shapes are supported:
- chapter-NN.json (e.g. chapter-01.json): the strict, source-quoted shape
  with word/definition/example_sentences/page, topic/explanation/
  example_sentences/page, and reading_text passages.
- book-structure.json: book-analyzer's LLM-extracted shape, keyed
  differently (term/definition/part_of_speech, name/explanation) and
  missing example_sentences, page numbers, and reading passages entirely.

One chunk = one vocabulary word, one grammar point, or one reading passage.
Nothing here invents content to fill gaps in either shape -- a field that
isn't in the source JSON is simply left out of the chunk.
"""

from __future__ import annotations

import json
from pathlib import Path


def _vocab_chunk(chapter_id, chapter_title, source_file, word, definition, example_sentences, page):
    lines = [f"Word: {word}"]
    if definition:
        lines.append(f"Definition: {definition}")
    if example_sentences:
        quoted = "; ".join(f'"{s}"' for s in example_sentences)
        lines.append(f"Example sentences: {quoted}")
    return {
        "chapter_id": chapter_id,
        "chunk_type": "vocabulary",
        "content": "\n".join(lines),
        "metadata": {
            "word": word,
            "definition": definition or "",
            "example_sentences": example_sentences or [],
            "page": page,
            "chapter_title": chapter_title,
            "source_file": source_file,
        },
    }


def _grammar_chunk(chapter_id, chapter_title, source_file, topic, explanation, example_sentences, page):
    lines = [f"Grammar point: {topic}"]
    if explanation:
        lines.append(f"Explanation: {explanation}")
    if example_sentences:
        quoted = "; ".join(f'"{s}"' for s in example_sentences)
        lines.append(f"Example sentences: {quoted}")
    return {
        "chapter_id": chapter_id,
        "chunk_type": "grammar",
        "content": "\n".join(lines),
        "metadata": {
            "topic": topic,
            "explanation": explanation or "",
            "example_sentences": example_sentences or [],
            "page": page,
            "chapter_title": chapter_title,
            "source_file": source_file,
        },
    }


def _reading_chunk(chapter_id, chapter_title, source_file, title, full_text, page):
    return {
        "chapter_id": chapter_id,
        "chunk_type": "reading",
        "content": f"Reading passage: {title}\n{full_text}",
        "metadata": {
            "title": title,
            "full_text": full_text,
            "page": page,
            "chapter_title": chapter_title,
            "source_file": source_file,
        },
    }


def chunk_strict_chapter(path: Path) -> list[dict]:
    """Chunk a chapter-NN.json file. chapter_id is taken from the filename stem."""
    data = json.loads(path.read_text(encoding="utf-8"))
    chapter_id = path.stem
    chapter_title = data.get("chapter_title", "")
    source_file = path.name

    chunks = []
    for v in data.get("vocabulary", []):
        chunks.append(_vocab_chunk(
            chapter_id, chapter_title, source_file,
            v.get("word", ""), v.get("definition", ""),
            v.get("example_sentences") or [], v.get("page"),
        ))
    for g in data.get("grammar_points", []):
        chunks.append(_grammar_chunk(
            chapter_id, chapter_title, source_file,
            g.get("topic", ""), g.get("explanation", ""),
            g.get("example_sentences") or [], g.get("page"),
        ))
    for r in data.get("reading_text", []):
        chunks.append(_reading_chunk(
            chapter_id, chapter_title, source_file,
            r.get("title", ""), r.get("full_text", ""), r.get("page"),
        ))
    return chunks


def chunk_book_structure(path: Path) -> list[dict]:
    """Chunk book-structure.json. Each chapter's own chapter_id field is used as-is."""
    data = json.loads(path.read_text(encoding="utf-8"))
    source_file = path.name

    chunks = []
    for chapter in data.get("chapters", []):
        chapter_id = chapter.get("chapter_id", "")
        chapter_title = chapter.get("chapter_title", "")
        for v in chapter.get("vocabulary", []):
            chunks.append(_vocab_chunk(
                chapter_id, chapter_title, source_file,
                v.get("term", ""), v.get("definition", ""),
                [], None,
            ))
        for g in chapter.get("grammar_points", []):
            chunks.append(_grammar_chunk(
                chapter_id, chapter_title, source_file,
                g.get("name", ""), g.get("explanation", ""),
                [], None,
            ))
    return chunks
