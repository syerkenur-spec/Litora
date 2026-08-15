#!/usr/bin/env python3
"""
book-analyzer agent (see ../book-analyzer.md for the spec this implements).

Reads a coursebook one chapter-file at a time, asks Claude to extract
vocabulary, grammar points, and a CEFR difficulty estimate per chapter, and
writes the results as one consistently-shaped book-structure.json plus a
one-line log of any chapters flagged as unclear.

Usage:
    python analyze_book.py <chapters_dir> [-o book-structure.json] [--log flagged-chapters.log]

<chapters_dir> must contain one .txt or .md file per chapter. Files are
processed in filename-sorted order, so name them so that order matches the
book (e.g. 01-intro.md, 02-greetings.md, ...).

Requires ANTHROPIC_API_KEY in the environment (or an `ant auth login` profile).
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You break down a single chapter of a language coursebook into structured data the rest of a coursebook app can use.

Rules:
- Never invent content that isn't in the chapter text. If the chapter is unclear, too sparse, or ambiguous, set "unclear" to true and explain why in "unclear_reason" instead of guessing.
- Extract only vocabulary and grammar points actually introduced in this chapter — don't pull in content from other chapters you might infer exist.
- Estimate difficulty using CEFR levels (A1-C2) where possible. If you can't judge confidently, use "Unknown" for cefr and "low" for confidence.
- Do not generate test questions, comprehension summaries, or prose commentary — only the structured fields requested."""

CHAPTER_SCHEMA = {
    "type": "object",
    "properties": {
        "chapter_title": {"type": "string"},
        "vocabulary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                    "part_of_speech": {"type": "string"},
                },
                "required": ["term", "definition", "part_of_speech"],
                "additionalProperties": False,
            },
        },
        "grammar_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["name", "explanation"],
                "additionalProperties": False,
            },
        },
        "difficulty": {
            "type": "object",
            "properties": {
                "cefr": {
                    "type": "string",
                    "enum": ["A1", "A2", "B1", "B2", "C1", "C2", "Unknown"],
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["cefr", "confidence"],
            "additionalProperties": False,
        },
        "unclear": {"type": "boolean"},
        "unclear_reason": {"type": "string"},
    },
    "required": [
        "chapter_title",
        "vocabulary",
        "grammar_points",
        "difficulty",
        "unclear",
        "unclear_reason",
    ],
    "additionalProperties": False,
}


def analyze_chapter(client: anthropic.Anthropic, chapter_id: str, text: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": CHAPTER_SCHEMA}},
        messages=[
            {"role": "user", "content": f"Chapter file: {chapter_id}\n\n{text}"}
        ],
    )
    text_block = next(b for b in response.content if b.type == "text")
    result = json.loads(text_block.text)
    result["chapter_id"] = chapter_id
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Directory with one .txt/.md file per chapter")
    parser.add_argument("-o", "--output", type=Path, default=Path("book-structure.json"))
    parser.add_argument("--log", type=Path, default=Path("flagged-chapters.log"))
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"error: {args.input_dir} is not a directory", file=sys.stderr)
        return 1

    chapter_files = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in (".txt", ".md")
    )
    if not chapter_files:
        print(f"error: no .txt/.md files found in {args.input_dir}", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    chapters = []
    flagged = []

    for path in chapter_files:
        chapter_id = path.stem
        text = path.read_text(encoding="utf-8")
        print(f"Analyzing {chapter_id}...", file=sys.stderr)
        result = analyze_chapter(client, chapter_id, text)
        chapters.append(result)
        if result["unclear"]:
            flagged.append(f"{chapter_id}: {result['unclear_reason']}")

    args.output.write_text(json.dumps({"chapters": chapters}, indent=2), encoding="utf-8")
    args.log.write_text("\n".join(flagged) + ("\n" if flagged else ""), encoding="utf-8")

    print(f"Wrote {len(chapters)} chapters to {args.output}", file=sys.stderr)
    if flagged:
        print(f"{len(flagged)} chapter(s) flagged as unclear -> see {args.log}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
