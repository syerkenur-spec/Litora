#!/usr/bin/env python3
"""
test-generator agent (see ../test-generator.md for the spec this implements).

Takes the week's selected chapters plus the book-structure.json produced by
book-analyzer, and asks Claude to generate a short, mixed-format vocab/grammar
test calibrated to those chapters' actual difficulty. Writes a student-facing
test file and a separate answer key.

Usage:
    python generate_test.py <book_structure.json> --chapters ID [ID ...] \
        [-o test.json] [--answer-key answer-key.json] [--label "Week 3"]

<book_structure.json> is the book-structure.json output from book-analyzer.
--chapters takes one or more chapter_id values (the "chapter_id" field in
book-structure.json, e.g. from a filename like 01-greetings.md).

Requires ANTHROPIC_API_KEY in the environment (or an `ant auth login` profile).
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You turn a set of already-analyzed coursebook chapters into a short test for the class that just covered them.

Rules:
- Match question difficulty to the chapters' own CEFR level and content — never default to a generic "intermediate" difficulty.
- Only test vocabulary and grammar points that are actually present in the given chapter data. Never invent new vocabulary, grammar rules, or examples.
- Vary question types across multiple_choice, fill_in_blank, and short_answer unless the user asks for a single type.
- Keep the test short enough for weekly/daily classroom use (aim for roughly 8-12 questions) — this is not a final exam.
- Do not grade answers or produce a comprehension report — only produce the test and its answer key.
- For multiple_choice questions, provide exactly 4 plausible choices in "choices" with one correct "answer" matching one of them exactly. For fill_in_blank and short_answer questions, set "choices" to an empty list."""

TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "class_or_week_label": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["multiple_choice", "fill_in_blank", "short_answer"],
                    },
                    "topic": {"type": "string", "enum": ["vocabulary", "grammar"]},
                    "source_chapter_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": "string"},
                },
                "required": [
                    "id",
                    "type",
                    "topic",
                    "source_chapter_id",
                    "prompt",
                    "choices",
                    "answer",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["class_or_week_label", "questions"],
    "additionalProperties": False,
}


def load_selected_chapters(structure_path: Path, chapter_ids: list[str]) -> list[dict]:
    data = json.loads(structure_path.read_text(encoding="utf-8"))
    by_id = {c["chapter_id"]: c for c in data.get("chapters", [])}
    missing = [cid for cid in chapter_ids if cid not in by_id]
    if missing:
        available = ", ".join(sorted(by_id)) or "(none)"
        raise SystemExit(
            f"error: chapter_id(s) not found in {structure_path}: {', '.join(missing)}\n"
            f"available chapter_id(s): {available}"
        )
    return [by_id[cid] for cid in chapter_ids]


def generate_test(client: anthropic.Anthropic, chapters: list[dict], label: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": TEST_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Class/week label: {label}\n\n"
                    "Generate a test from these chapters (JSON):\n\n"
                    f"{json.dumps(chapters, indent=2)}"
                ),
            }
        ],
    )
    text_block = next(b for b in response.content if b.type == "text")
    return json.loads(text_block.text)


def split_student_and_key(test: dict) -> tuple[dict, dict]:
    student_questions = []
    key_entries = []
    for q in test["questions"]:
        student_q = {k: v for k, v in q.items() if k != "answer"}
        student_questions.append(student_q)
        key_entries.append(
            {
                "id": q["id"],
                "answer": q["answer"],
                "source_chapter_id": q["source_chapter_id"],
            }
        )
    student_test = {
        "class_or_week_label": test["class_or_week_label"],
        "questions": student_questions,
    }
    answer_key = {
        "class_or_week_label": test["class_or_week_label"],
        "answers": key_entries,
    }
    return student_test, answer_key


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("book_structure", type=Path, help="Path to book-structure.json from book-analyzer")
    parser.add_argument(
        "--chapters",
        nargs="+",
        required=True,
        metavar="CHAPTER_ID",
        help="chapter_id values (from book-structure.json) covered this week",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("test.json"))
    parser.add_argument("--answer-key", type=Path, default=Path("answer-key.json"))
    parser.add_argument("--label", default=None, help='Class/week label, e.g. "Week 3" (defaults to chapter list)')
    args = parser.parse_args()

    if not args.book_structure.is_file():
        print(f"error: {args.book_structure} is not a file", file=sys.stderr)
        return 1

    chapters = load_selected_chapters(args.book_structure, args.chapters)
    label = args.label or ", ".join(args.chapters)

    client = anthropic.Anthropic()
    print(f"Generating test for: {label}...", file=sys.stderr)
    test = generate_test(client, chapters, label)

    student_test, answer_key = split_student_and_key(test)

    args.output.write_text(json.dumps(student_test, indent=2), encoding="utf-8")
    args.answer_key.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")

    print(f"Wrote {len(student_test['questions'])} questions to {args.output}", file=sys.stderr)
    print(f"Wrote answer key to {args.answer_key}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
