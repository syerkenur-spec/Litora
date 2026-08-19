#!/usr/bin/env python3
"""
test-generator agent (see ../test-generator.md for the spec this implements).

Takes the week's selected chapters plus the book-structure.json produced by
book-analyzer, and asks Gemini to generate a short, mixed-format vocab/grammar
test calibrated to those chapters' actual difficulty. Writes a student-facing
test file and a separate answer key.

Usage:
    python generate_test.py <book_structure.json> --chapters ID [ID ...] \
        [-o test.json] [--answer-key answer-key.json] [--label "Week 3"]

<book_structure.json> is the book-structure.json output from book-analyzer.
--chapters takes one or more chapter_id values (the "chapter_id" field in
book-structure.json, e.g. from a filename like 01-greetings.md).

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment, or in a .env
file anywhere above the current directory.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types

load_dotenv()

MODEL = "gemini-flash-lite-latest"

RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 5  # seconds; doubles each attempt

SYSTEM_PROMPT_TEMPLATE = """You turn a set of already-analyzed coursebook chapters into a short test for the class that just covered them. The material being taught is {target_language} — treat its vocabulary and grammar structures as data from the chapters given, not hardcoded rules for any one language.

Rules:
- Match question difficulty to the chapters' own CEFR level and content — never default to a generic "intermediate" difficulty.
- Only test vocabulary and grammar points that are actually present in the given chapter data. Never invent new vocabulary, grammar rules, or examples.
- Vary question types across multiple_choice, fill_in_blank, and short_answer unless the user asks for a single type.
- Keep the test short enough for weekly/daily classroom use ({question_count_guidance}) — this is not a final exam.
- Every question's "specific_topic" must name the exact grammar point (matching a "name" from the chapter's grammar_points) or vocabulary theme it tests — never just the broad "vocabulary"/"grammar" category. This is what lets per-topic understanding be tracked later, not just a raw score.
- Every question must be answerable with confidence from its own prompt text alone, with exactly one defensible correct answer. Book sentences often depend on context you don't have here (a photo, a matching list, an earlier line naming who "she" is) to be unambiguous — if reusing one of those, either add enough context into the prompt itself to make the answer unique, or don't use that sentence. Never leave a blank where a different word than the intended answer would also be correct (e.g. "Laura's ___ is orange" with no way to know it's specifically her book and not her pen; "The blue book is ___" with no way to know whose it is).
- Hold this same standard of quality and book-grounding for every question, not just the first few — do not let later questions get more generic or less carefully checked than earlier ones.
- Do not grade answers or produce a comprehension report — only produce the test and its answer key.
- For multiple_choice questions, provide exactly 4 plausible choices in "choices" with one correct "answer" matching one of them exactly. For fill_in_blank and short_answer questions, set "choices" to an empty list.
- {instruction_language_rule}
- {source_text_rule}"""


def _instruction_language_rule(instruction_language: str, target_language: str) -> str:
    if instruction_language.strip().lower() == target_language.strip().lower():
        return f'Write every question\'s "prompt" in {target_language}.'
    return (
        f'Write every question\'s "prompt" (the instruction/question text) in {instruction_language}, '
        f"since the learner may not understand grammatical terms in {target_language}, the language being taught. "
        f'The actual target-language material being tested — vocabulary words, "choices", and the correct '
        f'"answer" — must stay in {target_language} (the language of the coursebook), never translated into {instruction_language}. '
        'Only the surrounding instruction text changes language, e.g. prompt: '
        f'"{{sentence in {instruction_language}}}: ___" with choices still in {target_language}.'
    )


def _question_count_guidance(num_chapters: int) -> str:
    if num_chapters <= 1:
        return "aim for roughly 8-12 questions"
    low, high = 8 * num_chapters, 12 * num_chapters
    return f"aim for roughly {low}-{high} questions, covering all {num_chapters} chapters roughly evenly"


def _source_text_rule(has_source_text: bool) -> str:
    if not has_source_text:
        return "Base questions on the extracted vocabulary/grammar data provided."
    return (
        "The raw source chapter text is also provided below the extracted data — it shows the book's actual "
        "exercises, example sentences, and phrasing. Most questions should closely mirror that real material: "
        "reuse the book's own example sentences and exercise formats (gap-fills built from a sentence that "
        "actually appears in the text, matching-style prompts, true/false about something the text states, etc.) "
        "rather than inventing generic quiz wording from scratch. A small minority of questions may use your own "
        "phrasing if it helps test a point the extracted data covers, but the majority must be visibly grounded "
        "in the actual book text, not just the extracted term list."
    )


MIN_CONTENT_ITEMS = 3


def _content_item_count(chapters: list[dict]) -> int:
    return sum(len(c.get("vocabulary", [])) + len(c.get("grammar_points", [])) for c in chapters)


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
                    "specific_topic": {"type": "string"},
                    "source_chapter_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": "string"},
                },
                "required": [
                    "id",
                    "type",
                    "topic",
                    "specific_topic",
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


def load_source_texts(source_dir: Path, chapter_ids: list[str]) -> dict[str, str]:
    """Look up {source_dir}/{chapter_id}.txt or .md for each chapter_id. Missing files are skipped."""
    texts = {}
    for chapter_id in chapter_ids:
        for suffix in (".txt", ".md"):
            path = source_dir / f"{chapter_id}{suffix}"
            if path.is_file():
                texts[chapter_id] = path.read_text(encoding="utf-8")
                break
    return texts


def _gemini_schema(schema):
    """Drop keys Gemini's OpenAPI-subset response_schema doesn't support (e.g. additionalProperties)."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("error: set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or a .env file")
    return genai.Client(api_key=api_key)


def _generate_with_retry(client: genai.Client, **kwargs):
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"  Gemini overloaded, retrying in {delay}s (attempt {attempt}/{RETRY_ATTEMPTS})...", file=sys.stderr)
            time.sleep(delay)
            delay *= 2


def generate_test(
    client: genai.Client,
    chapters: list[dict],
    label: str,
    instruction_language: str = "English",
    source_texts: dict[str, str] | None = None,
    target_language: str = "English",
) -> dict:
    item_count = _content_item_count(chapters)
    if item_count < MIN_CONTENT_ITEMS:
        return {
            "status": "insufficient_content",
            "message": (
                f"Selected chapter(s) have only {item_count} vocabulary/grammar item(s) total "
                f"(need at least {MIN_CONTENT_ITEMS}) — too thin for a meaningful test. "
                "Select more chapters or wait until more content is covered."
            ),
        }

    user_content = (
        f"Class/week label: {label}\n\n"
        "Generate a test from these chapters (JSON):\n\n"
        f"{json.dumps(chapters, indent=2)}"
    )
    if source_texts:
        raw_blocks = "\n\n".join(
            f"--- Raw source text: {chapter_id} ---\n{text}"
            for chapter_id, text in source_texts.items()
        )
        user_content += f"\n\nRaw source chapter text (for matching the book's real style):\n\n{raw_blocks}"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        target_language=target_language,
        question_count_guidance=_question_count_guidance(len(chapters)),
        instruction_language_rule=_instruction_language_rule(instruction_language, target_language),
        source_text_rule=_source_text_rule(bool(source_texts)),
    )
    response = _generate_with_retry(
        client,
        model=MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=_gemini_schema(TEST_SCHEMA),
            max_output_tokens=8000,
        ),
    )
    result = json.loads(response.text)
    result["status"] = "ok"
    return result


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
    parser.add_argument(
        "--instruction-language",
        default="English",
        help='Language for question instructions/prompts, e.g. "Kazakh" (default: English). '
        "Tested vocabulary, choices, and answers always stay in the coursebook's own language "
        "regardless of this setting — only the surrounding instruction text is translated.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory of raw chapter .txt/.md files (e.g. book_analyzer's chapters dir). When given, "
        "questions are built to mirror the book's own exercise style and example sentences instead of "
        "generic quiz wording.",
    )
    parser.add_argument(
        "--target-language",
        default="English",
        help='Language being taught (default: English) — the language the coursebook itself is in. '
        "Vocabulary, choices, and answers are always in this language regardless of --instruction-language.",
    )
    args = parser.parse_args()

    if not args.book_structure.is_file():
        print(f"error: {args.book_structure} is not a file", file=sys.stderr)
        return 1

    chapters = load_selected_chapters(args.book_structure, args.chapters)
    label = args.label or ", ".join(args.chapters)

    source_texts = load_source_texts(args.source_dir, args.chapters) if args.source_dir else {}
    if args.source_dir and not source_texts:
        print(f"warning: no source text found in {args.source_dir} for {args.chapters}", file=sys.stderr)

    client = _client()
    print(f"Generating test for: {label}...", file=sys.stderr)
    test = generate_test(client, chapters, label, args.instruction_language, source_texts, args.target_language)

    if test["status"] == "insufficient_content":
        print(f"error: {test['message']}", file=sys.stderr)
        return 1

    student_test, answer_key = split_student_and_key(test)

    args.output.write_text(json.dumps(student_test, indent=2), encoding="utf-8")
    args.answer_key.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")

    print(f"Wrote {len(student_test['questions'])} questions to {args.output}", file=sys.stderr)
    print(f"Wrote answer key to {args.answer_key}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
