#!/usr/bin/env python3
"""
Litora web app (Hugging Face Spaces / Gradio).

Wraps the full pipeline behind an upload form: PDF -> extract text (native
or OCR fallback) -> detect chapter boundaries (Gemini) -> analyze each
chapter (Gemini, book-analyzer logic) -> generate a test for the chapters
the teacher selects (Gemini, test-generator logic).

Self-contained (duplicates book_analyzer/test_generator's core logic rather
than importing them) so this deploys as a single Hugging Face Space without
needing the rest of the repo.

Requires GEMINI_API_KEY set as a Space secret (Settings -> Repository secrets).
"""

from __future__ import annotations

import json
import os
import time

import fitz  # pymupdf
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MODEL = "gemini-flash-lite-latest"
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 5

OCR_DPI = 200
NATIVE_TEXT_THRESHOLD = 20  # avg chars/page below this triggers OCR fallback
PAGE_SNIPPET_CHARS = 220
MIN_CONTENT_ITEMS = 3

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (add it as a Space secret).")
    return genai.Client(api_key=api_key)


def _gemini_schema(schema):
    """Drop keys Gemini's OpenAPI-subset response_schema doesn't support (e.g. additionalProperties)."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


def _generate_with_retry(client: genai.Client, **kwargs):
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError:
            if attempt == RETRY_ATTEMPTS:
                raise
            time.sleep(delay)
            delay *= 2


# ---------- Stage 1: text extraction ----------


def extract_page_texts(pdf_path: str, progress: gr.Progress | None = None) -> tuple[list[str], bool]:
    """Try the PDF's native text layer first; fall back to local OCR if it's essentially empty
    (e.g. a scanned book with no text layer, as with the Harmonize Starter PDF)."""
    doc = fitz.open(pdf_path)
    native_pages = [doc[i].get_text() for i in range(doc.page_count)]
    avg_chars = sum(len(t) for t in native_pages) / max(doc.page_count, 1)

    if avg_chars >= NATIVE_TEXT_THRESHOLD:
        return native_pages, False

    reader = _get_ocr_reader()
    ocr_pages = []
    for i in range(doc.page_count):
        if progress is not None:
            progress((i + 1) / doc.page_count * 0.5, desc=f"OCR page {i + 1}/{doc.page_count}")
        pix = doc[i].get_pixmap(dpi=OCR_DPI)
        result = reader.readtext(pix.tobytes("png"), detail=1, paragraph=False)
        ocr_pages.append("\n".join(r[1] for r in result))
    return ocr_pages, True


# ---------- Stage 2: chapter boundary detection ----------

CHAPTER_DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter_id": {"type": "string"},
                    "title": {"type": "string"},
                    "start_page": {"type": "integer"},
                    "end_page": {"type": "integer"},
                },
                "required": ["chapter_id", "title", "start_page", "end_page"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chapters"],
    "additionalProperties": False,
}

CHAPTER_DETECT_PROMPT = """You are given the start of every page of an OCR'd or extracted coursebook, one line per page as "page N: <snippet>".

Identify the book's real teaching chapters/units in book order -- including an intro/welcome unit if it teaches actual vocabulary or grammar (e.g. greetings, nationalities, the verb "be"), even if it isn't numbered like the other units. Skip only pages with no teaching content: cover, title page, table of contents, pure copyright/image-credit pages, grammar reference appendices, glossary, and irregular verb lists -- unless the book has no numbered units at all, in which case use your best judgement to split it into sensible teaching chapters.

For each real chapter, give:
- chapter_id: lowercase, hyphenated, numbered to match book order (e.g. "00-welcome", "01-our-favourites")
- title: the chapter's actual title
- start_page / end_page: 0-indexed page numbers (inclusive) spanning from this chapter's opening page up to (not including) the next chapter's opening page

Every page that contains teaching content must fall inside exactly one chapter's start_page/end_page range -- do not leave teaching pages uncovered. Base this on structural cues: unit numbers, section headers, and the table of contents if visible in the early pages."""


def detect_chapters(client: genai.Client, page_texts: list[str]) -> list[dict]:
    snippets = "\n".join(
        f"page {i}: {t[:PAGE_SNIPPET_CHARS].replace(chr(10), ' ')}" for i, t in enumerate(page_texts)
    )
    response = _generate_with_retry(
        client,
        model=MODEL,
        contents=snippets,
        config=types.GenerateContentConfig(
            system_instruction=CHAPTER_DETECT_PROMPT,
            response_mime_type="application/json",
            response_schema=_gemini_schema(CHAPTER_DETECT_SCHEMA),
            max_output_tokens=4000,
        ),
    )
    return json.loads(response.text)["chapters"]


def build_chapter_texts(page_texts: list[str], chapters: list[dict]) -> dict[str, str]:
    texts = {}
    for c in chapters:
        parts = [page_texts[i] for i in range(c["start_page"], c["end_page"] + 1) if 0 <= i < len(page_texts)]
        texts[c["chapter_id"]] = "\n\n".join(parts)
    return texts


def uncovered_pages(page_texts: list[str], chapters: list[dict]) -> list[int]:
    """Page indices not claimed by any detected chapter's start_page..end_page range."""
    covered: set[int] = set()
    for c in chapters:
        covered.update(range(c["start_page"], c["end_page"] + 1))
    return [i for i in range(len(page_texts)) if i not in covered]


# ---------- Stage 3: book-analyzer (per chapter) ----------

ANALYZE_SYSTEM_PROMPT = """You break down a single chapter of a language coursebook into structured data the rest of a coursebook app can use.

Rules:
- Never invent content that isn't in the chapter text. If the chapter is unclear, too sparse, or ambiguous, set "unclear" to true and explain why in "unclear_reason" instead of guessing.
- Extract only vocabulary and grammar points actually introduced in this chapter -- don't pull in content from other chapters you might infer exist.
- Estimate difficulty using CEFR levels (A1-C2) where possible. If you can't judge confidently, use "Unknown" for cefr and "low" for confidence.
- Do not generate test questions, comprehension summaries, or prose commentary -- only the structured fields requested."""

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
                "cefr": {"type": "string", "enum": ["A1", "A2", "B1", "B2", "C1", "C2", "Unknown"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["cefr", "confidence"],
            "additionalProperties": False,
        },
        "unclear": {"type": "boolean"},
        "unclear_reason": {"type": "string"},
    },
    "required": ["chapter_title", "vocabulary", "grammar_points", "difficulty", "unclear", "unclear_reason"],
    "additionalProperties": False,
}


def analyze_chapter(client: genai.Client, chapter_id: str, text: str) -> dict:
    response = _generate_with_retry(
        client,
        model=MODEL,
        contents=f"Chapter file: {chapter_id}\n\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=ANALYZE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_gemini_schema(CHAPTER_SCHEMA),
            max_output_tokens=8000,
        ),
    )
    result = json.loads(response.text)
    result["chapter_id"] = chapter_id
    return result


# ---------- Stage 4: test-generator ----------

TEST_SYSTEM_PROMPT_TEMPLATE = """You turn a set of already-analyzed coursebook chapters into a short test for the class that just covered them. The material being taught is {target_language} -- treat its vocabulary and grammar structures as data from the chapters given, not hardcoded rules for any one language.

Rules:
- Match question difficulty to the chapters' own CEFR level and content -- never default to a generic "intermediate" difficulty.
- Only test vocabulary and grammar points that are actually present in the given chapter data. Never invent new vocabulary, grammar rules, or examples.
- Vary question types across multiple_choice, fill_in_blank, and short_answer unless the user asks for a single type.
- Keep the test short enough for weekly/daily classroom use ({question_count_guidance}) -- this is not a final exam.
- Every question's "specific_topic" must name the exact grammar point (matching a "name" from the chapter's grammar_points) or vocabulary theme it tests -- never just the broad "vocabulary"/"grammar" category.
- Every question must be answerable with confidence from its own prompt text alone, with exactly one defensible correct answer. Book sentences often depend on context you don't have here (a photo, a matching list, an earlier line naming who "she" is) to be unambiguous -- if reusing one of those, either add enough context into the prompt itself to make the answer unique, or don't use that sentence. Never leave a blank where a different word than the intended answer would also be correct.
- Hold this same standard of quality and book-grounding for every question, not just the first few -- do not let later questions get more generic or less carefully checked than earlier ones.
- Do not grade answers or produce a comprehension report -- only produce the test and its answer key.
- For multiple_choice questions, provide exactly 4 plausible choices in "choices" with one correct "answer" matching one of them exactly. For fill_in_blank and short_answer questions, set "choices" to an empty list.
- {instruction_language_rule}
- {source_text_rule}"""

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
                    "type": {"type": "string", "enum": ["multiple_choice", "fill_in_blank", "short_answer"]},
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


def _instruction_language_rule(instruction_language: str, target_language: str) -> str:
    if instruction_language.strip().lower() == target_language.strip().lower():
        return f'Write every question\'s "prompt" in {target_language}.'
    return (
        f'Write every question\'s "prompt" (the instruction/question text) in {instruction_language}, '
        f"since the learner may not understand grammatical terms in {target_language}, the language being taught. "
        f'The actual target-language material being tested -- vocabulary words, "choices", and the correct '
        f'"answer" -- must stay in {target_language} (the language of the coursebook), never translated into '
        f"{instruction_language}."
    )


def _question_count_guidance(num_chapters: int) -> str:
    if num_chapters <= 1:
        return "aim for roughly 8-12 questions"
    low, high = 8 * num_chapters, 12 * num_chapters
    return f"aim for roughly {low}-{high} questions, covering all {num_chapters} chapters roughly evenly"


def _content_item_count(chapters: list[dict]) -> int:
    return sum(len(c.get("vocabulary", [])) + len(c.get("grammar_points", [])) for c in chapters)


def generate_test(
    client: genai.Client,
    chapters: list[dict],
    label: str,
    instruction_language: str,
    target_language: str,
    source_texts: dict[str, str] | None = None,
) -> dict:
    item_count = _content_item_count(chapters)
    if item_count < MIN_CONTENT_ITEMS:
        return {
            "status": "insufficient_content",
            "message": (
                f"Selected chapter(s) have only {item_count} vocabulary/grammar item(s) total "
                f"(need at least {MIN_CONTENT_ITEMS}) -- too thin for a meaningful test."
            ),
        }

    user_content = (
        f"Class/week label: {label}\n\n"
        f"Generate a test from these chapters (JSON):\n\n{json.dumps(chapters, indent=2)}"
    )
    if source_texts:
        raw_blocks = "\n\n".join(f"--- Raw source text: {cid} ---\n{t}" for cid, t in source_texts.items())
        user_content += f"\n\nRaw source chapter text (for matching the book's real style):\n\n{raw_blocks}"
        source_text_rule = (
            "The raw source chapter text is also provided below the extracted data -- it shows the book's "
            "actual exercises, example sentences, and phrasing. Most questions should closely mirror that real "
            "material rather than inventing generic quiz wording from scratch."
        )
    else:
        source_text_rule = "Base questions on the extracted vocabulary/grammar data provided."

    system_prompt = TEST_SYSTEM_PROMPT_TEMPLATE.format(
        target_language=target_language,
        question_count_guidance=_question_count_guidance(len(chapters)),
        instruction_language_rule=_instruction_language_rule(instruction_language, target_language),
        source_text_rule=source_text_rule,
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


# ---------- formatting helpers ----------


def format_book_structure(chapters: list[dict]) -> str:
    lines = []
    for c in chapters:
        flag = f" — **FLAGGED**: {c['unclear_reason']}" if c.get("unclear") else ""
        lines.append(f"### {c['chapter_id']} — {c['chapter_title']}{flag}")
        lines.append(f"CEFR: {c['difficulty']['cefr']} ({c['difficulty']['confidence']} confidence)")
        lines.append(f"**Vocabulary** ({len(c['vocabulary'])}): " + ", ".join(v["term"] for v in c["vocabulary"]))
        lines.append(f"**Grammar** ({len(c['grammar_points'])}): " + ", ".join(g["name"] for g in c["grammar_points"]))
        lines.append("")
    return "\n".join(lines)


def format_test(test: dict) -> str:
    lines = [f"## {test['class_or_week_label']}", ""]
    for q in test["questions"]:
        lines.append(f"**{q['id']}** — _{q['topic']}: {q['specific_topic']}_ ({q['type']})")
        lines.append(q["prompt"])
        if q["choices"]:
            lines.append("")
            lines.extend(f"- {choice}" for choice in q["choices"])
        lines.append("")
    return "\n".join(lines)


def format_answer_key(test: dict) -> str:
    lines = ["## Answer key", ""]
    for q in test["questions"]:
        lines.append(f"- **{q['id']}**: {q['answer']}")
    return "\n".join(lines)


# ---------- Gradio callbacks ----------


def run_analysis(pdf_file, target_language, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")
    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    progress(0, desc="Reading PDF...")
    page_texts, used_ocr = extract_page_texts(pdf_path, progress)

    client = _client()

    progress(0.5, desc="Detecting chapters...")
    chapters_meta = detect_chapters(client, page_texts)
    if not chapters_meta:
        raise gr.Error("Couldn't detect any chapters in this PDF.")

    missed = uncovered_pages(page_texts, chapters_meta)
    if missed:
        gr.Warning(
            f"{len(missed)} page(s) didn't fall inside any detected chapter and were excluded "
            f"from analysis (0-indexed pages: {missed})."
        )

    chapter_texts = build_chapter_texts(page_texts, chapters_meta)

    analyzed = []
    for i, c in enumerate(chapters_meta):
        progress(0.55 + 0.4 * (i / len(chapters_meta)), desc=f"Analyzing {c['chapter_id']}...")
        analyzed.append(analyze_chapter(client, c["chapter_id"], chapter_texts[c["chapter_id"]]))

    progress(1.0, desc="Done")
    ocr_note = (
        "*OCR was used — no text layer was found in this PDF (it's likely a scan).*"
        if used_ocr
        else "*Text was read directly from the PDF — no OCR needed.*"
    )
    choices = [f"{c['chapter_id']} — {c['title']}" for c in chapters_meta]

    return (
        format_book_structure(analyzed),
        ocr_note,
        analyzed,
        chapter_texts,
        gr.update(choices=choices, value=choices[:1]),
    )


def run_test_generation(
    analyzed_state, chapter_texts_state, selected_labels, week_label, instruction_language, target_language
):
    if not analyzed_state or not selected_labels:
        raise gr.Error("Analyze a book and select at least one chapter first.")

    selected_ids = [label.split(" — ")[0] for label in selected_labels]
    by_id = {c["chapter_id"]: c for c in analyzed_state}
    selected_chapters = [by_id[cid] for cid in selected_ids if cid in by_id]
    selected_texts = {cid: chapter_texts_state[cid] for cid in selected_ids if cid in chapter_texts_state}

    client = _client()
    label = week_label or ", ".join(selected_ids)
    test = generate_test(
        client, selected_chapters, label, instruction_language or "English", target_language or "English", selected_texts
    )

    if test["status"] == "insufficient_content":
        raise gr.Error(test["message"])

    return format_test(test), format_answer_key(test)


with gr.Blocks(title="Litora") as demo:
    gr.Markdown(
        "# Litora — book to test\n"
        "Upload a coursebook PDF. Litora reads it (OCR if it's a scan), finds the real teaching "
        "chapters, extracts vocabulary/grammar per chapter, and generates a test from whichever "
        "chapters you pick."
    )

    analyzed_state = gr.State()
    chapter_texts_state = gr.State()

    with gr.Row():
        pdf_input = gr.File(label="Coursebook PDF", file_types=[".pdf"])
        target_language_input = gr.Textbox(label="Language being taught", value="English")

    analyze_btn = gr.Button("1. Analyze book", variant="primary")
    ocr_note_output = gr.Markdown()
    book_structure_output = gr.Markdown()

    gr.Markdown("---")

    chapter_select = gr.CheckboxGroup(label="2. Chapters to test", choices=[])
    with gr.Row():
        week_label_input = gr.Textbox(label="Week/class label", placeholder="e.g. Week 1")
        instruction_language_input = gr.Textbox(
            label="Instruction language", value="English", placeholder="e.g. Kazakh"
        )
    generate_btn = gr.Button("3. Generate test", variant="primary")

    with gr.Row():
        test_output = gr.Markdown(label="Student test")
        answer_key_output = gr.Markdown(label="Answer key")

    analyze_btn.click(
        run_analysis,
        inputs=[pdf_input, target_language_input],
        outputs=[book_structure_output, ocr_note_output, analyzed_state, chapter_texts_state, chapter_select],
    )
    generate_btn.click(
        run_test_generation,
        inputs=[
            analyzed_state,
            chapter_texts_state,
            chapter_select,
            week_label_input,
            instruction_language_input,
            target_language_input,
        ],
        outputs=[test_output, answer_key_output],
    )

if __name__ == "__main__":
    demo.queue().launch()
