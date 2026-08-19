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
import sys
import time

import fitz  # pymupdf
import gradio as gr
import langdetect
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

LANGDETECT_MIN_CHARS = 100  # langdetect can be confidently WRONG on shorter samples, not just
# unsure -- confirmed on a real book: a 63-char correctly-OCR'd Korean sample got misclassified
# as "Estonian" at 0.71 confidence.
OCR_SAMPLE_PAGES = 6  # OCR is far less text-dense per page than native text, and front matter
# (cover/copyright/blank pages) can span several pages before real content starts -- confirmed
# on a real book: pages 0-5 totaled only 131 chars once correctly OCR'd, but page 6 alone
# yielded 1038. Native-text sampling doesn't need this many since it's already text-dense.
LANGUAGE_PREFILL_MIN_CONFIDENCE = 0.7  # threshold for pre-filling "Language being taught"
LANGUAGE_MISMATCH_MIN_CONFIDENCE = 0.85  # stricter: only block Analyze on a confident mismatch

# Script families to probe locally (via easyocr's own per-detection confidence) when we
# don't know a book's language yet and Cloud Vision isn't configured -- covers Latin,
# Hangul, Cyrillic, and CJK. A confidence gap this size is typical of a script mismatch:
# our own test against a real scanned Korean textbook showed 0.235 avg confidence from a
# blind English-only reader (garbled) vs 0.762 from the correct Korean reader (clean text).
SCRIPT_PROBE_LANGS = ("en", "ko", "ru", "ch_sim")
SCRIPT_PROBE_MIN_CONFIDENCE = 0.4  # below this even for the best candidate, treat as unusable

# langdetect's ISO 639-1 codes -> the display names used in "Language being taught" and
# (lowercased) as keys in OCR_LANGUAGE_CODES/VISION_LANGUAGE_HINTS below.
LANGDETECT_CODE_TO_NAME = {
    "en": "English", "ko": "Korean", "ja": "Japanese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "ru": "Russian", "uk": "Ukrainian", "bg": "Bulgarian",
    "be": "Belarusian", "mn": "Mongolian", "sr": "Serbian", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "tr": "Turkish", "ar": "Arabic", "hi": "Hindi", "bn": "Bengali", "vi": "Vietnamese",
    "cs": "Czech", "sk": "Slovak", "sl": "Slovenian", "hr": "Croatian", "hu": "Hungarian",
    "ro": "Romanian", "sv": "Swedish", "no": "Norwegian", "da": "Danish", "id": "Indonesian",
    "ms": "Malay", "fa": "Persian", "ur": "Urdu", "az": "Azerbaijani", "uz": "Uzbek",
    "ne": "Nepali", "th": "Thai", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "kk": "Kazakh",
    "el": "Greek", "he": "Hebrew", "fi": "Finnish", "et": "Estonian", "lv": "Latvian",
    "lt": "Lithuanian", "ca": "Catalan", "gu": "Gujarati", "mr": "Marathi", "pa": "Punjabi",
    "sw": "Swahili", "af": "Afrikaans", "tl": "Tagalog", "cy": "Welsh",
}

# easyocr's recognition models are grouped by script; only one non-Latin/non-Cyrillic
# "other" script (Korean, Japanese, Chinese, Thai, Tamil, Telugu, Kannada) can be loaded
# at a time, but each pairs fine with English. Names below are matched case-insensitively
# against the "Language being taught" field.
OCR_LANGUAGE_CODES = {
    "korean": "ko", "japanese": "ja", "chinese": "ch_sim", "chinese (simplified)": "ch_sim",
    "chinese (traditional)": "ch_tra", "thai": "th", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "russian": "ru", "ukrainian": "uk", "bulgarian": "bg", "belarusian": "be", "mongolian": "mn",
    "serbian": "rs_latin", "french": "fr", "german": "de", "spanish": "es", "italian": "it",
    "portuguese": "pt", "dutch": "nl", "polish": "pl", "turkish": "tr", "arabic": "ar",
    "hindi": "hi", "bengali": "bn", "vietnamese": "vi", "czech": "cs", "slovak": "sk",
    "slovenian": "sl", "croatian": "hr", "hungarian": "hu", "romanian": "ro", "swedish": "sv",
    "norwegian": "no", "danish": "da", "indonesian": "id", "malay": "ms", "persian": "fa",
    "farsi": "fa", "urdu": "ur", "azerbaijani": "az", "uzbek": "uz", "nepali": "ne",
}
# Languages easyocr has no model for at all (e.g. Kazakh's Cyrillic+extra letters): fall
# back to English-only OCR rather than silently mis-loading an unrelated script.

_ocr_readers: dict[tuple[str, ...], object] = {}


def _ocr_langs_for(target_language: str) -> tuple[str, ...]:
    code = OCR_LANGUAGE_CODES.get((target_language or "").strip().lower())
    if not code or code == "en":
        return ("en",)
    return (code, "en")


def _get_ocr_reader(langs: tuple[str, ...]):
    if langs not in _ocr_readers:
        import easyocr

        _ocr_readers[langs] = easyocr.Reader(list(langs), gpu=False)
    return _ocr_readers[langs]


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise gr.Error("GEMINI_API_KEY is not set (add it as a Space secret in Settings -> Repository secrets).")
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


CLOUD_VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
CLOUD_VISION_TIMEOUT_S = 30

# Google Cloud Vision's language hints; mostly the same ISO codes as OCR_LANGUAGE_CODES,
# but Vision uses "zh"/"zh-Hans"/"zh-Hant" for Chinese (not easyocr's ch_sim/ch_tra) and
# it does support Kazakh, which easyocr has no model for at all.
VISION_LANGUAGE_HINTS = {
    **{name: code for name, code in OCR_LANGUAGE_CODES.items() if not code.startswith("ch_")},
    "chinese": "zh", "chinese (simplified)": "zh-Hans", "chinese (traditional)": "zh-Hant",
    "serbian": "sr", "kazakh": "kk",
}


def _cloud_vision_api_key() -> str | None:
    return os.environ.get("GOOGLE_CLOUD_VISION_API_KEY") or None


def _vision_ocr_page(image_bytes: bytes, api_key: str, language_hint: str | None) -> str:
    import base64
    import urllib.error
    import urllib.request

    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }
        ]
    }
    if language_hint:
        payload["requests"][0]["imageContext"] = {"languageHints": [language_hint]}

    req = urllib.request.Request(
        f"{CLOUD_VISION_ENDPOINT}?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=CLOUD_VISION_TIMEOUT_S) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Cloud Vision HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e

    page_result = result["responses"][0]
    if "error" in page_result:
        raise RuntimeError(page_result["error"].get("message", "Cloud Vision error"))
    return page_result.get("fullTextAnnotation", {}).get("text", "")


def _cloud_ocr_pages(doc, target_language: str, progress: gr.Progress | None) -> list[str]:
    api_key = _cloud_vision_api_key()
    language_hint = VISION_LANGUAGE_HINTS.get((target_language or "").strip().lower())
    pages = []
    failures = 0
    for i in range(doc.page_count):
        if progress is not None:
            progress((i + 1) / doc.page_count * 0.5, desc=f"Cloud OCR page {i + 1}/{doc.page_count}")
        pix = doc[i].get_pixmap(dpi=OCR_DPI)
        try:
            pages.append(_vision_ocr_page(pix.tobytes("png"), api_key, language_hint))
        except Exception:
            pages.append("")
            failures += 1
    if failures:
        gr.Warning(f"Cloud OCR failed on {failures} page(s); those pages were treated as blank.")
    return pages


def _local_ocr_pages(doc, target_language: str, progress: gr.Progress | None) -> list[str]:
    reader = _get_ocr_reader(_ocr_langs_for(target_language))
    pages = []
    for i in range(doc.page_count):
        if progress is not None:
            progress((i + 1) / doc.page_count * 0.5, desc=f"OCR page {i + 1}/{doc.page_count}")
        pix = doc[i].get_pixmap(dpi=OCR_DPI)
        result = reader.readtext(pix.tobytes("png"), detail=1, paragraph=False)
        pages.append("\n".join(r[1] for r in result))
    return pages


def extract_page_texts(
    pdf_path: str, target_language: str = "English", progress: gr.Progress | None = None
) -> tuple[list[str], bool]:
    """Try the PDF's native text layer first; fall back to OCR if it's essentially empty (e.g.
    a scanned book with no text layer). Uses Google Cloud Vision if GOOGLE_CLOUD_VISION_API_KEY
    is set (fast, runs on Google's infrastructure); otherwise falls back to local easyocr
    (free, but slow on the CPU-only free tier -- can take close to an hour for a 100+ page scan)."""
    doc = fitz.open(pdf_path)
    native_pages = [doc[i].get_text() for i in range(doc.page_count)]
    avg_chars = sum(len(t) for t in native_pages) / max(doc.page_count, 1)

    if avg_chars >= NATIVE_TEXT_THRESHOLD:
        return native_pages, False

    if _cloud_vision_api_key():
        print(
            f"[ocr] GOOGLE_CLOUD_VISION_API_KEY found -- using Cloud Vision for {doc.page_count} page(s).",
            file=sys.stderr,
        )
        return _cloud_ocr_pages(doc, target_language, progress), True
    print(
        f"[ocr] No GOOGLE_CLOUD_VISION_API_KEY found in environment -- falling back to local easyocr "
        f"for {doc.page_count} page(s). Set that secret in the Space's Settings to speed this up.",
        file=sys.stderr,
    )
    return _local_ocr_pages(doc, target_language, progress), True


# ---------- Stage 1b: language detection ----------


def _sample_text_for_detection(pdf_path: str, hint_language: str | None = None) -> tuple[str, str]:
    """Grab enough text to guess a book's language without processing the whole book:
    native text from the first couple of pages, or (if that's essentially empty --
    i.e. a scan) a quick OCR of just the first page. hint_language, if given, picks
    the OCR reader for that sample; Cloud Vision (when configured) doesn't need one at
    all, since unlike easyocr it isn't locked to a single script family per pass.

    Returns (sample_text, source), where source is one of "native", "cloud_vision",
    "local_ocr_scripted" (a specific, trusted-in script reader), or "local_ocr_probed"
    (no trusted hint -- picked the best of several scripts by OCR confidence, or came
    up empty if none cleared the confidence floor). A reader forced onto the wrong
    script produces garbled Latin-lookalike text that langdetect can mistake for a
    real language -- our own test against a real scanned Korean textbook showed a
    blind English-only pass at 0.235 avg OCR confidence (garbage) vs 0.762 with the
    correct Korean reader (clean text), which is why we probe by confidence with a
    minimum floor instead of guessing blind."""
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        print("[lang-detect] PDF has 0 pages -- no sample text available.", file=sys.stderr)
        return "", "native"
    sample_pages = min(2, doc.page_count)
    text = "\n".join(doc[i].get_text() for i in range(sample_pages))
    if len(text.strip()) >= NATIVE_TEXT_THRESHOLD * sample_pages:
        print(
            f"[lang-detect] Using native text from the first {sample_pages} page(s): {len(text.strip())} char(s).",
            file=sys.stderr,
        )
        return text, "native"

    # A scanned book's first couple of pages are often sparse front matter (confirmed
    # on a real book: pages 0-5 totaled only 131 chars once correctly OCR'd -- real
    # content didn't start until page 6, which alone yielded 1038 chars) -- OCR more
    # pages than native-text sampling needs, since each page is much less text-dense.
    ocr_sample_pages = min(OCR_SAMPLE_PAGES, doc.page_count)
    pix_bytes_list = [doc[i].get_pixmap(dpi=OCR_DPI).tobytes("png") for i in range(ocr_sample_pages)]
    vision_key = _cloud_vision_api_key()
    if vision_key:
        print(
            f"[lang-detect] Native text too sparse ({len(text.strip())} char(s)) -- OCR-sampling the first "
            f"{ocr_sample_pages} page(s) via Cloud Vision (no script hint needed).",
            file=sys.stderr,
        )
        parts = []
        for pix_bytes in pix_bytes_list:
            try:
                parts.append(_vision_ocr_page(pix_bytes, vision_key, None))
            except Exception as e:
                print(f"[lang-detect] Cloud Vision sample failed on a page: {e!r}.", file=sys.stderr)
        text = "\n".join(parts)
        print(f"[lang-detect] Cloud Vision sample produced {len(text.strip())} char(s).", file=sys.stderr)
        return text, "cloud_vision"

    hinted_langs = _ocr_langs_for(hint_language) if hint_language else None
    if hinted_langs is not None and hinted_langs != ("en",):
        print(
            f"[lang-detect] Native text too sparse ({len(text.strip())} char(s)) -- OCR-sampling the first "
            f"{ocr_sample_pages} page(s) with trusted-hint reader langs={hinted_langs} "
            f"(hint_language={hint_language!r}).",
            file=sys.stderr,
        )
        reader = _get_ocr_reader(hinted_langs)
        parts = []
        for pix_bytes in pix_bytes_list:
            result = reader.readtext(pix_bytes, detail=1, paragraph=False)
            parts.append("\n".join(r[1] for r in result))
        text = "\n".join(parts)
        print(f"[lang-detect] OCR sample produced {len(text.strip())} char(s).", file=sys.stderr)
        return text, "local_ocr_scripted"

    # No hint, or the hint itself resolves to a bare English reader (exactly as
    # unreliable as no hint at all, if that hint turns out to be wrong) -- probe a
    # handful of script families by OCR confidence instead of guessing blind.
    print(
        f"[lang-detect] Native text too sparse ({len(text.strip())} char(s)) and no trusted script hint "
        f"(hint_language={hint_language!r}) -- probing {SCRIPT_PROBE_LANGS} across {ocr_sample_pages} page(s) "
        "by OCR confidence.",
        file=sys.stderr,
    )
    text, probed_langs, probed_conf = _probe_script_locally(pix_bytes_list)
    if probed_conf < SCRIPT_PROBE_MIN_CONFIDENCE:
        print(
            f"[lang-detect] Best script probe ({probed_langs}) only reached confidence {probed_conf:.2f} "
            f"(< {SCRIPT_PROBE_MIN_CONFIDENCE} minimum) -- none of the probed scripts fit well, "
            "treating sample as unusable.",
            file=sys.stderr,
        )
        return "", "local_ocr_probed"
    # probed_langs winning fairly against ko/ru/ch_sim on OCR confidence is a real signal,
    # English included -- unlike a blind single-pass guess, this doesn't need extra distrust.
    return text, "local_ocr_probed"


def _probe_script_locally(pix_bytes_list: list[bytes]) -> tuple[str, tuple[str, ...], float]:
    """Try OCR-ing the same sample page(s) with a handful of readers covering different
    script families, and return (text, langs, avg_confidence) for whichever one
    easyocr itself was most confident about across all sampled pages (its own
    per-detection confidence scores, not langdetect). Used only when we have no
    trusted language hint and Cloud Vision isn't configured -- a genuine confidence
    comparison beats blindly assuming English."""
    best_conf = -1.0
    best_text = ""
    best_langs: tuple[str, ...] = ("en",)
    for lang in SCRIPT_PROBE_LANGS:
        langs = ("en",) if lang == "en" else (lang, "en")
        try:
            reader = _get_ocr_reader(langs)
        except Exception as e:
            print(f"[lang-detect] Script probe {langs} failed to load: {e!r}.", file=sys.stderr)
            continue
        detections = []
        parts = []
        for pix_bytes in pix_bytes_list:
            try:
                result = reader.readtext(pix_bytes, detail=1, paragraph=False)
            except Exception as e:
                print(f"[lang-detect] Script probe {langs} OCR failed on a page: {e!r}.", file=sys.stderr)
                continue
            detections.extend(result)
            parts.append("\n".join(r[1] for r in result))
        avg_conf = sum(r[2] for r in detections) / len(detections) if detections else 0.0
        text = "\n".join(parts)
        print(
            f"[lang-detect] Script probe {langs}: avg_confidence={avg_conf:.3f}, {len(text.strip())} char(s) "
            f"across {len(pix_bytes_list)} page(s).",
            file=sys.stderr,
        )
        if avg_conf > best_conf:
            best_conf, best_text, best_langs = avg_conf, text, langs
    print(f"[lang-detect] Best script probe: langs={best_langs}, confidence={best_conf:.3f}.", file=sys.stderr)
    return best_text, best_langs, best_conf


def detect_language(pdf_path: str, hint_language: str | None = None) -> tuple[str | None, float]:
    """Best-effort (display_name, confidence) guess of a book's language from a quick
    sample. Returns (None, 0.0) if there's too little text, langdetect can't decide,
    or the only signal is an untrustworthy "English" guess from a blind English-only
    OCR pass (see _sample_text_for_detection) -- callers must treat (None, 0.0) as
    genuinely undetermined, never guess further."""
    print(f"[lang-detect] Starting detection for {pdf_path!r} (hint_language={hint_language!r}).", file=sys.stderr)
    text, source = _sample_text_for_detection(pdf_path, hint_language)
    if len(text.strip()) < LANGDETECT_MIN_CHARS:
        print(
            f"[lang-detect] Only {len(text.strip())} char(s) of sample text (< {LANGDETECT_MIN_CHARS} "
            "minimum) -- giving up, returning (None, 0.0).",
            file=sys.stderr,
        )
        return None, 0.0
    try:
        candidates = langdetect.detect_langs(text)
    except Exception as e:
        print(f"[lang-detect] langdetect raised {e!r} -- giving up, returning (None, 0.0).", file=sys.stderr)
        return None, 0.0
    if not candidates:
        print("[lang-detect] langdetect returned no candidates -- giving up, returning (None, 0.0).", file=sys.stderr)
        return None, 0.0
    top = candidates[0]
    name = LANGDETECT_CODE_TO_NAME.get(top.lang.lower())
    if name is None:
        print(
            f"[lang-detect] langdetect guessed code {top.lang!r} (prob={top.prob:.2f}), but it's not in "
            "our name map -- giving up, returning (None, 0.0).",
            file=sys.stderr,
        )
        return None, 0.0
    print(f"[lang-detect] Detected {name!r} with confidence {top.prob:.2f} (source={source}).", file=sys.stderr)
    return name, top.prob


def _language_base(name: str) -> str:
    """Normalizes e.g. "Chinese (Simplified)" and "chinese" to the same "chinese" so
    reasonable variant spellings aren't treated as a mismatch."""
    return name.strip().lower().split(" (")[0]


def check_language_mismatch(pdf_path: str, target_language: str) -> None:
    """Raises gr.Error if a quick sample of the book clearly doesn't match the stated
    "Language being taught" -- called before the expensive full-book OCR/analysis
    pipeline runs, so a wrong language selection fails in seconds, not after an hour
    of OCR. Only blocks on a confident detection; an inconclusive sample never blocks,
    since this is a safety check, not a guess."""
    print(f"[lang-detect] Pre-analysis mismatch check: target_language={target_language!r}.", file=sys.stderr)
    detected_name, confidence = detect_language(pdf_path, hint_language=target_language)
    if detected_name is None or confidence < LANGUAGE_MISMATCH_MIN_CONFIDENCE:
        print(
            f"[lang-detect] Mismatch check inconclusive (detected={detected_name!r}, confidence={confidence:.2f}, "
            f"threshold={LANGUAGE_MISMATCH_MIN_CONFIDENCE}) -- proceeding without blocking.",
            file=sys.stderr,
        )
        return
    if _language_base(detected_name) == _language_base(target_language):
        print(
            f"[lang-detect] Mismatch check passed: detected {detected_name!r} matches "
            f"target_language {target_language!r}.",
            file=sys.stderr,
        )
        return
    print(
        f"[lang-detect] MISMATCH: detected {detected_name!r} (confidence={confidence:.2f}) but "
        f"target_language is {target_language!r} -- blocking before the full pipeline runs.",
        file=sys.stderr,
    )
    article = "an" if detected_name[:1].lower() in "aeiou" else "a"
    raise gr.Error(
        f'This looks like {article} {detected_name} book, but you selected "{target_language}" as the '
        "language being taught -- please check the PDF or update the language field."
    )


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
            max_output_tokens=8000,
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
    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        return {"status": "error", "message": f"Gemini's response couldn't be read as a test ({e}). Try again."}
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


def on_pdf_upload(pdf_file):
    """Pre-fills "Language being taught" from a quick sample of the upload. Never
    guesses past the confidence threshold -- on failure/low confidence, leaves the
    field blank with an inline note instead, so the user isn't silently steered wrong."""
    if pdf_file is None:
        return gr.update(value=""), ""
    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
    print(f"[lang-detect] PDF uploaded: {pdf_path!r} -- running pre-fill detection.", file=sys.stderr)
    try:
        name, confidence = detect_language(pdf_path)
    except Exception as e:
        print(f"[lang-detect] Unexpected error during upload-time detection: {e!r}", file=sys.stderr)
        name, confidence = None, 0.0
    if name is None or confidence < LANGUAGE_PREFILL_MIN_CONFIDENCE:
        print(
            f"[lang-detect] Pre-fill: leaving field blank (name={name!r}, confidence={confidence:.2f}, "
            f"threshold={LANGUAGE_PREFILL_MIN_CONFIDENCE}).",
            file=sys.stderr,
        )
        return gr.update(value=""), "_Couldn't auto-detect language from this PDF -- please enter it manually._"
    print(f"[lang-detect] Pre-fill: setting field to {name!r} (confidence={confidence:.2f}).", file=sys.stderr)
    return gr.update(value=name), ""


def toggle_action_buttons(instruction_language):
    enabled = bool((instruction_language or "").strip())
    return gr.update(interactive=enabled), gr.update(interactive=enabled)


def run_analysis(pdf_file, target_language, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")
    if not (target_language or "").strip():
        raise gr.Error("Enter the language being taught first (auto-detect couldn't determine it).")
    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    progress(0, desc="Checking language...")
    check_language_mismatch(pdf_path, target_language)

    progress(0.02, desc="Reading PDF...")
    page_texts, used_ocr = extract_page_texts(pdf_path, target_language, progress)

    client = _client()

    progress(0.5, desc="Detecting chapters...")
    try:
        chapters_meta = detect_chapters(client, page_texts)
    except Exception as e:
        raise gr.Error(
            f"Text extraction succeeded, but chapter detection failed ({e}). Your upload wasn't wasted -- "
            "try clicking Analyze again."
        ) from e
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
        try:
            analyzed.append(analyze_chapter(client, c["chapter_id"], chapter_texts[c["chapter_id"]]))
        except Exception as e:
            # One chapter's Gemini call failing (rate limit, safety block, malformed
            # response) shouldn't discard every other chapter's already-successful results.
            analyzed.append(
                {
                    "chapter_id": c["chapter_id"],
                    "chapter_title": c.get("title", c["chapter_id"]),
                    "vocabulary": [],
                    "grammar_points": [],
                    "difficulty": {"cefr": "Unknown", "confidence": "low"},
                    "unclear": True,
                    "unclear_reason": f"Analysis failed: {e}",
                }
            )

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
    try:
        test = generate_test(
            client,
            selected_chapters,
            label,
            instruction_language or "English",
            target_language or "English",
            selected_texts,
        )
    except Exception as e:
        raise gr.Error(f"Test generation failed ({e}). Your book analysis wasn't lost -- try generating again.") from e

    if test["status"] != "ok":
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
        target_language_input = gr.Textbox(
            label="Language being taught",
            value="English",
            info="Auto-filled from the PDF on upload; also picks the OCR language for scans "
            '(e.g. "Korean", "Russian"). Edit it if the guess is wrong.',
        )
    language_detect_note = gr.Markdown()

    analyze_btn = gr.Button("1. Analyze book", variant="primary", interactive=False)
    ocr_note_output = gr.Markdown()
    book_structure_output = gr.Markdown()

    gr.Markdown("---")

    chapter_select = gr.CheckboxGroup(label="2. Chapters to test", choices=[])
    with gr.Row():
        week_label_input = gr.Textbox(label="Week/class label", placeholder="e.g. Week 1")
        instruction_language_input = gr.Textbox(
            label="Instruction language", value="", placeholder="e.g. English, Kazakh"
        )
    generate_btn = gr.Button("3. Generate test", variant="primary", interactive=False)

    with gr.Row():
        test_output = gr.Markdown(label="Student test")
        answer_key_output = gr.Markdown(label="Answer key")

    pdf_input.upload(
        on_pdf_upload,
        inputs=[pdf_input],
        outputs=[target_language_input, language_detect_note],
    )
    instruction_language_input.change(
        toggle_action_buttons,
        inputs=[instruction_language_input],
        outputs=[analyze_btn, generate_btn],
    )
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
    if _cloud_vision_api_key():
        print("[startup] GOOGLE_CLOUD_VISION_API_KEY found -- OCR will use Cloud Vision.", file=sys.stderr)
    else:
        print(
            "[startup] No GOOGLE_CLOUD_VISION_API_KEY found in this container's environment -- OCR will "
            "fall back to local easyocr. If you added this secret in Space Settings, this container may "
            "not have picked it up yet (needs a rebuild/restart) -- check the secret name matches exactly.",
            file=sys.stderr,
        )
    demo.queue().launch()
