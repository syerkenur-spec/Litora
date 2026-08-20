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

Optional caching: if LITORA_CACHE_DATASET_REPO (e.g. "yourname/litora-book-cache") is set,
already-analyzed books are cached by PDF SHA-256 in that private Hugging Face Hub dataset repo,
so a re-upload of the same PDF skips OCR and Gemini analysis entirely. This also requires an
HF_TOKEN Space secret with write access to that repo (Spaces don't get a write-capable token by
default) -- without both set, caching is silently skipped and every upload is analyzed fresh.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fitz  # pymupdf
import gradio as gr
import langdetect
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()

MODEL = "gemini-flash-lite-latest"
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 5

OCR_DPI = 200
NATIVE_TEXT_THRESHOLD = 20  # avg chars/page below this triggers OCR fallback
PAGE_SNIPPET_CHARS = 220
MIN_CONTENT_ITEMS = 3

CACHE_DATASET_REPO = os.environ.get("LITORA_CACHE_DATASET_REPO")  # e.g. "yourname/litora-book-cache";
# unset (default) disables the analyzed-book cache entirely.

# In-app access-code gate (a single shared textbox, not a browser login popup) -- lets you share
# the app's URL publicly (e.g. on a custom domain) while still keeping randoms/search engines out.
# Set as a Space secret: "SDU2026,TEACHER-01,TEACHER-02" -- one code per teacher/class, editable
# without a redeploy (just update the secret; the Space restarts itself). Not meant to stop a
# determined attacker -- it's a lightweight, per-browser-session gate (see check_access_code()).
LITORA_ACCESS_CODES = os.environ.get("LITORA_ACCESS_CODES")

# Baked-in fallback code, always valid in addition to whatever LITORA_ACCESS_CODES adds -- a
# safety net in case that secret isn't set/picked up. NOTE: unlike the secret, this is visible
# to anyone who can view this file (e.g. the Space's Files tab on a Public Space) -- once the
# secret is confirmed working, consider removing this or rotating it to something you control.
FALLBACK_ACCESS_CODES = {"KPSM-J5W5"}


def _parse_access_codes(raw: str | None) -> set[str]:
    codes = set(FALLBACK_ACCESS_CODES)
    if raw:
        codes |= {code.strip() for code in raw.split(",") if code.strip()}
    return codes

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

# Best-effort flag emoji for the upload-time detection banner (cosmetic only -- a missing
# entry just means no emoji is shown, never an error).
LANGDETECT_CODE_TO_FLAG = {
    "en": "🇬🇧", "ko": "🇰🇷", "ja": "🇯🇵", "zh-cn": "🇨🇳", "zh-tw": "🇹🇼", "ru": "🇷🇺",
    "uk": "🇺🇦", "bg": "🇧🇬", "be": "🇧🇾", "mn": "🇲🇳", "sr": "🇷🇸", "fr": "🇫🇷", "de": "🇩🇪",
    "es": "🇪🇸", "it": "🇮🇹", "pt": "🇵🇹", "nl": "🇳🇱", "pl": "🇵🇱", "tr": "🇹🇷", "ar": "🇸🇦",
    "hi": "🇮🇳", "bn": "🇧🇩", "vi": "🇻🇳", "cs": "🇨🇿", "sk": "🇸🇰", "sl": "🇸🇮", "hr": "🇭🇷",
    "hu": "🇭🇺", "ro": "🇷🇴", "sv": "🇸🇪", "no": "🇳🇴", "da": "🇩🇰", "id": "🇮🇩", "ms": "🇲🇾",
    "fa": "🇮🇷", "ur": "🇵🇰", "az": "🇦🇿", "uz": "🇺🇿", "ne": "🇳🇵", "th": "🇹🇭", "ta": "🇮🇳",
    "te": "🇮🇳", "kn": "🇮🇳", "kk": "🇰🇿", "el": "🇬🇷", "he": "🇮🇱", "fi": "🇫🇮", "et": "🇪🇪",
    "lv": "🇱🇻", "lt": "🇱🇹", "ca": "🇪🇸", "gu": "🇮🇳", "mr": "🇮🇳", "pa": "🇮🇳", "sw": "🇰🇪",
    "af": "🇿🇦", "tl": "🇵🇭", "cy": "🏴",
}

# Keyed by display name (what detect_language() actually returns) rather than langdetect code,
# since that's what the upload banner has on hand.
LANGUAGE_NAME_TO_FLAG = {
    name: LANGDETECT_CODE_TO_FLAG[code]
    for code, name in LANGDETECT_CODE_TO_NAME.items()
    if code in LANGDETECT_CODE_TO_FLAG
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


# ---------- Stage 0: analyzed-book cache ----------
#
# Keyed by SHA-256 of the uploaded PDF, stored as one JSON file per book in a private Hugging
# Face Hub dataset repo (CACHE_DATASET_REPO). This survives Space restarts/rebuilds/sleep-wake
# without needing the paid Persistent Storage add-on, since it lives outside the Space's own
# (ephemeral) container filesystem. A cache miss, a missing HF_TOKEN, or any Hub error is treated
# as "not cached" -- caching is a pure optimization and must never block or fail an analysis run.

_cache_repo_ready = False


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _cache_filename(sha256: str) -> str:
    return f"book-cache/{sha256}.json"


def pdf_sha256(pdf_path: str) -> str:
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cached_analysis(sha256: str) -> dict | None:
    if not CACHE_DATASET_REPO:
        return None
    try:
        local_path = hf_hub_download(
            repo_id=CACHE_DATASET_REPO,
            repo_type="dataset",
            filename=_cache_filename(sha256),
            token=_hf_token(),
        )
        return json.loads(Path(local_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(
            f"[cache] No usable cached analysis for {sha256[:12]}... ({e.__class__.__name__}: {e}) "
            "-- analyzing fresh.",
            file=sys.stderr,
        )
        return None


def save_cached_analysis(sha256: str, payload: dict) -> None:
    global _cache_repo_ready
    if not CACHE_DATASET_REPO:
        return
    token = _hf_token()
    if not token:
        print(
            "[cache] LITORA_CACHE_DATASET_REPO is set but no HF_TOKEN/HUGGINGFACE_TOKEN secret was found "
            "-- skipping cache write for this run.",
            file=sys.stderr,
        )
        return
    api = HfApi(token=token)
    try:
        if not _cache_repo_ready:
            api.create_repo(repo_id=CACHE_DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
            _cache_repo_ready = True
        api.upload_file(
            path_or_fileobj=json.dumps(payload, indent=2).encode("utf-8"),
            path_in_repo=_cache_filename(sha256),
            repo_id=CACHE_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Cache analysis for {sha256[:12]}",
        )
        print(f"[cache] Saved analysis for {sha256[:12]}... to {CACHE_DATASET_REPO}.", file=sys.stderr)
    except Exception as e:
        print(f"[cache] Failed to save cached analysis ({e!r}) -- continuing without caching this run.", file=sys.stderr)


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
            progress((i + 1) / doc.page_count * 0.5, desc=f"Processing: reading page {i + 1}/{doc.page_count} (Cloud OCR)...")
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
            progress(
                (i + 1) / doc.page_count * 0.5,
                desc=f"Processing: reading page {i + 1}/{doc.page_count} (local OCR — this is the slow step)...",
            )
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
- For each grammar point, also extract "pattern" (the compact grammar form exactly as the book writes it, e.g. "-니?, -자") and "example_sentences" (1-2 real sentences copied verbatim from the chapter text that show the pattern in use). Never invent or paraphrase an example -- if the chapter doesn't give one for a point, leave "example_sentences" as an empty list.
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
                    "pattern": {"type": "string"},
                    "example_sentences": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "explanation", "pattern", "example_sentences"],
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

TEST_SYSTEM_PROMPT_TEMPLATE = """You turn a set of already-analyzed coursebook chapters into a short test for the class that just covered them, modeled on how this book itself drills material -- never generic "pick the matching phrase" multiple choice. The material being taught is {target_language} -- treat its vocabulary and grammar structures as data from the chapters given, not hardcoded rules for any one language.

Every question must be one of these three types, matching how real coursebooks drill material:
- "dialogue_completion": a short two-line exchange. Give speaker 가's line (plus enough situational context to make the reply unambiguous) in {target_language}, and the student writes speaker 나's response using the chapter's target grammar pattern, correctly conjugated/inflected for the context and formality level shown in the book. Put 가's line and any setup in "prompt", ending with the blank to fill (e.g. "나: ___", or the book's own dialogue-tag convention). "answer" is the one defensible correct 나-line.
- "word_bank_completion": pick ~5 target vocabulary words from the chapter (adjectives/verbs in their dictionary/base form) as "word_bank", then write 1-3 short context sentences or a short dialogue in "prompt" with a blank the student fills by choosing the right word from the bank AND inflecting/conjugating it correctly for that sentence. "answer" is the correctly conjugated word actually filled in.
- "guided_response": an open-ended production exercise -- state a topic/situation in "prompt", and if the chapter's source text or example_sentences show one, a model dialogue/sentence pattern to follow. The student produces their own original response using the chapter's grammar pattern; there is no single correct answer, so "answer" holds one reasonable sample response for the teacher's reference, not something to be graded by exact match.

Rules:
- Match question difficulty to the chapters' own CEFR level and content -- never default to a generic "intermediate" difficulty.
- Only test vocabulary and grammar points that are actually present in the given chapter data. Never invent new vocabulary, grammar rules, or examples.
- Test real understanding of meaning, usage, and how a grammar point actually functions -- not rote recognition. A student who has only memorized a word list, without understanding how to use it, should not be able to pass.
- Ground every dialogue_completion and guided_response in the chapter's own grammar_points "pattern" and "example_sentences" where given -- model the new dialogue/sentence on how the book's own examples actually use the pattern (register, sentence shape, typical subjects), rather than inventing an unrelated scenario.
- word_bank_completion's word bank must be drawn from the chapter's own "vocabulary" list, and the blank's correct answer must require real inflection/conjugation (not just the bare dictionary form) whenever the target language's grammar makes that possible.
- Vary which of the three types is used across the test and across chapters/topics rather than repeating one type throughout; use guided_response sparingly (a small minority of the test) since it can't be auto-graded by exact match.
- Every dialogue_completion and word_bank_completion question must be answerable with confidence from its own prompt text alone, with exactly one defensible correct answer. Book sentences often depend on context you don't have here (a photo, a matching list, an earlier line naming who "she" is) to be unambiguous -- if reusing one of those, either add enough context into the prompt itself to make the answer unique, or don't use that sentence. Never leave a blank where a different word than the intended answer would also be correct.
- Hold this same standard of quality and book-grounding for every question, not just the first few -- do not let later questions get more generic or less carefully checked than earlier ones.
- Keep the test short enough for weekly/daily classroom use ({question_count_guidance}) -- this is not a final exam.
- Every question's "specific_topic" must name the exact grammar point (matching a "name" from the chapter's grammar_points) or vocabulary theme it tests -- never just the broad "vocabulary"/"grammar" category.
- Set "word_bank" to a non-empty list of ~5 words only for word_bank_completion questions; leave it as an empty list for dialogue_completion and guided_response.
- Do not grade answers or produce a comprehension report -- only produce the test and its answer key.
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
                    "type": {
                        "type": "string",
                        "enum": ["dialogue_completion", "word_bank_completion", "guided_response"],
                    },
                    "topic": {"type": "string", "enum": ["vocabulary", "grammar"]},
                    "specific_topic": {"type": "string"},
                    "source_chapter_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "word_bank": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": "string"},
                },
                "required": [
                    "id",
                    "type",
                    "topic",
                    "specific_topic",
                    "source_chapter_id",
                    "prompt",
                    "word_bank",
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
        f'Write every question\'s "prompt" (the instruction/setup text) in {instruction_language}, '
        f"since the learner may not understand grammatical terms in {target_language}, the language being taught. "
        f'The actual target-language material being tested -- dialogue lines, "word_bank" entries, and the correct '
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
        grammar_labels = [
            f"{g['name']} ({g['pattern']})" if g.get("pattern") else g["name"] for g in c["grammar_points"]
        ]
        lines.append(f"**Grammar** ({len(c['grammar_points'])}): " + ", ".join(grammar_labels))
        lines.append("")
    return "\n".join(lines)


MAX_QUESTIONS = 30  # Gradio needs a fixed set of components pre-allocated; this comfortably
# covers the app's own guidance (8-12 questions/chapter) for a 1-3 chapter selection, which
# covers typical weekly-test usage. A larger generated test gets truncated with a warning.


def _blank_question_updates() -> tuple:
    """gr.update() tuple for one hidden, empty question slot: (group, label, textbox)."""
    return (
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=False, value=""),
    )


def _question_slot_updates(q: dict) -> tuple:
    """gr.update() tuple for one populated, visible question slot: (group, label, textbox)."""
    label = f"**{q['id']}** — _{q['topic']}: {q['specific_topic']}_ ({q['type']})\n\n"
    if q.get("word_bank"):
        label += f"Word bank: {', '.join(q['word_bank'])}\n\n"
    label += q["prompt"]
    return (
        gr.update(visible=True),
        gr.update(value=label),
        gr.update(visible=True, value=""),
    )


def grade_test(test: dict, *text_vals):
    """Grades submitted answers against test["answer"] -- case-insensitive, whitespace-trimmed
    exact match -- for dialogue_completion/word_bank_completion. guided_response has no single
    correct answer, so it's never marked right/wrong: the student's response and a sample answer
    are shown side by side for self-check instead, and it's excluded from the score fraction.
    Returns a single update for the results block, which is the only place the answer key is
    ever shown -- it's never rendered before this runs."""
    if not test:
        raise gr.Error("Generate a test first.")
    questions = test["questions"]

    correct_count = 0
    graded_count = 0
    lines = []
    for i, q in enumerate(questions):
        submitted_norm = (text_vals[i] or "").strip()
        if q["type"] == "guided_response":
            lines.append(
                f"📝 **{q['id']}** — your answer: {submitted_norm or '_(no answer)_'}\n"
                f"   _Sample response (many valid answers exist):_ {q['answer'].strip()}"
            )
            continue
        graded_count += 1
        correct_norm = q["answer"].strip()
        is_correct = submitted_norm.lower() == correct_norm.lower()
        if is_correct:
            correct_count += 1
            lines.append(f"✅ **{q['id']}** — your answer: {submitted_norm or '_(no answer)_'}")
        else:
            lines.append(
                f"❌ **{q['id']}** — your answer: {submitted_norm or '_(no answer)_'} "
                f"— correct answer: **{correct_norm}**"
            )

    footnote = ""
    if graded_count < len(questions):
        footnote = f" (+{len(questions) - graded_count} guided-response question(s) for self-check, not scored)"
    header = f"## Results: {correct_count}/{graded_count} correct{footnote}\n\n"
    return gr.update(visible=True, value=header + "\n\n".join(lines))


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
        return (
            gr.update(value=""),
            "_Couldn't auto-detect the language from this PDF -- please enter it below._",
        )
    print(f"[lang-detect] Pre-fill: setting field to {name!r} (confidence={confidence:.2f}).", file=sys.stderr)
    flag = LANGUAGE_NAME_TO_FLAG.get(name, "")
    note = (
        f"**Language detected: {name} {flag}**\n"
        f"We think this book teaches {name}. Already filled in below -- change it if that's wrong."
    )
    return gr.update(value=name), note


def toggle_action_buttons(instruction_language):
    enabled = bool((instruction_language or "").strip())
    return gr.update(interactive=enabled), gr.update(interactive=enabled)


def run_analysis(pdf_file, target_language, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")
    if not (target_language or "").strip():
        raise gr.Error("Enter the language being taught first (auto-detect couldn't determine it).")
    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    progress(0, desc="Processing: checking for a cached copy of this book...")
    sha256 = pdf_sha256(pdf_path)
    cached = load_cached_analysis(sha256)
    if cached:
        progress(1.0, desc="Ready — loaded from cache")
        chapters_meta = cached["chapters_meta"]
        analyzed = cached["chapters"]
        chapter_texts = cached["chapter_texts"]
        ocr_note = (
            f"*Loaded from cache — this exact PDF (SHA-256 `{sha256[:12]}…`) was already analyzed on "
            f"{cached.get('cached_at', 'a previous run')}; OCR and Gemini analysis were skipped.*"
        )
        choices = [f"{c['chapter_id']} — {c['title']}" for c in chapters_meta]
        return (
            format_book_structure(analyzed),
            ocr_note,
            analyzed,
            chapter_texts,
            gr.update(choices=choices, value=choices[:1]),
        )

    progress(0.01, desc="Processing: checking the book's language matches what you entered...")
    check_language_mismatch(pdf_path, target_language)

    progress(0.02, desc="Processing: reading the PDF...")
    page_texts, used_ocr = extract_page_texts(pdf_path, target_language, progress)

    client = _client()

    progress(0.5, desc="Analyzing: finding the book's chapters...")
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
        progress(
            0.55 + 0.4 * (i / len(chapters_meta)),
            desc=f"Analyzing: chapter {i + 1}/{len(chapters_meta)} ({c['chapter_id']})...",
        )
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

    progress(1.0, desc="Ready")
    ocr_note = (
        "*OCR was used — no text layer was found in this PDF (it's likely a scan).*"
        if used_ocr
        else "*Text was read directly from the PDF — no OCR needed.*"
    )
    choices = [f"{c['chapter_id']} — {c['title']}" for c in chapters_meta]

    save_cached_analysis(
        sha256,
        {
            "chapters_meta": chapters_meta,
            "chapters": analyzed,
            "chapter_texts": chapter_texts,
            "used_ocr": used_ocr,
            "target_language": target_language,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        },
    )

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

    questions = test["questions"]
    if len(questions) > MAX_QUESTIONS:
        gr.Warning(
            f"Generated {len(questions)} questions, but only the first {MAX_QUESTIONS} can be shown here "
            "-- try selecting fewer chapters at once for a shorter test."
        )
        questions = questions[:MAX_QUESTIONS]
        test = {**test, "questions": questions}

    slot_updates = []
    for i in range(MAX_QUESTIONS):
        if i < len(questions):
            slot_updates.extend(_question_slot_updates(questions[i]))
        else:
            slot_updates.extend(_blank_question_updates())

    return (
        test,
        gr.update(value=f"## {test['class_or_week_label']}"),
        gr.update(visible=True),  # submit_btn
        gr.update(visible=False, value=""),  # results_output
        *slot_updates,
    )


def check_access_code(code: str):
    """Reveals the app column and hides the gate column on a valid code; otherwise shows an
    error and leaves the app hidden. Visibility updates apply per browser session (Gradio's
    normal per-client state), so this is a lightweight session gate, not real authentication --
    by design (see LITORA_ACCESS_CODES above), it doesn't need to survive a determined attacker."""
    valid_codes = _parse_access_codes(LITORA_ACCESS_CODES)
    if (code or "").strip() in valid_codes:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False, value="")
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True, value="❌ Invalid access code."),
    )


# Mobile polish only -- no layout restructuring, just: no horizontal scroll, long text (chapter
# titles, question prompts, word banks) wraps instead of overflowing, and the two side-by-side
# input rows stack vertically on narrow screens instead of squeezing. Tested against 320/375/430px
# widths (iPhone SE / most phones / larger phones).
MOBILE_CSS = """
.gradio-container { overflow-x: hidden !important; max-width: 100% !important; }
.gradio-container * { word-break: break-word; overflow-wrap: break-word; }

@media (max-width: 480px) {
    .gradio-container { padding-left: 8px !important; padding-right: 8px !important; }
    #upload-row, #test-meta-row { flex-direction: column !important; }
    #upload-row > *, #test-meta-row > * { width: 100% !important; min-width: 0 !important; }
    #chapter-select label { font-size: 0.9em; line-height: 1.3; }
    .gradio-container button { min-height: 44px; }
    .gradio-container h1 { font-size: 1.3em !important; }
}
"""

with gr.Blocks(title="Litora", css=MOBILE_CSS) as demo:
    with gr.Column(visible=True) as gate_column:
        gr.Markdown("# Litora\nEnter your access code to continue.")
        access_code_input = gr.Textbox(label="Access code", type="password")
        access_code_btn = gr.Button("Enter", variant="primary")
        access_code_error = gr.Markdown(visible=False)

    with gr.Column(visible=False) as app_column:
        gr.Markdown(
            "# Litora — book to test\n"
            "Upload a coursebook PDF. Litora reads it (OCR if it's a scan), finds the real teaching "
            "chapters, extracts vocabulary/grammar per chapter, and generates a test from whichever "
            "chapters you pick."
        )

        analyzed_state = gr.State()
        chapter_texts_state = gr.State()

        with gr.Row(elem_id="upload-row"):
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

        chapter_select = gr.CheckboxGroup(label="2. Chapters to test", choices=[], elem_id="chapter-select")
        with gr.Row(elem_id="test-meta-row"):
            week_label_input = gr.Textbox(label="Week/class label", placeholder="e.g. Week 1")
            instruction_language_input = gr.Textbox(
                label="Instruction language", value="", placeholder="e.g. English, Kazakh"
            )
        generate_btn = gr.Button("3. Generate test", variant="primary", interactive=False)

        test_state = gr.State()
        test_header = gr.Markdown()
        question_groups, question_labels, question_texts = [], [], []
        for _ in range(MAX_QUESTIONS):
            with gr.Group(visible=False) as grp:
                lbl = gr.Markdown()
                text = gr.Textbox(show_label=False, visible=False, placeholder="Your answer")
            question_groups.append(grp)
            question_labels.append(lbl)
            question_texts.append(text)

        submit_btn = gr.Button("Submit answers", variant="primary", visible=False)
        results_output = gr.Markdown(visible=False)

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
        slot_outputs = [
            c for i in range(MAX_QUESTIONS) for c in (question_groups[i], question_labels[i], question_texts[i])
        ]
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
            outputs=[test_state, test_header, submit_btn, results_output] + slot_outputs,
        )
        submit_btn.click(
            grade_test,
            inputs=[test_state] + question_texts,
            outputs=[results_output],
        )

    access_code_btn.click(
        check_access_code,
        inputs=[access_code_input],
        outputs=[gate_column, app_column, access_code_error],
    )
    access_code_input.submit(
        check_access_code,
        inputs=[access_code_input],
        outputs=[gate_column, app_column, access_code_error],
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
    if CACHE_DATASET_REPO and _hf_token():
        print(f"[startup] Analyzed-book cache enabled -- writing to dataset repo {CACHE_DATASET_REPO}.", file=sys.stderr)
    elif CACHE_DATASET_REPO:
        print(
            f"[startup] LITORA_CACHE_DATASET_REPO={CACHE_DATASET_REPO!r} is set but no HF_TOKEN/HUGGINGFACE_TOKEN "
            "secret was found -- cache reads will be attempted but writes will be skipped.",
            file=sys.stderr,
        )
    else:
        print(
            "[startup] LITORA_CACHE_DATASET_REPO is not set -- every upload will be analyzed fresh "
            "(no book-analysis caching).",
            file=sys.stderr,
        )

    access_codes = _parse_access_codes(LITORA_ACCESS_CODES)
    if not LITORA_ACCESS_CODES:
        print(
            f"[startup] LITORA_ACCESS_CODES is not set -- only the {len(FALLBACK_ACCESS_CODES)} baked-in "
            "fallback code(s) work right now. Set LITORA_ACCESS_CODES=\"CODE1,CODE2,...\" as a Space secret "
            "to add more without editing code.",
            file=sys.stderr,
        )
    else:
        print(f"[startup] Access-code gate enabled -- {len(access_codes)} valid code(s) configured.", file=sys.stderr)
    demo.queue().launch()
