---
title: Litora
emoji: 📘
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
---

# Litora

Upload a language coursebook PDF. Litora reads it (OCR automatically if it's
a scan with no text layer), detects the real teaching chapters, extracts
vocabulary/grammar per chapter, and generates a test from whichever chapters
you select.

On upload, Litora also auto-detects and pre-fills "Language being taught"
from a quick sample of the PDF (editable if it guesses wrong; left blank
with a note if it can't tell). Before running the full analysis, it
double-checks that sample against whatever "Language being taught" says --
if they clearly don't match, it stops immediately with an error instead of
running the full OCR/analysis pipeline on the wrong assumption. "Analyze
book" and "Generate test" stay disabled until "Instruction language" is
filled in.

Self-contained: `app.py` duplicates the core logic from `../book_analyzer`
and `../test_generator` rather than importing them, so this directory can be
pushed to a Hugging Face Space on its own.

## Setup

1. Create a Space (SDK: Gradio) on huggingface.co, set visibility to
   **Private**.
2. In the Space's Settings -> Repository secrets, add `GEMINI_API_KEY`.
3. (Optional but recommended) Also add `GOOGLE_CLOUD_VISION_API_KEY` -- see
   "Scanned PDFs and OCR speed" below.
4. Push this directory's contents to the Space's git repo:

   ```
   cd web
   git init
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git add .
   git commit -m "Deploy Litora web app"
   git push --force space main
   ```

   (`--force` is safe here since the Space repo is otherwise empty; drop it
   on later pushes.) Alternatively, upload `app.py`, `requirements.txt`, and
   `README.md` via the Space's web UI.

## Caching analyzed books

By default every upload re-runs OCR and Gemini analysis from scratch, even
for a book that's already been analyzed. To skip that for a repeat upload of
the exact same PDF (matched by SHA-256), set:

1. `LITORA_CACHE_DATASET_REPO` -- a private Hugging Face Hub **dataset** repo
   to cache into, e.g. `yourname/litora-book-cache` (it's created
   automatically on first write if it doesn't exist).
2. An `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) Space secret with **write** access
   to that repo -- generate one at huggingface.co/settings/tokens. Spaces
   don't get a write-capable token by default, so this has to be added
   explicitly even if the Space itself lives under the same account.

Cached results are stored as one JSON file per book
(`book-cache/<sha256>.json`) in that dataset repo, which persists
independently of the Space's own container -- it survives restarts,
rebuilds, and sleep/wake cycles without needing the paid Persistent Storage
add-on. If either variable is missing, caching is silently skipped and every
upload is analyzed fresh, exactly as before.

## Scanned PDFs and OCR speed

Most coursebook PDFs are scans (page-images, no real text underneath) --
Litora detects this automatically and falls back to OCR. By default that
runs locally via `easyocr` on the Space's own CPU, which is free but slow:
a 100+ page scan can take close to an hour on the free CPU tier.

Setting `GOOGLE_CLOUD_VISION_API_KEY` (a Cloud Vision API key from a Google
Cloud project, with the Vision API enabled) switches OCR to Google Cloud
Vision instead -- the same book typically finishes in well under a minute,
since it runs on Google's infrastructure rather than the Space's CPU. It's
pay-per-page (a few dollars/month at most for occasional use) and needs its
own Google Cloud account, separate from your Gemini API key. Without this
secret set, the app works exactly as before (free local OCR, just slower).

## Local testing

```
pip install -r requirements.txt
python app.py
```

Requires `GEMINI_API_KEY` in the environment or a `.env` file (same
convention as `book_analyzer`/`test_generator`).
