# rag_pipeline

Chunks book_analyzer's chapter JSON, embeds each chunk, stores it in Supabase
(pgvector), and serves `generate_quiz()` for the test-generator/orchestrator
side of Litora to call.

## Setup

1. Create a Supabase project (or use an existing one) and enable pgvector by
   running `schema.sql` in the SQL editor.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` (or export the vars another way) and fill
   in `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and `VOYAGE_API_KEY`.

## Ingest

```
python ingest.py --chapter-files ../book_analyzer/chapter-01.json
```

`--book-structure` is opt-in (omitted by default) — see the design note below
on why. Pass it explicitly if you want book-analyzer's extraction ingested
too: `--book-structure ../book_analyzer/book-structure.json`.

Chunking rule: one chunk per vocabulary word, one per grammar point, one per
reading passage — never merged. Re-running is safe; rows upsert on
`(chapter_id, chunk_type, content)`.

## Generate a quiz

```python
from quiz import generate_quiz

generate_quiz(["chapter-01"], "mixed", 5)
```

`topic_type` is `"vocabulary"`, `"grammar"`, or `"mixed"`. If any requested
chapter isn't in the `chunks` table yet, the whole call returns
`{"status": "not_available", "message": "..."}` instead of generating
anything — see `test_quiz.py`.

Run the smoke test:

```
python test_quiz.py
```

## Design notes

- **No LLM in the quiz path.** Questions are built by template directly from
  each chunk's stored word/sentence/rule text, so there's no way for a
  question or its answer to contain anything not already in that chunk —
  stronger than asking an LLM not to invent things.
- **Retrieval is a metadata filter, not a similarity search.** `generate_quiz`
  wants "every vocabulary/grammar chunk for these chapters," which a
  `chapter_id`/`chunk_type` filter answers exactly; a nearest-neighbor search
  could miss chunks or pull in unrelated ones. The embeddings are stored so a
  future "ask the book a question" feature can do real semantic search —
  quizzes don't need it.
- **Two source shapes; `book-structure.json` is opt-in, not ingested by
  default.** `chapter-01.json` quotes the source text directly and leaves
  `definition`/`page` empty where the book doesn't provide one.
  `book-structure.json` (book-analyzer's LLM extraction) has no
  `example_sentences` or `page` fields, and its vocabulary `definition`s are
  the model's own gloss, not something quoted from the book — a quiz question
  built from it could contain a definition that isn't strictly "from the
  book." It also uses a different `chapter_id` for the same real chapter
  (`chapter1_our_favourites` vs. `chapter-01`), so ingesting both would put
  the same unit in the knowledge base twice under two IDs. For chapter 1,
  only `chapter-01` (the strict, source-quoted version) is in the knowledge
  base.
