# test-generator

Implements the `test-generator` agent spec from `../test-generator.md`: takes
the week's selected chapters plus the `book-structure.json` produced by
[book-analyzer](../book_analyzer), and generates a short, mixed-format
vocab/grammar test calibrated to those chapters' actual difficulty — not a
generic "intermediate" test.

## Setup

```
pip install -r requirements.txt
```

Requires a Gemini API key (free tier available at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)): set
`GEMINI_API_KEY`, or put it in a `.env` file at the repo root (see
`../.env.example`).

## Usage

Point it at a `book-structure.json` (from book-analyzer) and list the
`chapter_id`s covered this week:

```
python generate_test.py ../book_analyzer/book-structure.json --chapters chapter1_our_favourites --label "Week 1"
```

Outputs:

- `test.json` — the student-facing test (questions only, no answers)
- `answer-key.json` — answers keyed by question id, kept in a separate file
  from the student version

Override the output paths with `-o` / `--answer-key`. `--label` sets the
class/week label shown on the test; it defaults to the chapter id list if
omitted.

Questions mix `multiple_choice`, `fill_in_blank`, and `short_answer` types
covering both `vocabulary` and `grammar` topics, sized for weekly/daily use
(roughly 8-12 questions) rather than a final exam.

## Try it

Using book-analyzer's sample output against its sample chapter:

```
python ../book_analyzer/analyze_book.py ../book_analyzer/sample_chapters -o /tmp/book-structure.json --log /tmp/flagged-chapters.log
python generate_test.py /tmp/book-structure.json --chapters 01-greetings --label "Week 1" -o /tmp/test.json --answer-key /tmp/answer-key.json
```
