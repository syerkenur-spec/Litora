# comprehension-reporter

Implements the `comprehension-reporter` agent spec from
`../comprehension-reporter.md`: turns raw per-student, per-question test
results into a lightweight, per-class comprehension summary — each student
plus their weak topics only, based on their average across everything
they've attempted, never a single test.

## Why no Anthropic API call

[book-analyzer](../book_analyzer) and [test-generator](../test_generator)
both need language understanding (reading a chapter, writing test
questions), so they call Claude. This agent's job is arithmetic over data
those agents (plus grading, done elsewhere) already produced and labeled —
averaging scores and comparing them to a threshold. Doing that with an LLM
would risk hallucinated numbers for no benefit, so it's a small deterministic
script with no third-party dependencies.

## Input format: a "graded test"

Each input file is a student-facing test (the shape `test-generator` writes
to `test.json`, which already carries `topic` and `source_chapter_id` per
question) plus a `results` list saying who got what right:

```json
{
  "class_or_week_label": "Week 1",
  "questions": [
    { "id": "w1q1", "type": "multiple_choice", "topic": "vocabulary", "source_chapter_id": "01-greetings", "prompt": "...", "choices": ["..."] }
  ],
  "results": [
    { "student": "Ana", "question_id": "w1q1", "correct": true }
  ]
}
```

Grading (turning student answers into `correct: true/false`) is a separate
step upstream of this agent — see `sample_data/` for two ready-made weeks.

## Usage

```
python generate_report.py sample_data/week1-graded.json sample_data/week2-graded.json \
    --class-label "Grade 7A" -o comprehension-report.json
```

Pass as many graded-test files as you have (e.g. every test from a term) —
more files means a more reliable trend per topic.

Options:

- `--class-label` — label for the report (defaults to the input filenames).
- `--threshold` — average below this (0-1) flags a topic as weak (default `0.6`).
- `--min-attempts` — a topic needs at least this many questions *across all
  given files* before it can be flagged at all (default `2`). This is what
  enforces the spec's rule against judging a student off one bad test: a
  single missed question on a topic is never enough on its own.

## Output

One `comprehension-report.json`: every student in the input, each with a
`weak_topics` list (topics they're fine on are simply omitted):

```json
{
  "class_label": "Grade 7A",
  "students": [
    { "student": "Ana", "weak_topics": [] },
    { "student": "Tomas", "weak_topics": [{ "topic": "01-greetings::vocabulary", "average": 0.5, "questions_attempted": 4 }] },
    { "student": "Elif", "weak_topics": [{ "topic": "01-greetings::grammar", "average": 0.0, "questions_attempted": 4 }] },
    { "student": "Yuki", "weak_topics": [] }
  ]
}
```

`topic` is `<chapter_id>::<vocabulary|grammar>` — the finest-grained grouping
derivable from book-analyzer/test-generator's data without inventing a new
taxonomy.

## Try it

```
python generate_report.py sample_data/week1-graded.json sample_data/week2-graded.json --class-label "Grade 7A"
```

The sample data is built to show all three rules from the spec at once:

- **Ana** answers everything correctly both weeks → no weak topics.
- **Tomas** misses both vocabulary questions in week 1 but gets vocabulary
  fully right in week 2 → the *combined* average (2/4 = 0.5) still flags
  vocabulary, because a good week doesn't erase a real weak spot, but it's
  reported as an average+count, not a label.
- **Elif** misses every grammar question across both weeks (0/4) → clearly
  flagged, since it's a consistent trend, not a one-off.
- **Yuki** misses one grammar question in week 1 and has no other data →
  *not* flagged, because `--min-attempts 2` means one bad question never
  counts as a permanent judgment on its own.
