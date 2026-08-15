# orchestrator

Implements the pipeline-integrity part of the `orchestrator` agent spec from
`../orchestrator.md`: "never silently drop the book-analyzer -> test-generator
-> comprehension-reporter pipeline order." `run_pipeline.py` runs all three
agents for one week's cycle, each as its own subprocess, feeding one stage's
output file straight into the next stage's input:

```
book_analyzer          test_generator              comprehension_reporter
chapters/ ──────▶ book-structure.json ──────▶ test.json ──────▶ comprehension-report.json
                                                    ▲
                                        + your --results (real grades)
```

The rest of the spec's role — protecting single-teacher/single-book MVP
scope and keeping `notes/architecture.md` in sync with real stack decisions —
is a human/process responsibility this script doesn't try to automate. What
it does automate is the one piece of that role expressible as code: the
"Reports" line (`a short "what changed / what's next" note ... appended to
notes/todo.md`), which it does at the end of every run.

## Why grading isn't automated

comprehension_reporter needs to know which questions each student actually
got right or wrong. No agent in the spec produces that — it only exists once
real students take the test — so this script can't manufacture it without
inventing student performance. You supply it yourself as one or more
`--results` files:

```json
{
  "results": [
    { "student": "Ana", "question_id": "w1q1", "correct": true }
  ]
}
```

`question_id` must match an id from the `test.json` this run produced (or
reused) — the script merges your `--results` file with that test's
`questions` into the "graded test" shape `comprehension_reporter` expects,
and errors out if a `question_id` doesn't exist on the test.

## Setup

```
pip install -r ../book_analyzer/requirements.txt
pip install -r ../test_generator/requirements.txt
```

(`comprehension_reporter` has no dependencies.) Requires `ANTHROPIC_API_KEY`
for the book_analyzer/test_generator stages.

## Usage

First run for a book (analyzes chapters + generates week 1's test + reports
on week 1's graded results, all in one call):

```
python run_pipeline.py \
    --chapters-dir ../book_analyzer/sample_chapters \
    --chapters 01-greetings --label "Week 1" \
    --results week1-results.json \
    --class-label "Grade 7A" \
    --work-dir pipeline_out
```

Every artifact lands in `--work-dir` (default `pipeline_out/`):
`book-structure.json`, `flagged-chapters.log`, `test.json`, `answer-key.json`,
`graded-<results-file-stem>.json` (the merged input each `--results` file
became), and `comprehension-report.json`.

Later weeks: the book doesn't need re-analyzing, so pointing `--work-dir` at
the same directory automatically reuses its `book-structure.json` (skips
book_analyzer) and just runs test_generator + comprehension_reporter for the
new week:

```
python run_pipeline.py --chapters 02-daily-routine --label "Week 2" \
    --results week2-results.json --class-label "Grade 7A" --work-dir pipeline_out
```

Pass `--extra-graded pipeline_out/graded-week1-results.json` to fold prior
weeks into the trend alongside this week's, the same way you'd hand multiple
files to `comprehension_reporter` directly.

Other flags:

- `--reanalyze` / `--regenerate-test` — force re-running a stage even if its
  output already exists in `--work-dir`.
- `--skip-report` — stop after test_generator, for weeks where you just want
  the test and aren't ready to grade yet.
- `--threshold` / `--min-attempts` — passed straight through to
  comprehension_reporter.
- `--notes-dir` — where the session note gets appended (default: `../notes`).

## Pipeline order is structural, not just a rule

Each stage function checks for the artifact the previous stage must have
produced, and raises a clear error naming the missing stage if it's absent —
e.g. asking for a report with no `test.json` on disk fails with "test_generator
must run before comprehension_reporter" rather than silently skipping ahead.
Stages 1-2 can be *skipped on a given run* by reusing an existing artifact,
but that artifact still has to have come from that stage at some point.

## Try it

Without an API key, you can still exercise the stage-3 wiring (merge +
comprehension_reporter call) by pointing `--work-dir` at a directory that
already has a `book-structure.json` and `test.json` in it (e.g. copy
`../book_analyzer/book-structure.json` and hand-write a small `test.json`),
then running with matching `--results`. Both earlier stages will report
"reusing existing" and the run will still go end to end.

With a real `ANTHROPIC_API_KEY`, the command under Usage above runs the full
chain live.
