# book-analyzer

Implements the `book-analyzer` agent spec from `../book-analyzer.md`: reads an
uploaded coursebook chapter by chapter and extracts vocabulary, grammar
points, and a CEFR-style difficulty estimate as structured JSON — no prose
summaries, and chapters it can't confidently read are flagged instead of
guessed at.

## Setup

```
pip install -r requirements.txt
```

Requires an Anthropic API key: set `ANTHROPIC_API_KEY`, or run `ant auth login`
if you have the Anthropic CLI installed.

## Usage

Put one `.txt` or `.md` file per chapter in a directory, named so filename
order matches book order (e.g. `01-intro.md`, `02-greetings.md`, ...):

```
python analyze_book.py sample_chapters
```

Outputs:

- `book-structure.json` — one structured entry per chapter (same shape every
  time, per the spec's consistency rule)
- `flagged-chapters.log` — one line per chapter the model flagged as unclear,
  with the reason

Override the output paths with `-o` / `--log`.

## Try it

`sample_chapters/` has three example chapters, including one deliberately
garbled one, to show both the normal extraction path and the unclear-flagging
path:

```
python analyze_book.py sample_chapters -o /tmp/book-structure.json --log /tmp/flagged-chapters.log
```
