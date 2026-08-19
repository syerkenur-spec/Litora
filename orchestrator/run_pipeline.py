#!/usr/bin/env python3
"""
orchestrator agent (see ../orchestrator.md for the spec this implements).

Runs the Litora pipeline end to end for one week, calling each stage as a
separate agent and passing its output straight into the next:

    book_analyzer  ->  book-structure.json
    test_generator ->  test.json + answer-key.json   (consumes book-structure.json)
    comprehension_reporter -> comprehension-report.json  (consumes test.json + graded results)

The spec's rule is "never silently drop the pipeline order" -- this script
enforces that structurally: each stage either produces the artifact the next
stage needs, or the run stops with an error naming the missing stage. Stages
1 and 2 can be skipped on a given run by pointing at an already-produced
artifact (e.g. the book was analyzed last week and hasn't changed), but the
artifact still has to exist -- there's no way to jump straight to stage 3
without a real book-structure.json and test.json behind it.

Grading itself (turning a student's raw answers into correct/incorrect) is
not part of this pipeline -- no agent in the spec produces it, and fabricating
it would mean inventing student performance. Supply it as one or more
--results files; this script merges them with the test.json questions into
the "graded test" shape comprehension_reporter expects.

Usage:
    python run_pipeline.py \\
        --chapters-dir ../book_analyzer/sample_chapters \\
        --chapters 01-greetings --label "Week 1" \\
        --results week1-results.json \\
        --class-label "Grade 7A" \\
        --work-dir pipeline_out

Requires GEMINI_API_KEY in the environment (or a .env file at the repo root)
for stages 1-2 (book_analyzer and test_generator call Gemini); stage 3
(comprehension_reporter) does not.
"""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK_ANALYZER_SCRIPT = ROOT / "book_analyzer" / "analyze_book.py"
TEST_GENERATOR_SCRIPT = ROOT / "test_generator" / "generate_test.py"
COMPREHENSION_REPORTER_SCRIPT = ROOT / "comprehension_reporter" / "generate_report.py"


def run_stage(cmd: list[str], stage: str) -> None:
    print(f"--- {stage} ---", file=sys.stderr)
    result = subprocess.run([sys.executable, *cmd])
    if result.returncode != 0:
        raise SystemExit(f"error: {stage} failed (exit code {result.returncode})")


def stage1_book_analyzer(args, book_structure: Path, flagged_log: Path) -> str:
    if book_structure.exists() and not args.reanalyze:
        print(f"--- book_analyzer: reusing existing {book_structure} ---", file=sys.stderr)
        return f"reused existing {book_structure.name}"

    if not args.chapters_dir:
        raise SystemExit(
            f"error: {book_structure} does not exist and --chapters-dir was not given -- "
            "cannot run book_analyzer without chapter files"
        )

    run_stage(
        [
            str(BOOK_ANALYZER_SCRIPT),
            str(args.chapters_dir),
            "-o", str(book_structure),
            "--log", str(flagged_log),
        ],
        "book_analyzer",
    )
    return f"analyzed {args.chapters_dir} -> {book_structure.name}"


def stage2_test_generator(args, book_structure: Path, test_path: Path, answer_key: Path) -> str:
    if test_path.exists() and not args.regenerate_test:
        print(f"--- test_generator: reusing existing {test_path} ---", file=sys.stderr)
        return f"reused existing {test_path.name}"

    if not book_structure.exists():
        raise SystemExit(
            f"error: {book_structure} does not exist -- book_analyzer must run before test_generator"
        )
    if not args.chapters:
        raise SystemExit("error: --chapters is required to generate a test")

    cmd = [
        str(TEST_GENERATOR_SCRIPT),
        str(book_structure),
        "--chapters", *args.chapters,
        "-o", str(test_path),
        "--answer-key", str(answer_key),
    ]
    if args.label:
        cmd += ["--label", args.label]
    run_stage(cmd, "test_generator")
    return f"generated test from chapters {args.chapters} -> {test_path.name}"


def merge_results_into_graded_test(test_path: Path, results_path: Path, work_dir: Path) -> Path:
    test_data = json.loads(test_path.read_text(encoding="utf-8"))
    raw_results = json.loads(results_path.read_text(encoding="utf-8"))

    if "results" not in raw_results:
        raise SystemExit(f'error: {results_path} is missing a "results" list')

    question_ids = {q["id"] for q in test_data["questions"]}
    unknown = sorted({r["question_id"] for r in raw_results["results"]} - question_ids)
    if unknown:
        raise SystemExit(
            f"error: {results_path} references question_id(s) not in {test_path}: {', '.join(unknown)}"
        )

    graded = {
        "class_or_week_label": test_data.get("class_or_week_label"),
        "questions": test_data["questions"],
        "results": raw_results["results"],
    }
    graded_path = work_dir / f"graded-{results_path.stem}.json"
    graded_path.write_text(json.dumps(graded, indent=2), encoding="utf-8")
    return graded_path


def stage3_comprehension_reporter(args, test_path: Path, work_dir: Path, report_path: Path) -> str:
    if not test_path.exists():
        raise SystemExit(
            f"error: {test_path} does not exist -- test_generator must run before comprehension_reporter"
        )
    if not args.results:
        raise SystemExit("error: --results is required to run comprehension_reporter (need graded scores)")

    graded_paths = [merge_results_into_graded_test(test_path, Path(r), work_dir) for r in args.results]
    graded_paths += [Path(p) for p in args.extra_graded]

    cmd = [
        str(COMPREHENSION_REPORTER_SCRIPT),
        *(str(p) for p in graded_paths),
        "-o", str(report_path),
        "--threshold", str(args.threshold),
        "--min-attempts", str(args.min_attempts),
    ]
    if args.class_label:
        cmd += ["--class-label", args.class_label]
    run_stage(cmd, "comprehension_reporter")
    return f"merged {len(args.results)} results file(s) (+{len(args.extra_graded)} prior) -> {report_path.name}"


def append_session_note(notes_dir: Path, summary_lines: list[str]) -> Path:
    notes_dir.mkdir(parents=True, exist_ok=True)
    todo_path = notes_dir / "todo.md"
    entry = [f"## {date.today().isoformat()}", "", "What changed:"]
    entry += [f"- {line}" for line in summary_lines]
    entry += [
        "",
        "What's next:",
        "- Review comprehension-report.json for flagged students",
        "- Continue the weekly cycle: new chapters -> test_generator -> grade -> comprehension_reporter",
        "",
    ]
    text = "\n".join(entry) + "\n"
    with todo_path.open("a", encoding="utf-8") as f:
        f.write(text)
    return todo_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chapters-dir", type=Path, default=None, help="Dir of chapter files for book_analyzer (only needed if book-structure.json doesn't exist yet)")
    parser.add_argument("--chapters", nargs="+", metavar="CHAPTER_ID", help="chapter_id(s) covered this week, for test_generator")
    parser.add_argument("--label", default=None, help='Week label for the test, e.g. "Week 1"')
    parser.add_argument("--results", nargs="+", default=[], metavar="PATH", help='Raw graded results for this week: {"results": [{"student","question_id","correct"}]}')
    parser.add_argument("--extra-graded", nargs="+", default=[], metavar="PATH", help="Previously-produced graded-test.json files to include for trend history")
    parser.add_argument("--class-label", default=None, help='Class label for the comprehension report, e.g. "Grade 7A"')
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-attempts", type=int, default=2)
    parser.add_argument("--work-dir", type=Path, default=Path("pipeline_out"))
    parser.add_argument("--notes-dir", type=Path, default=ROOT / "notes")
    parser.add_argument("--reanalyze", action="store_true", help="Re-run book_analyzer even if book-structure.json already exists")
    parser.add_argument("--regenerate-test", action="store_true", help="Re-run test_generator even if test.json already exists")
    parser.add_argument("--skip-report", action="store_true", help="Stop after test_generator; don't require --results yet")
    args = parser.parse_args()

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    book_structure = work_dir / "book-structure.json"
    flagged_log = work_dir / "flagged-chapters.log"
    test_path = work_dir / "test.json"
    answer_key = work_dir / "answer-key.json"
    report_path = work_dir / "comprehension-report.json"

    summary = []
    summary.append(stage1_book_analyzer(args, book_structure, flagged_log))
    summary.append(stage2_test_generator(args, book_structure, test_path, answer_key))

    if args.skip_report:
        print("--skip-report set: stopping after test_generator", file=sys.stderr)
    else:
        summary.append(stage3_comprehension_reporter(args, test_path, work_dir, report_path))

    todo_path = append_session_note(args.notes_dir, summary)
    print(f"Appended session note to {todo_path}", file=sys.stderr)
    print("Pipeline complete:", file=sys.stderr)
    for line in summary:
        print(f"  - {line}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
