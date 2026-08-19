#!/usr/bin/env python3
"""
comprehension-reporter agent (see ../comprehension-reporter.md for the spec
this implements).

Aggregates raw per-student, per-question results from one or more graded
tests into a lightweight per-class comprehension summary: for each student,
which topics (chapter + vocabulary/grammar) they're weak on, based on their
average across everything they've attempted -- never a single test taken in
isolation.

Usage:
    python generate_report.py graded1.json [graded2.json ...] \\
        [-o comprehension-report.json] [--class-label "Grade 7A"] \\
        [--threshold 0.6] [--min-attempts 2]

Each <gradedN.json> is a graded test: a student-facing test (the shape
produced by test-generator's test.json) plus a "results" list of
{student, question_id, correct}. See sample_data/ for the expected shape and
README.md for how it's produced.

This agent is pure aggregation over already-graded, already-labeled data --
no language understanding is required, so unlike book-analyzer and
test-generator it does not call the Anthropic API.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_graded_test(path: Path) -> tuple[dict, list]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions_by_id = {q["id"]: q for q in data.get("questions", [])}
    results = data.get("results", [])

    missing = sorted({r["question_id"] for r in results} - set(questions_by_id))
    if missing:
        raise SystemExit(
            f"error: {path} results reference unknown question_id(s): {', '.join(missing)}"
        )
    return questions_by_id, results


def topic_key(question: dict) -> str:
    # specific_topic (e.g. "Possessive 's") gives a finer trend than the bare
    # vocabulary/grammar category; older test files without it fall back to topic.
    topic = question.get("specific_topic") or question["topic"]
    return f"{question['source_chapter_id']}::{topic}"


def aggregate(graded_files: list[Path]) -> dict[str, dict[str, list[int]]]:
    # student -> topic -> [correct_count, total_count]
    stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for path in graded_files:
        questions_by_id, results = load_graded_test(path)
        for r in results:
            question = questions_by_id[r["question_id"]]
            bucket = stats[r["student"]][topic_key(question)]
            bucket[1] += 1
            if r["correct"]:
                bucket[0] += 1

    return stats


def build_summary(
    stats: dict[str, dict[str, list[int]]], threshold: float, min_attempts: int
) -> list[dict]:
    students = []
    for student in sorted(stats):
        weak_topics = []
        for topic, (correct, total) in sorted(stats[student].items()):
            if total < min_attempts:
                # Not enough data to call this a trend -- skip rather than
                # judge the student on a single question or test.
                continue
            average = correct / total
            if average < threshold:
                weak_topics.append(
                    {
                        "topic": topic,
                        "average": round(average, 2),
                        "questions_attempted": total,
                    }
                )
        students.append({"student": student, "weak_topics": weak_topics})
    return students


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "graded_tests", nargs="+", type=Path, help="One or more graded-test JSON files"
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("comprehension-report.json"))
    parser.add_argument("--class-label", default=None, help='Class label, e.g. "Grade 7A"')
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Average (0-1) below which a topic is flagged as weak (default: 0.6)",
    )
    parser.add_argument(
        "--min-attempts",
        type=int,
        default=2,
        help="Minimum questions on a topic (across all given tests) before it can be flagged (default: 2)",
    )
    args = parser.parse_args()

    missing_files = [p for p in args.graded_tests if not p.is_file()]
    if missing_files:
        print(f"error: file(s) not found: {', '.join(str(p) for p in missing_files)}", file=sys.stderr)
        return 1

    stats = aggregate(args.graded_tests)
    students = build_summary(stats, args.threshold, args.min_attempts)

    label = args.class_label or ", ".join(p.stem for p in args.graded_tests)
    report = {"class_label": label, "students": students}

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    flagged = sum(1 for s in students if s["weak_topics"])
    print(f"Wrote comprehension report for {len(students)} student(s) to {args.output}", file=sys.stderr)
    print(f"{flagged} student(s) have at least one flagged topic", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
