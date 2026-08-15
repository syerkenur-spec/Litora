#!/usr/bin/env python3
"""
Quick smoke test for generate_quiz (see quiz.py).

Run after ingest.py has loaded chapter-01 into Supabase:

    python test_quiz.py

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) in the
environment.
"""

import json

from dotenv import load_dotenv

load_dotenv()

from quiz import generate_quiz

print("=== generate_quiz(['chapter-01'], 'mixed', 5) ===")
result = generate_quiz(["chapter-01"], "mixed", 5)
print(json.dumps(result, indent=2, ensure_ascii=False))

print()
print("=== generate_quiz(['chapter-02'], 'mixed', 5)  (expect a refusal) ===")
result = generate_quiz(["chapter-02"], "mixed", 5)
print(json.dumps(result, indent=2, ensure_ascii=False))
assert result["status"] == "not_available", "expected chapter-02 to be refused, not generated"
print()
print("OK: chapter-02 was correctly refused instead of generating anything.")
