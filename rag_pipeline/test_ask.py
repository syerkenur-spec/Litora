#!/usr/bin/env python3
"""
Quick smoke test for ask() (see ask.py).

Run after ingest.py has loaded chapter-01 into Supabase:

    python test_ask.py

Requires SUPABASE_URL, SUPABASE_SERVICE_KEY (or SUPABASE_KEY), and
VOYAGE_API_KEY in the environment.
"""

from dotenv import load_dotenv

load_dotenv()

from ask import ask

QUESTIONS = [
    "What does the word 'singer' mean in Chapter 1, and can you give an example sentence?",
    "What is the boiling point of water in degrees Celsius?",
]

for question in QUESTIONS:
    result = ask(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"   (status={result['status']}, sources={result['sources']})")
    print()

assert result["status"] == "not_found", "expected the unrelated question to be refused, not answered"
print("OK: the unrelated question was correctly refused instead of answering from outside knowledge.")
