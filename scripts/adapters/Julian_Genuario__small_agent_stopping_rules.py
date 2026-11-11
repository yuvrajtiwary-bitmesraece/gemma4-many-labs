"""Adapter for Julian-Genuario/small-agent-stopping-rules ("When should a small repair agent stop testing?").

Not eligible: every task is a seeded one-token mutant of rich 14.3.4 or click 8.5.0, not a competition task.  The bug
ids look like `rich-000`; they are mutant indices and must not be read as development-set ids, so nothing is yielded.
"""

REPO = "Julian-Genuario/small-agent-stopping-rules"

CARD = {
    "eligible": False,
    "reason": "no attempt on the competition's development set: the 120 tasks are one-token bugs that the lab seeded "
              "itself into rich 14.3.4 and click 8.5.0 (ids rich-000 ... rich-059, click-000 ... click-059, mutant "
              "indices, not competition task numbers), repaired by Gemma 4 E2B (llama.cpp, single function, up to 3 "
              "turns) and by two deterministic search baselines, and graded by the projects' own test suites with "
              "held-out failing tests (README.md: 'Bugs are seeded, not mined'; 'No data from the Kaggle competition is "
              "used or included.')",
    "licence": {"spdx": "MIT",
                "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Julián Genuario'; README.md: 'License: MIT (see "
                            "LICENSE). Upstream excerpts in `data/` and `results/` keep their licenses (see NOTICE.md).'; "
                            "NOTICE.md: rich excerpts MIT, click excerpts BSD-3-Clause, 'Everything else in this "
                            "repository is MIT (see LICENSE).'"},
    "models": [],
    "harness": None,
    "summary": "A study of stopping rules for a repair loop: 120 seeded mutants (60 rich, 60 click), one visible failing "
               "test per bug, the other failing tests held out; four validation rules compared for a token-search "
               "repairer and for Gemma 4 E2B on a 4-core CPU.",
    "evidence": {
        "model": "README.md: 'Gemma 4 E2B (Google's QAT Q4_0 GGUF) served by llama.cpp on a 4-core CPU, up to 3 turns "
                 "with test feedback'",
        "harness": "the lab's own harness (heldout/runner.py, heldout/evaluate.py: pytest on the mutated checkout); the "
                   "competition harness is not used",
        "systems": "results/llm_e2b.jsonl (tag e2b, 120 bugs, temperature 0.0) and results/llm_e2b-t08.jsonl (tag "
                   "e2b-t08, 60 click bugs, retries at temperature 0.8); results/search_oracle-line.jsonl and "
                   "results/search_sbfl.jsonl are non-LLM search baselines",
        "replicates": "none",
        "task_sets": "writeup.md s2.1: 'rich is one of the four repositories in the competition's public task set; click "
                     "is not ... No competition data was used.'; data/bugs_rich.jsonl, data/bugs_click.jsonl (60 bugs "
                     "each: file, function, edit, failing tests, visible test, diff)",
    },
    "decisions": [
        "seeded (synthetic, mutated) tasks are outside the public development set by the inclusion rules; nothing is "
        "yielded",
        "ids of the form rich-NNN would pass the harmonizer's id pattern (rich_0 ... rich_59) although they are mutant "
        "indices; they are deliberately not yielded and not rewritten",
    ],
    "excluded": [
        "results/llm_e2b.jsonl (121 records, 120 bugs), results/llm_e2b-t08.jsonl (60 click bugs): Gemma 4 E2B repair "
        "attempts on seeded mutants of rich and click",
        "results/search_oracle-line.jsonl, results/search_sbfl.jsonl (120 records each): deterministic single-token "
        "search, no model",
        "data/bugs_*.jsonl, data/seedlog_*.jsonl, data/coverage_*.json, data/tests_*.json, data/baseline_*.json: the "
        "seeded bugs, the mutants tried, per-test coverage and test lists",
        "results/manual_review.json, results/review_candidates.md, results/summary.md: manual equivalence review and "
        "generated tables",
    ],
    "dropped": {},
    "flags": ["synthetic_tasks_only"],
    "sampling": "temperature 0.0 on the first turn (results/llm_e2b.jsonl attempts[].llm.temperature); run e2b-t08 "
                "retries at temperature 0.8 (README.md: '--retry-temperature 0.8'); not competition tasks",
    "leaderboard": "none",
    "controls": [],
    "reported_numbers": [
        {"what": "seeded bugs", "value": "120 (60 rich, 60 click)", "where": "results/summary.md; writeup.md Table 2",
         "adapter": "not applicable (nothing yielded)", "match": True},
    ],
}


def attempts(src):
    return
    yield
