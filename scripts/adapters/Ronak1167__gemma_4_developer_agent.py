"""Adapter for Ronak1167/gemma-4-developer-agent ("Bounded Rationality in Autonomous Coding Agents", Gemma 4 12B).

Not eligible: the repository holds the agent bundle, packaging scripts and the paper, but no result files.
"""

REPO = "Ronak1167/gemma-4-developer-agent"

PAPER = "paper/gemma4_swe_agent_research_paper.md"

CARD = {
    "eligible": False,
    "reason": "no per-task outcomes: the only data file (starter/official/tasks.jsonl, 129 records) is the competition's task "
              "manifest (instance_id, repo, base_commit, patch, test_patch, problem_statement), not results; the paper gives "
              "one hidden-test leaderboard score and prose about four attempts",
    "licence": {"spdx": "Apache-2.0", "evidence": "no LICENSE file in the snapshot; README.md line 7 is a badge "
                                                  "'[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-"
                                                  "blue.svg)](LICENSE)' whose link target LICENSE is missing"},
    "models": [],
    "harness": None,
    "summary": "Gemma 4 12B agent with a 'bounded-rationality' prompt and budget envelope (1,024-token thinking budget, "
               "action-only turns, 28 tool calls / 5 minutes); the repository has the bundle, scripts and paper only.",
    "evidence": {
        "model": "README.md: 'our **Bounded-Rationality Architecture** for **Gemma 4 12B** (`gemma-4-12b-it-qat-w4a16-ct`)'",
        "harness": "README.md: 'achieving an official Kaggle public score of **`0.03`** (Candidate v23, Submission ID "
                   "`58187123`)' - the only scored result is the hidden-test leaderboard",
        "systems": "none with data",
        "replicates": "none",
        "task_sets": "README.md repository tree: 'official/tasks.jsonl       # Official 129 SWE-bench benchmark tasks'",
    },
    "decisions": [
        "the four attempts described in the paper's prose are not turned into records: they are illustrative examples chosen by "
        "the author, and the result folders they come from (reports/version-18-smoke-test-output, reports/gpu-diagnostic-v22-"
        "output, named in scripts/) are not in the repository",
        "the README badge is taken as the licence statement, as instructed for this source; the licence file it links to is absent",
    ],
    "excluded": [
        "starter/official/tasks.jsonl: competition task manifest (129 task definitions including reference patches), no outcomes",
        PAPER + " 6.2: 'In baseline evaluation (Version 18 smoke test), the agent resolved task `rich_4077`' - prose, one task",
        PAPER + " 6.3: 'diagnostic runs (Candidate v22 across 3 tasks)': fastapi_15589 crashed with ContextWindowExceededError, "
                "rich_4077 exceeded the 32k window, rich_3944 exhausted the 15-minute session limit - prose, all three failed",
        PAPER + " 6.1 table of four shards (32, 33, 32, 32 tasks, 'Valid'): a sharding plan, no outcomes",
        "README.md 'Official Score 0.03': Kaggle hidden-test leaderboard",
    ],
    "dropped": {},
    "flags": ["no_outcome_data"],
    "reported_numbers": [],
}


def attempts(src):
    return
    yield
