"""Adapter for DesignMaster-Solutions/where-the-fix-lives ("Where the Fix Lives: Lexical anchors, not graph structure,
decide bug localization in the Gemma 4 Developer Agent benchmark, and what that means for small local models",
Alan De Bortolo, Design Master Solutions LLC).

Not eligible: a localisation study.  Its only per-task model output, results/llm_predictions.jsonl, holds single-shot
Gemma 4 E4B guesses of which functions a fix touches (three prompt conditions x 115 tasks); no agent run, no patch,
no resolved / not-resolved outcome.

It shares its main title with the registered source Haresh276/where-the-fix-lives but is different work by a
different author (see CARD["decisions"]), so it is NOT a duplicate of that source.
"""

REPO = "DesignMaster-Solutions/where-the-fix-lives"

CARD = {
    "eligible": False,
    "reason": ("no per-task resolved outcomes: localisation study (hit@k of training-free rankers and of single-shot Gemma 4 E4B "
               "function predictions on 115 public tasks); no agent runs, no patches, no grading"),
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: 'Author: Alan De Bortolo, Design Master "
                            "Solutions LLC. License: Apache-2.0.'"},
    "models": [],
    "harness": None,
    "summary": ("Maps the gold patches of the 129 public tasks onto the provided call graphs (function-level labels for 115 tasks), "
                "compares training-free localisation rankers (BM25, embedding kNN, personalised PageRank, hybrid) and asks Gemma 4 "
                "E4B (Q8_0 GGUF, llama.cpp on a laptop, temperature 0, seed 0, one call per task and condition) to name the "
                "functions to edit from the file tree, BM25's top 20 or the hybrid's top 20."),
    "evidence": {
        "model": "paper/writeup.md s5: 'We serve Gemma 4 E4B (instruction-tuned, Q8_0 GGUF from `ggml-org/gemma-4-E4B-it-GGUF`) with "
                 "llama.cpp on an Apple M5 Max, temperature 0, at most 4,096 generated tokens'",
        "harness": "none: scripts/llm_localize.py sends one chat completion per (task, condition) and parses 'a JSON list of up to 5 "
                   "strings ... \"path/to/file.py::QualifiedName\"'; scripts/score_llm.py scores them against gold labels (hit@k)",
        "systems": "results/llm_predictions.jsonl column `condition`: tree, bm25_top20, hybrid_top20 (115 rows each)",
        "replicates": "none: paper/writeup.md s6 'RQ3 is a single model, quantization and prompt, run once at temperature 0, and "
                      "measures localization only, not repair'",
        "task_sets": "paper/writeup.md s2: 'Our function-level evaluation set therefore has 115 tasks ... and the file-level set has "
                     "128 tasks'",
    },
    "decisions": [
        "not eligible under the README rule 'metrics that are not a per-task resolved/not-resolved outcome (localisation hits, retrieval ranks ...)': results/llm_predictions.jsonl has keys instance_id, condition, preds, parsed, finish_reason, prompt_tokens, completion_tokens, reasoning_chars, secs; there is no outcome column and the gold labels are not shipped",
        "same title, different work - checked against the registered source Haresh276/where-the-fix-lives: (1) authors: 'Alan De Bortolo, Design Master Solutions LLC' here vs 'Haresh G. (Kaggle: Haresh276)' there; (2) subtitles and questions differ ('Lexical anchors, not graph structure, decide bug localization ...' vs 'Code-Graph Structure and the Triage Gap for Local Coding Agents', a SWE-bench Verified analysis plus two agent runs); (3) models and executions: single-shot E4B localisation calls through llama.cpp on a laptop here vs two Gemma 4 31B agent runs scored by the swegemma Evaluator on Kaggle 4xL4 there; (4) no file, table or number is shared (their data/gemma/own/*.csv hold resolved outcomes for 129 tasks, this repository holds no outcome at all) and neither repository mentions the other. Hence no possible_duplicate_of flag",
    ],
    "excluded": [
        "results/llm_predictions.jsonl: 345 rows = 115 tasks x 3 prompt conditions of Gemma 4 E4B function-localisation predictions (up to 5 'file.py::QualifiedName' guesses per row); localisation only",
        "results/summary.json, results/tables.md, figures/: aggregate structure statistics and hit@k / MRR of rankers and of the model's predictions with bootstrap intervals",
        "notebooks/where-the-fix-lives.ipynb: pipeline notebook without outcome data",
    ],
    "dropped": {},
    "flags": ["localisation_only", "no_per_task_outcomes", "same_title_as:Haresh276/where-the-fix-lives(different_work)"],
    "sampling": "temperature 0, seed 0, one completion per task and condition (scripts/llm_localize.py; paper/writeup.md s5-s6) - localisation calls, not agent runs",
    "leaderboard": "none",
    "controls": [],
    "reported_numbers": [
        {"what": "Gemma 4 E4B prediction rows", "value": "115 function-level tasks x 3 conditions", "where": "paper/writeup.md s2, s5; README.md",
         "adapter": "results/llm_predictions.jsonl: 345 rows, 115 distinct ids, 115 per condition (not yielded)", "match": True},
    ],
}


def attempts(src):
    return
    yield
