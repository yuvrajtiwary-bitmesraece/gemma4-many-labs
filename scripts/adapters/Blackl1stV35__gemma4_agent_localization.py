"""Adapter for Blackl1stV35/gemma4-agent-localization ("Lexical Search Is Hard to Beat: Seeded Graph and Embedding
Signals for Issue Localization by Small Offline Coding Agents").

Not eligible: a file-localisation study of six retrieval strategies.  No language model is run and there is no
resolved / not-resolved outcome.
"""

REPO = "Blackl1stV35/gemma4-agent-localization"

CARD = {
    "eligible": False,
    "reason": "no agent outcomes: the only data file (paper/results.json, 124 per-task records) holds file-localisation "
              "results (per task the rank at which each of six retrieval strategies first returns a file the reference "
              "patch edits, summarised as MRR and recall@k) without any language model (lexical_bm25, seed_only, "
              "graph_2hop, embedding_seed, hybrid_rrf, hybrid_bm25_graph); paper/writeup.md s6: 'We measure retrieval "
              "quality only and did not run a language "
              "model inside the study.' The only Gemma 4 results are three public-leaderboard scores (0.05, 0.01, 0.10), "
              "which are hidden-test aggregates",
    "licence": {"spdx": "MIT",
                "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Blackl1stV35'; README.md '## License' / 'MIT.'"},
    "models": [],
    "harness": None,
    "summary": "A CPU-only retrieval benchmark on the 124 public tasks whose gold files exist in the competition's code "
               "graph: BM25, issue-named seed symbols, 2-hop graph expansion, embedding similarity and two rank fusions, "
               "scored by MRR and recall@k of the reference patch's files; plus the lab's main-track agent configuration "
               "(Gemma 4 E4B) with no local runs.",
    "evidence": {
        "model": "no model in the study (README.md: 'Both scripts are deterministic and run on a CPU.'); "
                 "submission/agent.yaml declares 'model: gemma-4-e4b-it-qat-w4a16-ct' for the main-track submission; "
                 "colab/STATUS.md: 'CLOSED, inconclusive ... No model behavior was observed.'",
        "harness": "none: paper/study.py scores retrieval rankings; no patch is produced or tested",
        "systems": "paper/results_table.md: six retrieval methods (rows of Table 1)",
        "replicates": "none (deterministic scripts)",
        "task_sets": "paper/writeup.md s3.1: 'Five tasks are excluded because their patches only add new files, which leaves "
                     "124 tasks (FastAPI 64, Rich 46, requests 13, httpx 1).'",
    },
    "decisions": [
        "localisation / retrieval metrics are excluded by the inclusion rules; nothing is yielded",
        "the three leaderboard scores have no development-set counterpart in the repository, so no leaderboard pair is "
        "recorded",
    ],
    "excluded": [
        "paper/results.json per_task (124 records: id, gold files, n_seeds and, per retrieval method, the rank of the "
        "first gold file or null): localisation metrics, no model, no resolved outcome",
        "paper/results_table.md, paper/analysis.md, paper/extra.md, paper/ppr.md, paper/router.md: aggregate tables of the "
        "same retrieval scores",
        "submission/: agent configuration for the main track, no run results",
        "public-leaderboard scores 0.05 (baseline prompt), 0.01 (prompt following the paper's policy), 0.10 (third "
        "configuration) in paper/writeup.md s6: hidden-test aggregates",
    ],
    "dropped": {},
    "flags": ["localisation_only", "no_model_run"],
    "sampling": "none: no model run in the study (the unevaluated submission/agent.yaml sets temperature 0.2, top_p 0.95)",
    "leaderboard": "none usable: paper/writeup.md s6 states public-leaderboard scores 0.05, 0.01 and 0.10 for three "
                   "configurations, but no development-set run of any of them exists",
    "controls": [],
    "reported_numbers": [
        {"what": "usable tasks", "value": "124 of 129 (FastAPI 64, Rich 46, requests 13, httpx 1)",
         "where": "paper/writeup.md s3.1; paper/results.json coverage", "adapter": "not applicable (nothing yielded)",
         "match": True},
    ],
}


def attempts(src):
    return
    yield
