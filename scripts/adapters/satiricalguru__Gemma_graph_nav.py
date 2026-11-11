"""Adapter for satiricalguru/Gemma-graph-nav ("Who Should Walk the Graph?").

Not eligible: a function-level localisation study.  No patch is generated and no test is run, so there is no
resolved / not-resolved outcome.
"""

REPO = "satiricalguru/Gemma-graph-nav"

CARD = {
    "eligible": False,
    "reason": "localisation study without resolved/not-resolved outcomes: the 16 per-task files "
              "(results/runs/<method>/<model>/seed<k>/tasks_public.jsonl, 1,973 records on the 129 public tasks, 122 with "
              "function-level gold labels) hold a ranked list of code-graph symbols and Recall@k / Acc@k against the "
              "symbols the reference patch changes, for Gemma 4 E2B, E4B, 12B and 31B and three no-LLM baselines; no "
              "patch is generated and no test is run (README.md: 'localization only (no patch generation)'; "
              "paper/paper.md s4: 'Not run. Agentless and no-retrieval for E4B and 12B; downstream repair (H4); the "
              "external-benchmark experiment.')",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License and data': 'Code: "
                            "**Apache 2.0**'; paper/paper.md abstract: 'Code, prompts, per-task results and statistics are "
                            "released (Apache-2.0).'"},
    "models": [],
    "harness": None,
    "summary": "Six localisation methods (BM25, static graph, personalised PageRank 'APN', model-driven navigation 'MDN', "
               "gated delegation 'GDN', Agentless-style, no-retrieval) at four Gemma 4 sizes, scored by Recall@k of the "
               "functions the reference fix changes; one run per configuration at temperature 0, seed 1234.",
    "evidence": {
        "model": "results/runs/*/gemma4_*/seed1234/meta.json: e.g. \"name\": "
                 "\"/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2\"; README.md: 'E2B/E4B/12B: "
                 "Apple M2 16 GB, Ollama 0.35.0, Q4 QAT ... 31B: `gemma-4-31b-it-qat-w4a16-ct`, vLLM on Kaggle L4x4'",
        "harness": "none: the competition harness is not used; src/ggd/tools.py is 'a faithful replica of the competition's "
                   "`search_similar_code` / `get_code_neighbors` / `get_code_subgraph`' (README.md) and the score is "
                   "src/ggd/metrics.py Recall@k / Acc@k",
        "systems": "README.md results table: method x model, '122 tasks, single run, T = 0'",
        "replicates": "none: README.md 'Single run per configuration'",
        "task_sets": "docs/EXPERIMENT_LOG.md: '122/129 tasks have function/class gold; 448/499 gold symbols in official "
                     "graph; 65/129 tasks have no anchor'",
    },
    "decisions": [
        "per-task Recall@k / Acc@k are localisation metrics, which the inclusion rules exclude; nothing is yielded",
        "the outcome-like key `error` is null in all 1,973 records and `stop` is the navigation stop reason (done, final, "
        "budget, final_empty, invalid_calls), not a test result",
    ],
    "excluded": [
        "results/runs/*/*/seed*/tasks_public.jsonl (16 files, 1,973 records): ranked symbol lists with R@1..R@10 and "
        "Acc@1..Acc@10, no resolved outcome",
        "results/runs/{apn,apn_cv,bm25,static_graph}/no_llm: no-LLM baselines",
        "results/tables/*, results/tuning/*, results/audit/graph_audit.json: aggregates and a graph audit",
        "results/sanity/*.json: five auto-graded capability probes per model (not competition tasks)",
    ],
    "dropped": {},
    "flags": ["localisation_only"],
    "sampling": "temperature 0.0, top_p 1.0, seed 1234, thinking off, one run per configuration "
                "(results/runs/*/gemma4_*/seed1234/meta.json; docs/EXPERIMENT_LOG.md 'T=0, seed 1234, think off')",
    "leaderboard": "none",
    "controls": [],
    "reported_numbers": [
        {"what": "tasks with function-level gold labels", "value": "122 of 129",
         "where": "docs/EXPERIMENT_LOG.md; README.md", "adapter": "not applicable (nothing yielded)", "match": True},
    ],
}


def attempts(src):
    return
    yield
