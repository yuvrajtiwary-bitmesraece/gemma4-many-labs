"""Adapter for Mattbusel/swegraph-lite ("Who Should Drive the Graph? Harness-Driven vs Model-Driven Bug Localization
for On-Device Gemma 4 Agents").

Not eligible: another benchmark (SWE-bench Lite) and localisation metrics only.  The six ids that look like
development-set ids are SWE-bench instances `psf__requests-<n>`; nothing is yielded and no id is rewritten.
"""

REPO = "Mattbusel/swegraph-lite"

CARD = {
    "eligible": False,
    "reason": "no development-set attempts and no resolved/not-resolved outcome: every run is on the 300 tasks of "
              "SWE-bench Lite (plus its 23-task dev split for tuning), rebuilt as code graphs in the competition's "
              "format (ids such as astropy__astropy-12907, django__django-11099, psf__requests-1963), and the Gemma 4 "
              "E2B / E4B / 12B runs only rank graph nodes for fault localisation, scored by hit@k and file@k against the "
              "functions the gold patch changes; no patch is written and no test is run. The six ids of the form "
              "psf__requests-<n> are SWE-bench instances, not competition tasks",
    "licence": {"spdx": "MIT",
                "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Tensorust LLC'; README.md '## License': 'MIT, "
                            "Tensorust LLC. The graphs contain source code from the benchmark repositories, which keeps "
                            "each repository's own license (see the dataset card).'"},
    "models": [],
    "harness": None,
    "summary": "A localisation study on SWE-bench Lite: three designs (harness-driven rerank, model-driven agent with "
               "graph tools and 10 calls, agent seeded with the harness top 10) at Gemma 4 E2B, E4B and 12B through "
               "Ollama on one RTX 4070, scored by top-k localisation hits.",
    "evidence": {
        "model": "README.md: 'runs Gemma 4 through Ollama in three modes' with gemma4:e2b-it-qat, gemma4:e4b-it-qat, "
                 "gemma4:12b-it-qat; run files out/runs/<mode>__gemma4_<size>-it-qat_think0[_s10].jsonl",
        "harness": "none: swegraph/evaluate.py and swegraph/analysis.py 'compute hit@k, file@k, token cost, paired "
                   "bootstrap CIs, McNemar tests, tool usage and invalid-id rates' (README.md); the competition harness "
                   "is not used",
        "systems": "README.md: modes `rerank`, `agent`, `agentseed` x three model sizes (9 run files, 300 records each "
                   "except agent__gemma4_e2b with 281 and agentseed__gemma4_12b with 94)",
        "replicates": "none: README.md 'All runs use temperature 0 and seed 0.'; the suffix _s10 of the agent runs is "
                      "the 10-call budget, not a seed",
        "task_sets": "paper/paper.md abstract: 'On all 300 SWE-bench Lite tasks, rebuilt as code graphs in the exact "
                     "format of the Gemma 4 Developer Agent competition'; README.md: 'picks the fusion weights on the "
                     "SWE-bench Lite **dev** split, whose 6 repos never appear in test'",
    },
    "decisions": [
        "SWE-bench Lite tasks are outside the competition's public development set, and hit@k is a localisation "
        "metric: excluded on both counts; nothing is yielded",
        "out_dev/ is the SWE-bench Lite dev split (pvlib, pydicom, astroid, sqlfluff, marshmallow, pyvista), not the "
        "competition's development set",
    ],
    "excluded": [
        "out/runs/*.jsonl (9 files, 2,475 records): per-task rankings, tool calls and token counts of Gemma 4 E2B, E4B "
        "and 12B on SWE-bench Lite; no resolved outcome",
        "out/rank/*.json (300 files), out_dev/rank/*.json (23 files): harness rankings (bm25, dense, anchor, fused, "
        "fused_tuned) per SWE-bench instance, no model",
        "out/labels.jsonl, out_dev/labels.jsonl: gold localisation labels per SWE-bench instance",
        "out/metrics.json, out_dev/metrics.json, out/analysis.json, out/ablation.json, out/fusion_tuning.json: "
        "aggregated localisation metrics",
    ],
    "dropped": {},
    "flags": ["other_benchmark_swebench_lite", "localisation_only"],
    "sampling": "temperature 0 and seed 0 (README.md: 'All runs use temperature 0 and seed 0.'); not competition tasks",
    "leaderboard": "none",
    "controls": [],
    "reported_numbers": [
        {"what": "tasks", "value": "300 SWE-bench Lite tasks from 12 repositories", "where": "paper/paper.md abstract",
         "adapter": "not applicable (nothing yielded)", "match": True},
    ],
}


def attempts(src):
    return
    yield
