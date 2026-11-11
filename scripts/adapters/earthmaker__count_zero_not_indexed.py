"""Adapter for earthmaker/count-zero-not-indexed
("When count: 0 Means "Not Indexed": Code-Graph Gaps and a 4-bit Gemma 4 Agent").

Not eligible: the repository holds code, agent bundles, the pre-registration and a
supplement, but no run outputs.  Resolution is reported only as counts per
condition, so nothing is yielded.
"""

REPO = "earthmaker/count-zero-not-indexed"

CARD = {
    "eligible": False,
    "reason": (
        "no per-task outcomes in the repository: the run outputs (runs/v6_confirm1 ... v11_confirm5) and derived "
        "tables are not published; resolution appears only as aggregate counts per condition in SUPPLEMENT.md "
        "(e.g. 'B - A: 14/81 vs. 14/81'), and the 12 task ids in PREREGISTRATION_en.md are sampling / pilot notes "
        "without a per-task outcome table"
    ),
    "licence": {
        "spdx": "Apache-2.0",
        "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: 'Code and bundles: Apache "
                    "License 2.0 (`LICENSE`). Documentation, including this README, `SUPPLEMENT.md` and "
                    "`PREREGISTRATION_en.md`: CC BY 4.0 (`LICENSE-docs`).'; LICENSE-docs: 'The documentation in this "
                    "repository (README.md, SUPPLEMENT.md and PREREGISTRATION_en.md) is licensed under the Creative "
                    "Commons Attribution 4.0 International License (CC BY 4.0).'",
    },
    "models": [],
    "harness": None,
    "summary": (
        "Pre-registered study of code-graph gaps: four bundles of a gemma-4-31b-it-qat-w4a16-ct agent (A public "
        "baseline, B empty-answer guard, C graph-first, D both) run once on 81 dev tasks each on Kaggle L4x4 with "
        "swegemma 0.2.7 (324 runs). Only code, bundles, the pre-registration and aggregate results are public."
    ),
    "evidence": {
        "model": "bundles/A_baseline/agent.yaml: 'model: gemma-4-31b-it-qat-w4a16-ct'",
        "harness": "README.md: 'Kaggle notebook that runs the bundles on the public tasks (L4 x4) and keeps the "
                   "traces'; 'Harness: swegemma 0.2.7 from the competition wheelhouse of 30 September 2026'",
        "systems": "README.md: 'The four agent configurations: A (public baseline used as is), B (A plus the "
                   "empty-answer guard), C (A plus the graph-first block), D (B plus the graph-first block).'",
        "replicates": "none published; PREREGISTRATION_en.md s4: 'Repeat r = 1 is first run on all 81 tasks; if quota "
                      "remains, r = 2 is added in the same permutation' - SUPPLEMENT.md S5 reports 324 runs = 4 "
                      "conditions x 81 tasks, i.e. one run per condition",
        "task_sets": "PREREGISTRATION_en.md s4: 'Target: exposed 47 + covered 34 = 81 tasks (within the strict "
                     "109)'; the order file derived/part2_order.tsv is not in the repository",
        "no_outcome": "README.md: 'Competition data (repository snapshots, call graphs, embeddings, `tasks.jsonl`, "
                      "patches) are not redistributed here'; 'Not included: the three derived data files listed in "
                      "`MANIFEST.txt`'; the snapshot has no data file with task ids (checked: 59 files; "
                      "coder/frozen/selftest.log contains synthetic self-test lines only)",
    },
    "decisions": [
        "aggregates are not turned into runs: SUPPLEMENT.md gives resolved counts for A and B only (14/81 each, "
        "discordant pairs 1 vs. 1) and no task-level list",
        "task ids named in PREREGISTRATION_en.md (pilot, smoke-test and first-five-of-permutation lists) carry no "
        "outcome except one pilot remark ('pilot task requests_7315, one A run ... unresolved'), a pilot the lab "
        "itself excludes from analysis",
    ],
    "excluded": [
        "SUPPLEMENT.md S5 (aggregate only): 'P4 Resolution | B - A | 14/81 vs. 14/81; discordant 1 vs. 1 | 0.0 pts "
        "[-4.5, 4.5]'; 'Runs that ended in a harness error (68 of 324: A 15, B 18, C 20, D 15; 53 of them "
        "context-window overflows) count as unresolved'; resolution of C and D is not printed in the repository",
        "SUPPLEMENT.md S2 / PREREGISTRATION_en.md change log: auxiliary arm through a model proxy, 18 tasks x 2 "
        "conditions, '0/36 resolved' - stopped because the proxy returned empty responses (infrastructure "
        "failure; aggregate only)",
        "PREREGISTRATION_en.md: smoke tests v3 (fastapi_11194, rich_3061, requests_6757, A x 3) and v5 (C and D x "
        "fastapi_14953, rich_4077), pilot v4 (A and B x 10 tasks) and the proxy pilot (requests_7315, one A run, "
        "'unresolved, 28 tool calls'): pilots / smoke tests the lab excludes from its analysis, no outcome table",
        "gold-patch judgeability control (probes/gold_check.py; results as counts only): 'strict 109' of 129 tasks "
        "judgeable, 121 under a relaxed rule ('no test newly broken by the reference patch, at least one test "
        "failing at base'), SUPPLEMENT.md S1",
        "coder/frozen/selftest.log, probes/probe_empty.py: synthetic self-tests and interface probes",
        "bundles/*, coder/*, kernel/*, probes/*: configurations and code, no results",
    ],
    "dropped": {},
    "flags": ["aggregated_only"],
    "reported_numbers": [
        {"what": "resolved, condition A and condition B (81 tasks each)", "value": "14/81 and 14/81",
         "where": "SUPPLEMENT.md S5", "adapter": "not available: no per-task rows in the repository", "match": None},
    ],
}


def attempts(src):
    return
    yield
