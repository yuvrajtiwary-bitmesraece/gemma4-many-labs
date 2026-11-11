"""Adapter for HIAPPtitude-owner/gemma-eval-kit (a repaired local evaluator plus gold/empty labels; no agent runs)."""

REPO = "HIAPPtitude-owner/gemma-eval-kit"

CARD = {
    "eligible": False,
    "reason": "no agent attempts in the snapshot: the only per-task file (labels/dev129_labels.csv, 387 rows = 129 "
              "tasks x 3 evaluation routes) holds gold-patch and empty-patch control verdicts",
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md "
                                                  "'## License Apache License 2.0 (see `LICENSE` and `NOTICE`)'"},
    "models": [],
    "harness": None,
    "summary": "An evaluator repair kit: the lab verified every dev task with its gold patch and with an empty patch "
               "on three harness routes (as_shipped and diagnostic WSL1 subprocess set-ups, the public Docker kit) "
               "and classified why gold patches fail. No model is run in anything published here.",
    "evidence": {
        "model": "none: README.md 'labels/dev129_labels.csv: one row per (task, route), 129 tasks x 3 routes = 387 "
                 "rows'; columns 'task_id,repo,route,gold_pass,empty_fail,root_cause,runs'",
        "harness": "README.md: 'This is not the competition's scorer. It follows the scorer's documented "
                   "environment model'; '`route` is the harness: `as_shipped` and `diagnostic` (a WSL1 subprocess "
                   "setup) and `public_docker` (the competition's public Docker kit)'",
        "systems": "none (controls only)",
        "replicates": "none: README.md '`runs` is always 1. These labels are single-run: each comes from one run, "
                      "and there is no stability (3 of 3) label in this release'",
        "task_sets": "all 129 dev tasks on each route (fastapi 67, rich 48, requests 13, httpx 1)",
    },
    "decisions": [
        "gold_pass / empty_fail are verdicts on the reference patch and on no patch: non-agent controls, nothing "
        "to yield",
        "the agent runs the code refers to (src/factory/rescore.py: 'R2 (config v1) and R3 (config v1_calls50) ran "
        "the agent on Kaggle', 40 tasks each, re-scored on the three routes) are not in the repository; README.md "
        "places them in the Kaggle dataset bscode/gemma-eval-kit-results, which is not part of this snapshot",
    ],
    "excluded": [
        "labels/dev129_labels.csv route=as_shipped: gold patch passes 52/129 (fastapi 10, rich 39, requests 3), "
        "empty patch passes 4/129 (fastapi_14873, requests_7309, requests_7315, requests_7427); gold failures: "
        "missing_dependency 36, host_venv_shadowing 30, unclassified 10, invalid_task 1",
        "labels/dev129_labels.csv route=diagnostic: gold patch passes 45/129 (rich 39, requests 5, httpx 1), empty "
        "patch passes 0/129; gold failures: missing_dependency 76, unclassified 7, invalid_task 1",
        "labels/dev129_labels.csv route=public_docker: gold patch passes 43/129 (rich 39, requests 3, fastapi 1), "
        "empty patch passes 4/129 (requests_7309, requests_7315, requests_7427, rich_3468); gold failures: "
        "missing_dependency 75, unclassified 9, version_drift 1, invalid_task 1",
        "src/factory/test_*.py: unit-test fixtures with task ids, not results",
    ],
    "dropped": {},
    "flags": ["controls_only"],
    "reported_numbers": [
        {"what": "label rows", "value": "129 tasks x 3 routes = 387 rows", "where": "README.md",
         "adapter": "387 rows, 129 ids, 129 per route (read, not yielded)", "match": True},
    ],
}


def attempts(src):
    return
    yield
