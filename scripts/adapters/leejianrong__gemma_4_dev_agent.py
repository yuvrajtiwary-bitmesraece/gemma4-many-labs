"""Adapter for leejianrong/gemma-4-dev-agent ("Where does a code graph help a bug-fixing agent?").

Not eligible: the repository publishes run-level aggregates only; per-task files are test fixtures.
"""

REPO = "leejianrong/gemma-4-dev-agent"

CARD = {
    "eligible": False,
    "reason": "no per-task outcomes are published: experiments/runs.csv has one row per run with counts only (6 runs of "
              "Gemma 4 31B on 5, 30 and 36 public tasks), and the lab keeps per-task data local on purpose (README.md: "
              "'the repo holds only aggregates and code'); the only per-task tables are synthetic test fixtures "
              "(tests/fixtures, ids fastapi_0001, rich_0002, requests_0003)",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License': 'Apache-2.0 "
                            "([ADR-0001](docs/adr/0001-license-apache-2.md))'"},
    "models": [],
    "harness": None,
    "summary": "Competition entry and paper draft on code-graph localization. Agent runs: a reused budget-aware "
               "Gemma 4 31B bundle on a stratified 30-task sample (3/30) and a baseline-vs-localizer-skill ablation on a "
               "36-task dev subset with two seeded replicates per arm (13, 15 vs 14, 14 of 36), on Kaggle 4xL4 with the "
               "competition harness. Only run-level counts are in the repository.",
    "evidence": {
        "model": "experiments/runs.csv column model: 'gemma-4-31b-it-qat-w4a16-ct' in all 6 rows",
        "harness": "experiments/runs.csv notes: 'first real run: 4xL4 tp4, subprocess sandbox, vLLM up in 494s'; "
                   "docs/GPU_RUNS.md: '`swegemma` 0.2.11'",
        "systems": "experiments/runs.csv column variant: baseline-bundle-v2, baseline-v2, skill-v1 (parent_variant "
                   "baseline-v2), identified by zip_sha256",
        "replicates": "experiments/runs.csv: run ids dev-a0-r1 / dev-a0-r2 (variant baseline-v2, seed 1 / 2) and "
                      "dev-a2-r1 / dev-a2-r2 (variant skill-v1, seed 1 / 2); docs/PLAN.md: 'Identity of an experiment is "
                      "the SHA-256 of the zip. Two runs of the same hash are replicates'; docs/QUESTIONS.md A40: "
                      "'2 replicates per arm, 31B w4a16, seeds 1 and 2' (aggregates only, not usable here)",
        "task_sets": "experiments/runs.csv column subset: smoke-5-stratified, stratified30, dev36; notes: 'dev subset 36 "
                     "(fastapi 30, requests 5, httpx 1)'; the task ids of these subsets are not published "
                     "(docs/ABLATION_PLAN.md: '`data/dev_subset.json` (gitignored ...)')",
    },
    "decisions": [
        "nothing is yielded: the source gives per-run counts and no per-task rows (flag aggregated_only); rows are not "
        "invented from counts, and the task ids of the 30- and 36-task subsets are not in the repository",
        "the fixture ids (fastapi_0001, rich_0002, requests_0003) would pass the harmonizer's id pattern although they "
        "are synthetic, so the fixture tables must not be read",
    ],
    "excluded": [
        "experiments/runs.csv: run-level aggregates only (Gemma 4 31B, temperature 0.2, 28 tool calls, 480 s): "
        "smoke-2026-11-08 0/5; baseline30-2026-11-08 3/30 (stratified30); dev36 replicates dev-a0-r1 13/36 and dev-a0-r2 "
        "15/36 (baseline-v2, seeds 1 and 2); dev-a2-r1 14/36 and dev-a2-r2 14/36 (skill-v1, seeds 1 and 2)",
        "tests/fixtures/runs/task_results.csv, runs.csv, traces/trace_fx_baseline_*.json: synthetic test fixtures "
        "(runs.csv notes: 'synthetic fixture'; ids fastapi_0001, rich_0002, requests_0003)",
        "tests/fixtures/localize/localize_tasks.csv, localize_summary.csv: fixtures of localisation metrics (hit@k, "
        "recall@k), not resolved/not-resolved outcomes",
        "synthetic/: generator of synthetic (mutated) tasks, outside the development set",
        "reference-patch control (docs/GPU_RUNS.md s7, counts only): `verify_task` with the gold patch resolved 0 of 9 "
        "inconclusive baseline tasks and 1 of 3 resolved controls in the lab's local Docker setup, which the lab calls "
        "'not validated'",
        "public leaderboard score 0.18 of the baseline zip (README.md): hidden-test result",
    ],
    "dropped": {},
    "flags": ["aggregated_only"],
    "reported_numbers": [
        {"what": "baseline bundle on the stratified 30-task sample", "value": "3 of 30",
         "where": "README.md; paper/DRAFT.md s8", "adapter": "not yielded; experiments/runs.csv baseline30-2026-11-08: "
                                                             "n_tasks 30, n_resolved 3", "match": True},
        {"what": "36 dev tasks, 2 replicates: baseline; skill", "value": "13, 15; 14, 14",
         "where": "README.md; docs/QUESTIONS.md A40; paper/DRAFT.md s10",
         "adapter": "not yielded; experiments/runs.csv dev-a0-r1 13/36, dev-a0-r2 15/36, dev-a2-r1 14/36, dev-a2-r2 14/36",
         "match": True},
    ],
}


def attempts(src):
    return
    yield
