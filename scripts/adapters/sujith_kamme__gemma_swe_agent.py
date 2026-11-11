"""Adapter for sujith-kamme/gemma-swe-agent.

The repository holds agent configs (v1-v3), a gold-patch solvability check and tool
(localisation) measurements.  The only per-task agent outcome is one smoke run of
agent v3 that stopped after its first task.
"""

REPO = "sujith-kamme/gemma-swe-agent"

ROWS = "kaggle/v3-smoke/output2/baseline_rows.json"

CARD = {
    "eligible": True,
    "licence": {"spdx": "MIT", "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Sujith Kamme'; README.md '## License MIT'"},
    "models": ["31B"],
    "harness": "official",
    "summary": (
        "One Kaggle 4xL4 smoke run of the lab's skills-based agent v3 (gemma-4-31b-it-qat-w4a16-ct, swegemma "
        "Evaluator): 1 task attempted and resolved before a quota guard stopped the run. Everything else is "
        "gold-patch checks and localisation-tool measurements."
    ),
    "evidence": {
        "model": "kaggle/v3-smoke/output2/baseline_summary.json: '\"model\": \"gemma-4-31b-it-qat-w4a16-ct\"'; "
                 "kaggle/v3-smoke/output2/agent/agent.yaml: 'model: gemma-4-31b-it-qat-w4a16-ct'",
        "harness": "kaggle/v3-smoke/run.py: 'official vLLM + swegemma harness, same setup as scoring' ... 'from "
                   "swegemma.evaluate import Evaluator' ... 'evaluator.evaluate_task(task=task, ...)' with "
                   "sandbox='subprocess'; README.md: 'runs Gemma 4 inside the official `swegemma` harness'",
        "systems": "kaggle/v3-smoke/kernel-metadata.json: 'gemma4-agent-v3-smoke' with dataset "
                   "'sujithkamme/gemma4-agent-v3'; output2/agent/agent.yaml: 'name: swe_agent_v3'; budget "
                   "max_time_minutes 5, max_tool_calls 50, max_turns 100",
        "replicates": "none: a single run",
        "task_sets": "kaggle/v3-smoke/run.py: 'TASK_IDS = [\"rich_3006\", \"fastapi_14794\"]'; "
                     "gemma4-agent-v3-smoke.log: 'quota guard: stopping before task 2' / 'DONE 1/1 resolved'",
    },
    "decisions": [
        "the smoke run is yielded although it has one task: it is a real agent attempt scored by the official "
        "harness; the harmonizer's minimum of 10 tasks per run will remove it, so this source adds nothing to "
        "the pooled table",
        "system label 'v3-smoke': the agent files in kaggle/v3-smoke/output2/agent differ from agents/v3 (earlier "
        "prompt / skills / eval_config), and the run is a partial smoke test (1 of 2 planned tasks, hand-picked)",
        "baseline_summary.json repeats the row of baseline_rows.json and is not read",
        "err = the row's 'error' field is non-empty",
    ],
    "excluded": [
        "kaggle/v3-smoke/output2/baseline_summary.json: same single row as baseline_rows.json plus totals",
        "README.md results table ('10-task subset, gemma-4-31b-it-qat-w4a16-ct, Kaggle 4x L4'): "
        "'sample_submission (no adapters) 4/10', 'v1 pending' - aggregate only, no per-task rows or outputs of "
        "kaggle/baseline, kaggle/v1run, kaggle/eval30-baseline or kaggle/eval30-v2 are in the repository",
        "kaggle/solvable_tasks.json: gold-patch control on the dev set, no model - 'Tasks whose gold patch passes "
        "in the Kaggle subprocess sandbox (goldcheck run 2026-10-25)', count 75 of 129 (31 fastapi, 43 rich, "
        "1 requests, 0 httpx); kaggle/eval30_tasks.json is a seeded sample of 30 of these",
        "kaggle/goldmined/output2/gold/gold_check.json (+ 240 .txt logs) and kaggle/mined_solvable.json: gold-patch "
        "control on 240 tasks the lab mined from other repositories (arrow, boltons, more_itertools, starlette, "
        "...): gold patch resolves 236/240; not dev-set tasks",
        "kaggle/mine/output2/mined/tasks.jsonl, mine_*.log: mined (non-competition) task definitions",
        "kaggle/measure/output2|output3/measure/competition_rows.json (129 rows each), mined_rows.json, "
        "summary.json: localisation / test-discovery ranks of the agent's helper tools (file_rank, sym_rank, "
        "tests_for_rank, run_tests_hit), no agent run and no resolved outcome",
        "local_eval/: local loop against Google AI Studio's hosted gemma-4-31b-it through an OpenAI-compatible "
        "proxy (gemini_proxy.py); scripts only, no results in the repository",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["smoke_test_only", "partial_runs", "aggregated_only"],
    "reported_numbers": [
        {"what": "v3 smoke run", "value": "1/1 resolved (fastapi_14794)",
         "where": "kaggle/v3-smoke/output2/baseline_summary.json; gemma4-agent-v3-smoke.log 'DONE 1/1 resolved'",
         "adapter": "1/1", "match": True},
        {"what": "sample_submission baseline, 10-task subset", "value": "4/10",
         "where": "README.md", "adapter": "not available: no per-task rows in the repository", "match": None},
    ],
}


def attempts(src):
    for path, rows in src.json(r"^kaggle/v3-smoke/output2/baseline_rows\.json$"):
        for row in rows:
            error = row.get("error")
            yield {
                "file": path,
                "system": "v3-smoke",
                "run": "v3-smoke-output2",
                "task": row["id"],
                "y": int(bool(row["resolved"])),
                "model": "31B",
                "harness": "official",
                "err": bool(error),
                "status": error or f"exit {row.get('exit_code')}",
            }
