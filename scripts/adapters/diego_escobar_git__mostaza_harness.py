"""Adapter for diego-escobar-git/mostaza-harness (E0 baseline vs E1 hybrid-retrieval pilot, 3 tasks)."""
import json
import re

REPO = "diego-escobar-git/mostaza-harness"

CARD = {
    "eligible": True,
    "reason": "",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: 'Original code is licensed "
                            "under [Apache-2.0](LICENSE)'"},
    "models": ["31B"],
    "harness": "official",
    "summary": "Pilot of a baseline (E0) and a hybrid-retrieval prompt (E1) with Gemma 4 31B on Kaggle L4x4 through the "
               "official harness, on a frozen cohort of 3 public tasks (requests_7205, fastapi_15589, rich_3061): two "
               "paired batches, 12 attempts, none resolved. Every run has 3 tasks, below the harmonizer's minimum of 10 "
               "tasks per run, so nothing from this source reaches the pooled table.",
    "evidence": {
        "model": "experiments/E0-B180/manifest.json and experiments/E0-baseline/manifest.json: 'model': "
                 "'gemma-4-31b-it-qat-w4a16-ct'; docs/experiments.md: 'executed in both arms on Kaggle with Gemma 4 31B "
                 "and four NVIDIA L4 GPUs'",
        "harness": "docs/implementation/evidence/mostaza-phase2-startup-run/report.md: 'Model Gemma4 31B QAT revision2, "
                   "official harness, Python3.12.13, torch2.10.0+cu128, vLLM0.19.1, swegemma0.2.11'; records carry "
                   "run_kind 'real_kaggle'",
        "systems": "docs/experiments.md: 'E0 is the minimal single-agent baseline; E1 changes exploration instructions "
                   "only'; experiments/E0-baseline/manifest.json budgets max_time_minutes 1, max_tool_calls 10; "
                   "experiments/E0-B180/manifest.json budgets max_time_minutes 3, max_tool_calls 30",
        "replicates": "none: each configuration was executed once (phase 1: E0, E1 with the one-minute budget; phase 2: "
                      "E0-B180, E1-B180); the two phases differ in budgets and are different systems",
        "task_sets": "experiments/cohort.json: 'selection_basis': 'pre-outcome', tasks requests_7205, fastapi_15589, "
                     "rich_3061; docs/research-proposal.md: 'Preselect three public tasks from Requests, FastAPI, and "
                     "Rich or HTTPX ... before inspecting outcomes'",
    },
    "decisions": [
        "system = the lab's configuration name: phase 1 arms are 'E0-baseline' and 'E1-hybrid-retrieval' (one-minute "
        "session, 10 tool calls), phase 2 arms are 'E0-B180' and 'E1-B180' (three-minute session, 30 tool calls)",
        "run = experiment id / arm / first 8 characters of the lab's run_id",
        "status FAIL -> y=0; status ERROR (no patch because of session timeout or context-window overflow, tests not "
        "run, resolved null) -> y=0, because the competition scores these as failures; status NOT_RUN -> dropped",
        "err = the record has a non-empty reason (timeout, context-window error) or status ERROR",
        "the source is kept as eligible although all four runs have 3 tasks: the minimum-tasks-per-run rule is applied "
        "by the harmonizer, not by the adapter",
    ],
    "excluded": [
        "docs/implementation/evidence/mostaza-phase2-observed-run/evaluation-records.json: 6 NOT_RUN rows (vLLM CUDA OOM "
        "before any task started), see dropped",
        "docs/implementation/evidence/mostaza-phase1-foundation/comparison.json, official-validation.json, "
        "resource-review.json, report.md: derived views of the 6 phase-1 attempts",
        "docs/implementation/evidence/mostaza-phase2-budget-retrieval/comparison.json, trace-diagnosis.json: re-analysis "
        "of the same 6 phase-1 attempts; readiness.json: 6 NOT_RUN placeholders of prepared, never executed runs",
        "docs/implementation/evidence/mostaza-phase2-startup-run/comparison.json, trace-observations.json, "
        "E*-B180-derived-manifest.json, paired-report.md and mostaza-phase2-postrun-diagnosis/trace-evidence.json: "
        "derived views of the 6 phase-2 attempts",
        "experiments/E*-B300, E*-C8K*, E1-R180: prepared configurations, 'compiler-validated, with no real results' "
        "(README.md)",
        "docs/implementation/evidence/*-cpu/, mostaza-pcv01-*: CPU gates on synthetic fixtures, no Gemma execution",
        "tests/: unit and integration fixtures",
    ],
    "dropped": {"NOT_RUN: vLLM EngineCore CUDA OOM during warm-up, no task started (phase-2 observed run)": 6},
    "flags": ["pilot_3_tasks", "all_runs_below_10_tasks"],
    "reported_numbers": [
        {"what": "phase 1, per arm PASS / FAIL / ERROR / NOT_RUN", "value": "E0 0/1/2/0, E1 0/1/2/0 of 3",
         "where": "docs/implementation/evidence/mostaza-phase1-foundation/report.md",
         "adapter": "E0-baseline 0/3 resolved (err 3), E1-hybrid-retrieval 0/3 resolved (err 3)", "match": True},
        {"what": "phase 2 B180-startup-v3, per arm PASS / FAIL / ERROR / NOT_RUN", "value": "E0 0/2/1/0, E1 0/2/1/0 of 3",
         "where": "README.md 'Each arm produced PASS0 / FAIL2 / ERROR1'; "
                  "docs/implementation/evidence/mostaza-phase2-startup-run/report.md",
         "adapter": "E0-B180 0/3 resolved, E1-B180 0/3 resolved", "match": True},
        {"what": "phase 2 observed run", "value": "all six rows NOT_RUN",
         "where": "docs/implementation/evidence/mostaza-phase2-observed-run/report.md",
         "adapter": "6 rows dropped", "match": True},
    ],
}

FILES = (r"^docs/implementation/evidence/(mostaza-phase1-foundation/paired-results\.jsonl"
         r"|mostaza-phase2-(observed|startup)-run/evaluation-records\.json)$")
PHASE1 = {"E0": "E0-baseline", "E1": "E1-hybrid-retrieval"}


def _rows(path, raw):
    text = raw.decode("utf-8", "replace")
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def attempts(src):
    pattern = re.compile(FILES)
    for path, raw in src.read_many(pattern.pattern):
        for row in _rows(path, raw):
            status = row.get("status")
            if status == "NOT_RUN":
                continue
            arm, experiment = row["arm"], row["experiment_id"]
            if experiment == "mostaza-phase1-foundation":
                system = PHASE1[arm]
            elif experiment == "B180-startup-v3":
                system = f"{arm}-B180"
            else:
                system = f"{arm}@{experiment}"
            reason = row.get("reason")
            yield {
                "file": path,
                "system": system,
                "run": f"{experiment}/{arm}/{str(row['run_id'])[:8]}",
                "task": row["instance_id"],
                "y": int(row.get("resolved") is True),
                "model": "31B",
                "harness": "official",
                "err": bool(reason) or status == "ERROR",
                "status": f"{status}: {reason}" if reason else status,
            }
