"""Adapter for Janeferdinant/gemma-4-developer-agent-paper (GETVP). Not eligible: simulated benchmark."""

REPO = "Janeferdinant/gemma-4-developer-agent-paper"

CARD = {
    "eligible": False,
    "reason": "no development-set attempts and no Gemma 4 execution: the 150 records (6 configurations x 25 instances) "
              "are produced by research/scripts/run_benchmarks.py on 25 instances written inside that script "
              "('GETVP-RepoBench-25', ids FASTAPI-01 ... FLASK-05, repos fastapi/pydantic/sympy/requests/flask), and the "
              "'repair' is a hard-coded simulated function that returns the stored fix or a broken stub depending on the "
              "configuration name and hash(instance id); no model is called (average latency 0.002-0.003 s per instance in "
              "research/ablation_results.csv)",
    "licence": {"spdx": "MIT", "evidence": "LICENSE: 'MIT License Copyright (c) 2026 AI Research Team'; README.md badge "
                                           "'License: MIT' and 'LICENSE # MIT Open Source License'"},
    "models": [],
    "harness": None,
    "summary": "Paper-track repository for GETVP (graph-guided retrieval + evidence planning + test-verified repair); its "
               "results table comes from a scripted simulation on 25 self-written toy instances, not from Gemma 4 runs on "
               "competition tasks.",
    "evidence": {
        "model": "paper/final_writeup.md s4.2: 'We evaluate `gemma-4-31b-it-qat-w4a16-ct`' (stated); but "
                 "research/scripts/run_benchmarks.py: '# Simulated repair logic based on config capabilities' ... "
                 "'return inst[\"fixed_code\"] if (hash(inst_id) % 3 == 0) else \"def broken(x\\n\"' and "
                 "src/orchestrator.py: 'if simulated_repair_fn: new_block = simulated_repair_fn(issue_id, current_repair_try)'",
        "harness": "src/orchestrator.py: 'self.test_runner.evaluate_in_memory_suite(module_code=..., test_code=test_script, ...)' "
                   "(in-memory assert snippets from the script; not the competition harness)",
        "systems": "research/scripts/run_benchmarks.py main(): six configs baseline_vanilla, baseline_text_only, "
                   "ablation_no_graph, ablation_no_plan, ablation_no_verif, full_getvp",
        "replicates": "none: one pass per configuration",
        "task_sets": "research/data_inventory.md: 'an evaluation suite comprising 25 curated repository-level bug-fixing "
                     "instances drawn from representative open-source Python repositories (FastAPI, SymPy, PyDantic, "
                     "Requests, Flask)'; instance ids are FASTAPI-01..05, PYDANTIC-01..05, SYMPY-01..05, REQUESTS-01..05, "
                     "FLASK-01..05",
    },
    "decisions": [
        "nothing is yielded: ids such as FASTAPI-01 or REQUESTS-02 would pass the harmonizer's loose id pattern "
        "(fastapi_1, requests_2) although they are not competition tasks, so the adapter must stay empty",
        "the instances are synthetic (file contents, test snippet and 'fixed_code' are string literals in "
        "run_benchmarks.py) and the outcome is determined by the configuration name, not by a model",
    ],
    "excluded": [
        "research/results.csv (150 rows): simulated outcomes on 25 synthetic instances",
        "dashboard/src/data/instances.json (150 rows): copy of research/results.csv for the web dashboard",
        "research/experiment_logs/<config>_<instance>.json (150 files): per-instance logs of the same simulated runs",
        "research/ablation_results.csv, research/metrics_summary.json, dashboard/src/data/metrics.json, "
        "paper/tables/*.md: aggregates of the same 150 rows",
        "tests/: unit tests of the framework",
    ],
    "dropped": {},
    "flags": ["simulated_outcomes", "synthetic_tasks"],
    "reported_numbers": [
        {"what": "resolution rate per configuration on GETVP-RepoBench-25 (vanilla, text-only, w/o graph, w/o planning, "
                 "w/o verification, full)", "value": "20.0%, 48.0%, 76.0%, 88.0%, 88.0%, 88.0% of 25",
         "where": "README.md s2; paper/final_writeup.md Table (s5); research/ablation_results.csv",
         "adapter": "not yielded; research/results.csv has resolved=True in 5, 12, 19, 22, 22, 22 of 25 rows",
         "match": True},
    ],
}


def attempts(src):
    return
    yield
