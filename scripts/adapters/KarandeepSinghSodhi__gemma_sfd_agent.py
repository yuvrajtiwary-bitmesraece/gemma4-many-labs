"""Adapter for KarandeepSinghSodhi/gemma-sfd-agent.

Two write-ups share the data: "State-Filtered Deliberation: Stage-Isolated Context
Management for Small Open-Weight Software Engineering Agents" (KAGGLE_WRITEUP.md)
and "ATIF-Gemma: Diagnostic Corpus and Failure Taxonomy for Small Developer Agents"
(KAGGLE_WRITEUP_RESOURCE.md).
"""
import re

REPO = "KarandeepSinghSodhi/gemma-sfd-agent"

CARD = {
    "eligible": True,
    "licence": {
        "spdx": "Apache-2.0",
        "evidence": (
            "LICENSE (short form, which is why GitHub reports NOASSERTION): 'Apache License Version 2.0, January 2004 "
            "... Copyright 2026 Karandeep Singh Sodhi Licensed under the Apache License, Version 2.0'; README.md "
            "badge 'License: Apache 2.0'. The copy of the traces under dataset_resource/ is additionally "
            "CC-BY-4.0: dataset_resource/README.md 'Dataset License: Creative Commons Attribution 4.0 International "
            "(CC BY 4.0)'; dataset_resource/NOTICE 'trajectory traces, metadata, and diagnostic analysis tools "
            "licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license and the Apache "
            "License 2.0'. The adapter reads traces/**/paired_progress.jsonl (root, Apache-2.0)."
        ),
    },
    "models": ["E4B"],
    "harness": "official",
    "summary": (
        "112 Kaggle 4xL4 runs of gemma-4-e4b-it-qat-w4a16-ct through the swegemma Evaluator: three agent "
        "configurations (official starter, controlled single agent, 3-stage SFD) on 32 dev tasks in four 8-task "
        "shards, plus a reproducibility rerun of the first shard for two configurations."
    ),
    "evidence": {
        "model": "README.md: 'We evaluated three controlled configurations on `gemma-4-e4b-it-qat-w4a16-ct` on "
                 "Kaggle's 4x NVIDIA L4 GPU cluster (... `temperature=0.0`, `seed=42`) across 32 SWE-bench tasks'",
        "harness": "README.md: 'Every result reported in this repository is computed strictly from real two-phase "
                   "`swegemma` + `vLLM` execution traces'; scripts/build_eval_notebook.py builds one "
                   "swegemma.evaluate.Evaluator per config and writes each ev.evaluate_task result "
                   "(resolved, test_exit_code, error) to eval_results/paired_progress.jsonl",
        "systems": "experiment_plan.md s1.1: 'Config 1 (`flat_starter`): Official competition starter kit ... "
                   "Config 2 (`flat_controlled`): Single `LlmAgent` ... Config 3 (`sfd`): Three-stage "
                   "`SequentialAgent`'; all with max_tool_calls 15, max_turns 20, max_time_minutes 2.5",
        "replicates": "KAGGLE_WRITEUP_RESOURCE.md Table 1 note: 'The Cohort 1 Rerun repeats the flat_controlled and "
                      "sfd configurations across the same eight tasks, yielding 16 rerun traces. The flat_starter "
                      "configuration was not rerun.'; dataset_resource/README.md: '`sfd_rerun` (8 runs): Independent "
                      "run at T=0.0, seed=42, tensor parallel=4.'; checked: all files under traces/**/sub_<config>/ "
                      "(agent.yaml, sampling.yaml, eval_config.yaml, prompts, sub_agents) are byte-identical in the "
                      "four shard directories and the rerun directory",
        "task_sets": "README.md: 'traces/eval_results/ # Cohort 1 paired runs (24 runs across 8 canonical tasks) ... "
                     "shard2_results/ # Cohort 2 ... shard3_results/ # Cohort 3 ... shard4_results/ # Cohort 4 "
                     "paired runs (24 runs across 8 stratified disjoint tasks)'; Table: 'flat_starter | 32 | 2/32'",
    },
    "decisions": [
        "run = one configuration over the four disjoint 8-task shards (Kaggle kernels 'sfd-paired-eval-shard-N'); "
        "the lab reports them as one 32-task evaluation per configuration ('2/32'), so shards share the run label",
        "the Cohort-1 rerun (config '<name>_rerun', 8 tasks) is a second run of the same system; system label = "
        "config name without '_rerun'",
        "the 20 runs the lab flags env_suspected=True (pytest collection crashes it attributes to missing wheels / "
        "path shadowing in the sandbox) are KEPT as y=0: the flag is 'suspected' only, the official harness scored "
        "them, and the lab's primary numbers (2/32 per config) include them; its sensitivity analysis without them "
        "gives 6/80 primary",
        "two resolved attempts have patch_chars=0 (fastapi_14077, flat_controlled and sfd, shard 3): kept as y=1 "
        "because the harness reports resolved=True / exit 0",
        "err = paired_progress 'error' is non-empty (session timeout, turns budget or tool-call budget)",
    ],
    "excluded": [
        "traces/**/traces/trace_<task>.json (112) and dataset_resource/traces/cohort_*_<task>.json (112 copies): "
        "ATIF step traces without an outcome field; outcomes come from paired_progress.jsonl",
        "dataset_resource/corpus_metadata.csv, results/resource_summary.json, results/failure_taxonomy_runs.csv "
        "(+ identical dataset_resource copy), results/summary.json, results/rerun_determinism.json, "
        "results/exit1_breakdown.*, results/exit2_breakdown.*, results/remediation_eval.json, results/table*.csv, "
        "results/summary_table.csv: derived tables of the same 112 runs",
        "_retracted_simulation/* (benchmark_results.csv, *_seed42.json): retracted synthetic simulation - "
        "'produced by a simulation, not by running the agent'",
        "results/leaderboard_snapshot_2026_10_26.json, results/small_model_leaderboard_2026_10_24.csv: hidden-test "
        "leaderboard scores of other teams (small-model prize: 197 teams, 0.0-0.10, median 0.05); the author's own "
        "submission 57643078 is PENDING with no score",
        "agent_sfd_31b_v2/, submission_sfd_31b_v2.zip: a 31B variant of the agent with no per-task results",
        "analysis/gold_patch_control.py: gold-patch control was written but not executed (CHANGELOG.md 'T2 "
        "(Gold-Patch Control) Status: BLOCKED ... oracle ceiling remains uncomputed')",
        "benchmarks_ref/*.ipynb: reference notebooks of other participants, no per-task outcomes",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": [],
    "reported_numbers": [
        {"what": "flat_starter / flat_controlled / sfd resolved (32 primary tasks each)", "value": "2/32, 2/32, 2/32",
         "where": "README.md table; KAGGLE_WRITEUP_RESOURCE.md Table 3", "adapter": "2/32, 2/32, 2/32", "match": True},
        {"what": "rerun flat_controlled / sfd resolved (Cohort 1)", "value": "1/8, 1/8",
         "where": "README.md table; KAGGLE_WRITEUP_RESOURCE.md Table 3", "adapter": "1/8, 1/8", "match": True},
        {"what": "resolved traces over all 112 runs", "value": "8/112",
         "where": "KAGGLE_WRITEUP_RESOURCE.md abstract", "adapter": "8/112", "match": True},
        {"what": "methods paper (Cohorts 1+2, 16 tasks): sfd / flat_starter / flat_controlled", "value": "1/16, 1/16, 0/16",
         "where": "KAGGLE_WRITEUP.md abstract; results/summary_table.csv", "adapter": "1/16, 1/16, 0/16", "match": True},
        {"what": "runs ending in budget exhaustion without a patch (exit -1)", "value": "9/112",
         "where": "KAGGLE_WRITEUP_RESOURCE.md Table 2",
         "adapter": "12 err=True rows, 9 of them with test_exit_code -1 (other 3: session timeouts with exit 1/2)",
         "match": True},
    ],
}

PROGRESS = re.compile(r"^traces/(?:(shard\d+|rerun)_results/)?eval_results/paired_progress\.jsonl$")


def attempts(src):
    for path, rows in src.jsonl(PROGRESS.pattern):
        for row in rows:
            config = row["config"]
            system = config[:-len("_rerun")] if config.endswith("_rerun") else config
            error = row.get("error")
            yield {
                "file": path,
                "system": system,
                "run": config,
                "task": row["instance_id"],
                "y": int(bool(row["resolved"])),
                "model": "E4B",
                "harness": "official",
                "err": bool(error),
                "status": error or f"exit {row.get('test_exit_code')}",
            }
