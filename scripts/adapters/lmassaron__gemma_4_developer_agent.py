"""Adapter for lmassaron/gemma-4-developer-agent.

Local benchmark (one NVIDIA DGX Spark, the lab's own AgentSimulator) of about fifty
Gemma 4 31B agent configurations: prompt bundles v7-v19, hill-climb prompt iterations,
and LoRA / SVD-compressed LoRA variants v20-v30, on the 3-task smoke suite, a 60-task
suite, a 12-task suite, a 48-task calibration suite and the full 129 tasks.

The same run often appears in several files (merged comparison tables, a copy of the
root-level files under state_of_the_art/benchmarks_data/, a core_12 file that was used
to seed a full_129 file).  Files are read in the order the lab produced them and an
attempt is yielded only the first time its (task, status, duration, turns, patch size)
record is seen, so every attempt is yielded once.  The lab's scripts also re-use cached
results of a hill-climb winner under the next version's label; see CARD["decisions"].
"""
import json
import re

REPO = "lmassaron/gemma-4-developer-agent"

B = "state_of_the_art/benchmarks_data/"
PAPER = "articles/gemma_4_developer_agent_paper_track_writeup.md"

CARD = {
    "eligible": True,
    "licence": {"spdx": "MIT", "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Luca Massaron Permission is hereby "
                                           "granted, free of charge, to any person obtaining a copy of this software'"},
    "models": ["31B"],
    "harness": "local",
    "summary": "Local single-GPU benchmark of Gemma 4 31B agent bundles (prompt versions v7-v19, hill-climb prompt "
               "iterations) and LoRA variants (community adapters, own trajectory-SFT adapter, SVD-compressed versions) "
               "with the lab's own agent simulator; six configurations were run on all 129 tasks.",
    "evidence": {
        "model": B + "bench_core_60.log: '[Model] Loading Base Model: google/gemma-4-31B-it in bfloat16 on cuda:0...' (runs up "
                     "to v19); " + B + "benchmark_v22_fastpaced_lora_full129.log: 'Loading QAT model from "
                     "/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct onto cuda:0...' (runs from v20)",
        "harness": "state_of_the_art/16_FOUR_WAY_COMPARATIVE_BENCHMARK_REPORT.md: 'the complete 129-task suite ... is currently "
                   "executing under our dedicated evaluation harness: scripts/evaluate_v21_finetuned_lora_full.py'; "
                   "state_of_the_art/13_V15_BENCHMARK_VERIFICATION_REPORT.md: 'The simulator threw `[Error: File not found: ]` "
                   "... Patch Applied: Updated `TaskWorkspace` and `execute_tool` in `scripts/run_local_benchmark_leaderboard.py`'; "
                   "logs: 'Timeout: 240s | Max Turns: 20'",
        "systems": PAPER + ": 'We evaluated all eight agent iterations across both standardized subsets (`core_12`) and the "
                           "complete 129-task SWE-bench benchmark.' (table rows v20_scottweeden, v20_loopbreaker, v21_finetuned, "
                           "v21_plus, v22_fastpaced, v24_rank4_svd); result files carry the lab's own `label` / candidate key",
        "replicates": "v22_fastpaced_lora: both result files have \"label\": \"v22_fastpaced_lora\" with \"suite\": "
                      "\"calibration_48\" and \"full_129\"; both logs say 'Using bundle configuration from: "
                      "state_of_the_art/bundle_v22', 'Attaching Fine-Tuned LoRA from .../models/lora_v21_scottweeden_finetuned', "
                      "'Timeout: 240s | Max Turns: 20', and each starts with 'Completed: 0'. "
                      "v24_rank4_svd: submissions_archive/SUBMISSIONS_LOG.md 'v24 (Rank-4 SVD Compressed LoRA) ... Solves 3/12 "
                      "(25%) on core_12 and 14/129 locally'; both logs: bundle_v22 + models/lora_v24_rank4_svd. "
                      "v20_scottweeden: " + PAPER + " table row '`v20_scottweeden` | Community LoRA (Scott Weeden) | ... | 1 / 12 "
                      "(8.3%) | 11 / 129 (8.53%)'; scripts/evaluate_lora_only.py (core_12: bundle_v20, timeout_sec=240, "
                      "max_turns=20) and the full_129 queue (bundle_v20, 'Timeout: 240s | Max Turns: 20') both attach "
                      "/home/lmassaron/models/kaggle/lora_scottweeden. "
                      "v17_hill_climbed: LEADERBOARD.md rows '**v17_hill_climbed** | `climb_11` | **63.6%** | 7/11 | 158.7s' and "
                      "'**v17_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 225.4s' are the winner iter2 of "
                      "hill_climb_v16_to_v17_results.json (7/11, mean_duration_sec 158.69) and the baseline 'v17_baseline' of "
                      "hill_climb_v17_to_v18_results.json (6/12, 225.37); both climb scripts run 360 s / 22 turns. "
                      "v20_loopbreaker: " + PAPER + " table row '`v20_loopbreaker` | Community LoRA (Robin Amir Bahar) "
                      "| 217 KB | ... | 1 / 12 (8.3%) | 11 / 129 (8.53%)'; scripts/evaluate_loopbreaker_lora.py '(Robin Amir "
                      "Bahar - Loop Breaker) on core_12' (bundle_v20, 240 s / 20 turns) and "
                      "scripts/run_community_loras_full129_queue.py 'v20_loopbreaker' (bundle_v20) attach the same "
                      "community adapter from two local directory names (lora_robinamirbahar, lora_loopbreaker). "
                      "In all five pairs no per-task record is shared between the two runs (separate executions).",
        "task_sets": "LEADERBOARD.md: '`smoke_3`: 3 representative tasks ... for rapid timeout verification and regression "
                     "testing', '`core_12`: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich)'; " + B + "test_tasks_core_60.json; "
                     + PAPER + ": 'We constructed a 48-task calibration suite composed of 17 Guard Tasks (previously resolved in "
                               "`v21`) and 31 Near-Miss Tasks.'",
    },
    "decisions": [
        "harness = local for every run: the lab's AgentSimulator/TaskWorkspace (transformers on one GPU), not swegemma; its "
        "per-task budget changed between script generations (compare_*.py 330-360 s / 20 turns, benchmark_large_suite.py "
        "300 s / 30 turns, run_local_benchmark_leaderboard.py default 360 s / 20 turns, evaluate_*.py 240 s / 20 turns)",
        "model = 31B for every run; runs up to v19 load google/gemma-4-31B-it (bfloat16 in the scripts), runs from v20 the "
        "competition QAT build gemma-4-31b-it-qat-w4a16-ct",
        "y = the `resolved` flag; statuses TIMEOUT, NO_PATCH and FAILED are failures; err = status is TIMEOUT",
        "exception: a record marked RESOLVED with an empty patch (patch size 0) is y = 0. There are three, all "
        "rich_3468 (flexonafft_v2, v14_heredoc_synthesis and v15_surgical_triage on core_60). The lab calls this an "
        "'Environmental False Positive' (state_of_the_art/14_SOTA_30B_QUANTIZED_SWE_BENCH_RESEARCH.md: 'Pytest suite "
        "has early returns ... that exit 0 on unedited workspaces. Empty-Diff Guard: Never evaluate or mark resolved "
        "if `git diff HEAD` is 0 bytes') and its later scorer has that guard (run_local_benchmark_leaderboard.py: "
        "`resolved = passed and has_patch`). No other resolved record in the source has an empty patch (found by an "
        "independent audit; DEVIATIONS.md, Amendment 4)",
        "the eight early core_60 runs (v10, v11, v12, v13_non_empty_patch, v13_claude_clean, flexonafft_v2, v14, v15; "
        "480 attempts, 0 or 1 of 60 resolved) are KEPT because the lab presents them as valid results and the plan "
        "drops attempts only where the source marks them invalid. An independent audit of the result files shows "
        "what they measure: every test session on a fastapi task in these runs ends in 'ModuleNotFoundError: No "
        "module named 'starlette'' (115 of 115 sessions) and every one on a requests task in an import error for "
        "urllib3 (22 of 22), so 37 of the 60 tasks could not be scored as resolved whatever the patch; the "
        "simulator's tool layer lost most edits (26 of 29 edit calls of v15 arrived without a path; non-empty "
        "patches in 2, 4, 0, 0, 1, 0, 2 of 60 attempts of the seven JSON-recorded runs against 34 of 60 in v16). "
        "One correct fix is scored as failed five times (fastapi_14786, the same 406-byte patch that is RESOLVED "
        "in all 18 executions from v16 on). From v16 on the import errors are gone. The sensitivity analysis S3 "
        "removes the eight runs",
        "scorer (the lab's own, scripts/run_local_benchmark_leaderboard.py): only the test functions that the "
        "task's test patch adds are run (no check of previously passing tests), in the lab's own Python "
        "environment with a 45 s pytest limit (rich_4006, rich_3772 and rich_4070 always time out there and are "
        "never resolved); an attempt that runs out of time is recorded as TIMEOUT without being tested (229 pooled "
        "attempts are TIMEOUT with a non-empty patch)",
        "system = the lab's label / candidate key with the suite suffix (_core12, _full129) removed; "
        "'v20_lora_scottweeden' (core_12 file) and 'v20_scottweeden_full129' are the same system v20_scottweeden",
        "five systems have two runs of 10 or more tasks: v22_fastpaced_lora (calibration_48, full_129), v24_rank4_svd (core_12, "
        "full_129), v20_scottweeden (core_12, full_129), v20_loopbreaker (core_12, full_129), v17_hill_climbed (climb_11, "
        "climb_12_expanded); see evidence.replicates. None is a designed repeat: the second run is the same "
        "configuration on a larger or another suite",
        "the two Loop-Breaker runs ('v20_lora_loopbreaker' on core_12, 'v20_loopbreaker' on full_129) are one system, "
        "like the two Scott Weeden runs: the paper table gives one row for both and both scripts attach the same "
        "community adapter with bundle_v20; only the local directory name differs (lora_robinamirbahar, "
        "lora_loopbreaker). The first version of this adapter kept them apart on the directory name (changed after "
        "an independent audit; DEVIATIONS.md, Amendment 4)",
        "v21_finetuned_lora is one run of 129 tasks: the full_129 file was resumed from the 12 core_12 results ('[Resumption] "
        "Found 12 already completed tasks in benchmark_v21_finetuned_lora_full129_results.json'); the core_12 file is a copy of "
        "those 12 records and yields nothing",
        "v20_scottweeden full_129 was run in two sessions (resumed after 26 tasks); it is one run",
        "every smoke_3 run (3 tasks) is its own system: budgets differ between the scripts and nothing calls them repeats; all "
        "are below the harmonizer's 10-task minimum. The lab documents simulator bugs for the v7-v13 smoke runs "
        "(06_FORENSIC_EVALUATION_AND_V12_BLUEPRINT.md section 4: tool arguments passed as a literal 'command: ...' string; "
        "07_CLAUDE_CRITIQUE_AND_HEREDOC_ARCHITECTURE.md 1.3: '`max_new_tokens` was capped at 512', so the generation was cut "
        "before the tool call)",
        "hill-climb files: every iteration id is a system with one run on the 11- or 12-task climb suite, except that the "
        "winner of a climb carries the name of the version it became and the baseline of the next climb is that version again: "
        "iter2 and 'v17_baseline' = v17_hill_climbed, v18_iter3 and 'v18_baseline' = v18_hill_climbed, v19_iter5 = "
        "v19_hill_climbed (LEADERBOARD.md lists these runs under the version names with the same counts and mean durations as "
        "the JSON winner / baseline entries); the baseline of the v16-to-v17 climb has no per-task rows",
        "v17_hill_climbed therefore has two runs (climb_11 and climb_12_expanded); note that its first run is the selected "
        "winner of four tried iterations, the second a fresh run of the same prompt",
        "cached results re-used under a new label are yielded once: 5 records of benchmark_v19_clean_60_results.json are the "
        "records of climb iteration v19_iter5 and are yielded with the v19 core_60 run, as the lab counts them (14/60), leaving "
        "2 dev tasks in the v19 climb_12_cross run; 6 records of 'v18_baseline' in the v18-to-v19 climb are the records of "
        "v18_iter3 and stay with that run, leaving 1 dev task in the v18 climb_12_cross run",
        "hill-climb scripts run 330-360 s / 22 turns (scripts/hill_climb_v17.py, _v18.py, _v19.py); benchmark_large_suite.py "
        "takes the budget from each bundle's eval_config (default 300 s / 30 turns)",
        "'+hints' in a system label: the prompt holds task-specific fixes (07_...md 1.2: 'In `v11`, `v12`, and `v13`, the system "
        "prompt contained the exact answer for `requests_7205` ... and `rich_2725`'; SUBMISSIONS_LOG.md v19: 'prompt full of "
        "task-specific gold-fix hints learned on 12 local tasks'). The label marks the prompt, not the outcome: the two "
        "hinted tasks of v11-v13 are not among the 60 tasks of their core_60 runs. The v18 climb iterations are not "
        "labelled although bundle_v18 carries rules derived from two climb tasks (rich_3486, fastapi_14372)",
        "'+lora' is appended to LoRA systems whose label does not say so; '+partial' marks the hill-climb iteration stopped after "
        "1 of 11 tasks",
        "SymPy tasks inside the v18-to-v19 climb suite are skipped (SWE-bench tasks, not the dev set); those runs keep 7 dev tasks",
        "two runs exist only as console logs and are parsed from them: v20_base_budget_fit on core_12 "
        "(benchmark_v20_vs_lora_core12.log) and the public bundle flexonafft_v2 on core_60 (bench_flexonafft_vs_v14.log)",
        "the calibration_48 suite was chosen from the outcomes of the v21 full_129 run (its 17 resolved tasks plus 31 "
        "'near-miss' tasks), so the two v22 runs overlap on a selected, not a random, set of 48 tasks",
    ],
    "excluded": [
        B + "benchmark_comparison_results.json, benchmark_v9_vs_v10.json, benchmark_v10_vs_v11.json, benchmark_v10_v11_v12.json, "
            "benchmark_v10_v11_v12_v13.json: byte-identical copies of the root-level files",
        "benchmark_v10_v11_v12.json, benchmark_v10_v11_v12_v13.json, " + B + "benchmark_5way_v10_to_v13_claude.json: merged "
        "tables that repeat the v10/v11/v12/v13 smoke runs of earlier files; only the candidate new in each file is yielded",
        B + "benchmark_v21_finetuned_lora_core12_results.json: its 12 records are the first 12 records of the full_129 file",
        B + "*.log: console output of runs already yielded from their JSON result files, except the two log-only runs "
            "(yielded) and four interrupted first attempts that the lab re-ran from scratch and that exist only in a log: "
            "hill_climb_v17.log iter4 (6 tasks, crashed with CUDA out of memory), hill_climb_v18.log v18_iter2 (10 tasks, "
            "process restarted), hill_climb_v19_and_core60.log v19_iter2 (11 and 1 tasks, interrupted twice). They are "
            "superseded by the complete re-runs, which are yielded",
        B + "bench_flexonafft_vs_v14.log, candidate v14_heredoc_synthesis: interrupted first run (5 tasks, all NO_PATCH, 138 "
            "'[NO TOOL PARSED]' turns), replaced by the lab's re-run benchmark_v14_clean_60_results.json",
        B + "benchmark_v20_vs_lora_core12.log, 'Run 2: LoRA Model': crashed while attaching the adapter ('ValueError: Target "
            "module Gemma4ClippableLinear'), no task ran",
        B + "benchmark_v18_external_hf_py312_results.json, external_hf_benchmark_py312.jsonl and the SymPy tasks in "
            "hill_climb_v18_to_v19_results.json: SWE-bench tasks, not the competition dev set (lab: v18 5/12 on them)",
        B + "hill_climb_v16_to_v17_results.json baseline 'v16_phased_edit_enforcement' (5/11): aggregate only",
        "evaluation_results.json: control, not an agent run - reference (gold) patch check of requests_7205 by "
        "scripts/evaluate_local.py, resolved (1 of 1 task checked)",
        "evaluation_sft_comparison.json: validation loss / perplexity of LoRA adapters for E2B, 12B and 31B, no task outcomes",
        "paper table 'v20_base ... ~12 / 129': no per-task rows",
        "leaderboard.json: the lab's LOCAL leaderboard (LEADERBOARD.md: 'Local Benchmark Leaderboard'), 15 runs with "
        "per-task rows; 8 are copies of result files read above and 7 are further 3-task smoke executions (one with "
        "gemma-4-12b-sft-v6, one v10, five v16), all below the 10-task minimum, so nothing is read from it",
        "submissions_archive/SUBMISSIONS_LOG.md LB scores: Kaggle hidden-test leaderboard, aggregates",
        B + "test_tasks_*.json(l), documents/, state_of_the_art/flexonafft_v2/, matterhorn3838/: task lists and third-party notebooks",
    ],
    "dropped": {"interrupted first v14 run in bench_flexonafft_vs_v14.log (tool calls not parsed; re-run by the lab)": 5},
    "recoded": {"RESOLVED with an empty patch (rich_3468 in flexonafft_v2, v14, v15) -> y = 0": 3},
    "flags": ["fine_tuned", "oracle_hints", "partial_runs", "local_simulator_bugs_documented", "log_parsed_runs",
              "replicate_tasks_selected_on_outcomes", "early_runs_scored_in_a_broken_test_environment",
              "timed_out_attempts_not_tested"],
    "reported_numbers": [
        {"what": "v19_hill_climbed on core_60", "value": "14/60", "where": "LEADERBOARD.md", "adapter": "14/60", "match": True},
        {"what": "v16_phased_edit_enforcement on core_60", "value": "12/60", "where": "LEADERBOARD.md", "adapter": "12/60",
         "match": True},
        {"what": "flexonafft_v2, v14_heredoc_synthesis, v15_surgical_triage on core_60", "value": "1/60 each",
         "where": "state_of_the_art/13_V15_BENCHMARK_VERIFICATION_REPORT.md",
         "adapter": "0/60 each: the one 'resolved' task of each run, rich_3468, has an empty patch and is recoded to "
                    "y = 0 (see decisions)", "match": False},
        {"what": "v12_native_discipline on the 60 tasks", "value": "0.00 (60-Task)",
         "where": "state_of_the_art/04_EMPIRICAL_BENCHMARK_RESULTS.md", "adapter": "0/60", "match": True},
        {"what": "v17_hill_climbed on climb_11 (= iteration iter2)", "value": "7/11", "where": "LEADERBOARD.md",
         "adapter": "7/11", "match": True},
        {"what": "v17_hill_climbed on climb_12_expanded (= v17_baseline)", "value": "6/12", "where": "LEADERBOARD.md",
         "adapter": "6/12", "match": True},
        {"what": "v18_hill_climbed on climb_12_expanded (= v18_iter3)", "value": "6/12", "where": "LEADERBOARD.md",
         "adapter": "6/12", "match": True},
        {"what": "v19_hill_climbed on climb_12_cross (= v19_iter5)", "value": "8/12", "where": "LEADERBOARD.md",
         "adapter": "5/7 on its dev tasks (4/5 of them are yielded inside the v19 core_60 run that re-uses them, 1/2 in the "
                    "climb run); the 5 SymPy tasks of the suite (3 resolved) are not dev tasks and are skipped", "match": False},
        {"what": "core_12 four-way: v20_base, Scott Weeden LoRA, Loop-Breaker LoRA, v21 fine-tuned LoRA",
         "value": "1/12, 1/12, 1/12, 4/12", "where": "state_of_the_art/16_FOUR_WAY_COMPARATIVE_BENCHMARK_REPORT.md",
         "adapter": "1/12, 1/12, 1/12, 4/12 (v21: the 12 core tasks inside its 129-task run)", "match": True},
        {"what": "full 129: v20_scottweeden, v20_loopbreaker, v21_finetuned, v21_plus, v22_fastpaced, v24_rank4_svd",
         "value": "11, 11, 17, 14, 17, 14 of 129", "where": PAPER + " (results table)",
         "adapter": "11, 11, 17, 14, 17, 14 of 129", "match": True},
        {"what": "v22 on the 48 calibration tasks", "value": "20/48", "where": PAPER + " section 4.2", "adapter": "20/48",
         "match": True},
        {"what": "v24_rank4_svd on core_12", "value": "3/12", "where": "submissions_archive/SUBMISSIONS_LOG.md", "adapter": "3/12",
         "match": True},
        {"what": "v29 and v30 on core_12", "value": "2/12 and 3/12", "where": "submissions_archive/SUBMISSIONS_LOG.md",
         "adapter": "2/12 and 3/12", "match": True},
        {"what": "v22_fastpaced core_12 column", "value": "4/12", "where": PAPER + " (results table)",
         "adapter": "no core_12 run file; the full_129 run resolves 5 of the 12 core tasks", "match": False},
        {"what": "v16_phased_edit_enforcement on smoke_3", "value": "2/3", "where": "LEADERBOARD.md", "adapter": "2/3",
         "match": True},
    ],
}

# Files in the order the lab produced them, with two exceptions: a results file that absorbed records of an
# earlier session is read before that session's file, so that the complete run stays in one piece
# (v21 full_129 before v21 core_12; v19 core_60 before the v18-to-v19 climb whose winner it re-uses).
ORDER = [
    "benchmark_comparison_results.json",
    "benchmark_v9_vs_v10.json",
    "benchmark_v10_vs_v11.json",
    "benchmark_v10_v11_v12.json",
    "benchmark_v10_v11_v12_v13.json",
    B + "benchmark_5way_v10_to_v13_claude.json",
    "benchmark_core_60_results.json",
    B + "bench_flexonafft_vs_v14.log",
    B + "benchmark_v14_clean_60_results.json",
    B + "benchmark_v15_smoke_3_results.json",
    B + "benchmark_v15_clean_60_results.json",
    B + "benchmark_v16_smoke_3_results.json",
    B + "benchmark_v16_clean_60_results.json",
    B + "hill_climb_v16_to_v17_results.json",
    B + "hill_climb_v17_to_v18_results.json",
    B + "benchmark_v19_clean_60_results.json",
    B + "hill_climb_v18_to_v19_results.json",
    B + "benchmark_v20_vs_lora_core12.log",
    B + "benchmark_v20_lora_core12_results.json",
    B + "benchmark_v20_loopbreaker_core12_results.json",
    B + "benchmark_v21_finetuned_lora_full129_results.json",
    B + "benchmark_v21_finetuned_lora_core12_results.json",
    B + "benchmark_v21_plus_lora_full129_results.json",
    B + "benchmark_v22_fastpaced_lora_calibration48_results.json",
    B + "benchmark_v22_fastpaced_lora_full129_results.json",
    B + "benchmark_v24_rank4_svd_core12_results.json",
    B + "benchmark_v24_rank4_svd_full129_results.json",
    B + "benchmark_v20_scottweeden_full129_results.json",
    B + "benchmark_v20_loopbreaker_full129_results.json",
    B + "benchmark_v25_rank6_svd_core12_results.json",
    B + "benchmark_v25_rank8_attn_svd_core12_results.json",
    B + "benchmark_v28_full_rank6_svd_core12_results.json",
    B + "benchmark_v28_full_rank8_svd_core12_results.json",
    B + "benchmark_v28_scottweeden_core12_results.json",
    B + "benchmark_v29_fastpaced_attn_svd_core12_results.json",
    B + "benchmark_v30_tuned_attn_lora_core12_results.json",
]

ALIAS = {
    "v20_lora_scottweeden": "v20_scottweeden",
    "v20_lora_loopbreaker": "v20_loopbreaker",      # the same community adapter on core_12 and on full_129
    # the winner of a hill climb is the next version, and the baseline of the next climb is that version again
    "hill_climb_v16_to_v17/iter2_compound_locate_and_zero_result_pytest_escape": "v17_hill_climbed",
    "hill_climb_v17_to_v18/v17_baseline": "v17_hill_climbed",
    "hill_climb_v17_to_v18/v18_iter3_single_existing_pytest_gate": "v18_hill_climbed",
    "hill_climb_v18_to_v19/v18_baseline": "v18_hill_climbed",
    "hill_climb_v18_to_v19/v19_iter5_sampling_sharpening": "v19_hill_climbed",
}
CLIMB_SUITE = {"hill_climb_v16_to_v17": "climb_11", "hill_climb_v17_to_v18": "climb_12_expanded",
               "hill_climb_v18_to_v19": "climb_12_cross"}     # suite names of LEADERBOARD.md
HINTS = {"v11_targeted_playbooks", "v12_native_discipline", "v13_non_empty_patch", "v19_hill_climbed"}
LORA_NOT_IN_NAME = {"v20_scottweeden", "v20_loopbreaker", "v24_rank4_svd", "v25_rank6_svd", "v25_rank8_attn_svd",
                    "v28_full_rank6_svd", "v28_full_rank8_svd", "v28_scottweeden", "v29_fastpaced_attn_svd"}
SUITE_IN_NAME = (("full129", "full_129"), ("core12", "core_12"), ("calibration48", "calibration_48"),
                 ("_60", "core_60"), ("smoke_3", "smoke_3"))
LOG_ONLY = {B + "bench_flexonafft_vs_v14.log": {"flexonafft_v2": "core_60"},
            B + "benchmark_v20_vs_lora_core12.log": {"v20_base_budget_fit": "core_12"}}


def _rows(tasks):
    """(task, status, resolved, duration, turns, patch size) per task record, in file order."""
    return [(t["instance_id"], t.get("status"), bool(t.get("resolved")), t.get("duration_sec"), t.get("turns"),
             t.get("patch_size")) for t in tasks if isinstance(t, dict) and t.get("instance_id")]


def _suite(path, n, declared=None):
    if declared:
        return declared
    name = path.rsplit("/", 1)[-1]
    for key, suite in SUITE_IN_NAME:
        if key in name:
            return suite
    return "smoke_3" if n == 3 else f"n{n}"


def _json_runs(path, obj):
    """[(label, suite, rows, partial)] for the three shapes of result file."""
    out = []
    if not isinstance(obj, dict):
        return out
    if isinstance(obj.get("tasks"), list):                       # one run: {"label"|"model_name", "suite"?, "tasks"}
        rows = _rows(obj["tasks"])
        out.append((obj.get("label") or obj.get("model_name"), _suite(path, len(rows), obj.get("suite")), rows, False))
    elif "iterations" in obj:                                    # hill climb: baseline + iterations
        tag = path.rsplit("/", 1)[-1].replace("_results.json", "")
        for it in [obj.get("baseline") or {}] + list(obj.get("iterations") or []):
            if isinstance(it, dict) and isinstance(it.get("tasks"), list) and it["tasks"]:
                rows = _rows(it["tasks"])
                out.append((f"{tag}/{it.get('id')}", CLIMB_SUITE.get(tag, "climb"), rows,
                            len(rows) < int(it.get("total") or 0)))
    else:                                                        # {candidate: [tasks]} or {candidate: {"tasks": [...]}}
        for cand, val in obj.items():
            tasks = val.get("tasks") if isinstance(val, dict) else val
            if isinstance(tasks, list):
                rows = _rows(tasks)
                out.append((cand, _suite(path, len(rows)), rows, False))
    return out


LOG_TASK = re.compile(r"^\s*(?:--- )?\[\d+/\d+\] (?:Candidate: (\S+) \| )?Task: (\S+)")
LOG_START = re.compile(r"STARTING BENCHMARK: (\S+) \(\d+ tasks\)")
LOG_RESULT = re.compile(r"Result(?: ->|:) [Ss]tatus[:=]\s?(\w+) \| [Rr]esolved[:=]\s?(True|False) \| (?:dur=|Duration: )([\d.]+)s"
                        r" \| [Tt]urns[:=]\s?(\d+) \| (?:patch=|Patch: )(\d+)")


def _log_runs(path, text):
    """Runs that exist only as console logs: '[k/n] ... Task: <id>' followed by a 'Result' line."""
    wanted, found, label, task = LOG_ONLY[path], {}, None, None
    for line in text.splitlines():
        m = LOG_START.search(line)
        if m:
            label = m.group(1)
            continue
        m = LOG_TASK.match(line)
        if m:
            label, task = m.group(1) or label, m.group(2)
            continue
        m = LOG_RESULT.search(line)
        if m and task and label in wanted:
            found.setdefault(label, []).append((task, m.group(1), m.group(2) == "True", float(m.group(3)),
                                                int(m.group(4)), int(m.group(5))))
        if m:
            task = None
    return [(label, wanted[label], rows, False) for label, rows in found.items()]


def _system(label):
    name = re.sub(r"_(core12|full129)$", "", label)
    name = ALIAS.get(name, name)
    if name in HINTS:
        name += "+hints"
    if name in LORA_NOT_IN_NAME:
        name += "+lora"
    return name


def attempts(src):
    pattern = "^(" + "|".join(re.escape(p) for p in ORDER) + ")$"
    runs = {}
    for path, raw in src.read_many(pattern):
        text = raw.decode("utf-8", "replace")
        if path.endswith(".log"):
            runs[path] = _log_runs(path, text)
        else:
            try:
                runs[path] = _json_runs(path, json.loads(text))
            except ValueError:
                runs[path] = []
    seen = set()
    for path in ORDER:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for label, suite, rows, partial in runs.get(path, []):
            if not label:
                continue
            system = _system(label) + ("+partial" if partial else "")
            if suite == "smoke_3":
                system = run = f"{system}@smoke_3:{stem}"
            else:
                run = f"{system}@{suite}"
            for row in rows:
                task, status, resolved = row[0], row[1], row[2]
                if row in seen:            # the same attempt, already yielded from an earlier file
                    continue
                seen.add(row)
                if "__" in task:           # SWE-bench style id (sympy__sympy-...), not a dev task
                    continue
                status = status or ("RESOLVED" if resolved else "FAILED")
                if resolved and row[5] == 0:
                    # scored as resolved with an empty patch: the lab's "Environmental False Positive"
                    resolved, status = False, "NO_PATCH (scored RESOLVED with an empty patch)"
                yield {
                    "file": path,
                    "system": system,
                    "run": run,
                    "task": task,
                    "y": int(resolved),
                    "model": "31B",
                    "harness": "local",
                    "err": status == "TIMEOUT",
                    "status": status,
                }
