"""Adapter for JayGupta2003/gemma4-swe-agent-posttraining ("Where the Gains Aren't: Noise, Bimodality and the
Limits of LoRA Post-Training for a 4-bit Gemma 4 SWE Agent").

Not eligible: the lab ran many Gemma 4 rounds (31B, 12B, E4B, with and without LoRA adapters) on the 75 public
tasks whose reference patch passes on Kaggle, but the repository holds only the totals of those rounds.  Per-task
results stay in the Kaggle kernel outputs (README: 'Fetch results: kaggle kernels output <kernel id> ...'); the 49
notebooks under kaggle_assets/ are generated code without saved outputs.  The one file with a per-task outcome,
docs/kaggle_env_gold_results.json, is a reference-patch control (see CARD["controls"]).
"""

REPO = "JayGupta2003/gemma4-swe-agent-posttraining"

CARD = {
    "eligible": False,
    "reason": ("aggregated only: the agent rounds on public tasks are reported as totals per round (docs/paper/tables.md, "
               "docs/paper/tables_2.md, docs/paper/figure_data*.json, docs/STRATEGY.md); no per-task agent outcome is in "
               "the repository. The only per-task file, docs/kaggle_env_gold_results.json (129 records), holds "
               "reference-patch results, a non-agent control"),
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## Licence': 'Apache License "
                            "2.0, see LICENSE and NOTICE. The agent prompts in `submission*/prompts/` are adapted from the "
                            "competition's sample submission.'; NOTICE: 'Copyright 2026 Jay Gupta. This product is "
                            "licensed under the Apache License, Version 2.0'"},
    "models": [],
    "harness": "official",
    "summary": ("Post-training study (LoRA SFT, preference training, distillation into E4B) with local rounds on the 75 "
                "public tasks whose reference patch passes in the Kaggle environment (swegemma 0.2.11, Kaggle 4x L4, four "
                "attempts concurrently); the repository publishes code, task lists, the reference-patch check and totals "
                "per round, not per-task agent outcomes."),
    "evidence": {
        "model": ("docs/paper/DRAFT.md s2: 'the model gemma-4-31b-it-qat-w4a16-ct runs on vLLM (tensor parallel 4, "
                  "32,768-token context ...)'; README.md: 'Models: `google/gemma-4` variants `gemma-4-31b-it-qat-w4a16-ct` "
                  "and `gemma-4-e4b-it-qat-w4a16-ct` (served, scored)'; docs/paper/tables.md Table 2 also has a '12B base' "
                  "row"),
        "harness": ("README.md: 'Everything runs on Kaggle notebooks (4x NVIDIA L4, offline) with the competition's own "
                    "harness (`swegemma`, `adk_submission`, vLLM) from the official wheelhouse'; docs/paper/DRAFT.md s2: "
                    "'Local evaluation runs the scorer's harness (swegemma 0.2.11) on the 75 competition tasks whose "
                    "reference patch passes in our environment, four attempts concurrently'"),
        "systems": ("docs/paper/tables.md Table 2 and tables_2.md Table 4: rounds 'base r1', 'base r2', 'G3 adapter', 'G11 "
                    "adapter', '12B base', four E4B base rounds (three prompts, one with capped reasoning) and five E4B + "
                    "LoRA / distillation rounds, each with a resolved total of 75 (or of the tasks run)"),
        "replicates": ("docs/paper/DRAFT.md s3: 'Two rounds of the same agent solve 25 and 28 of 75; 7 of the 30 tasks ever "
                       "solved flip'; docs/paper/tables_2.md Table 5 labels them 'base r1 (T 0.2)' and 'base r2 (T 0.2)'. "
                       "Totals only, so no run is yielded"),
        "task_sets": ("README.md: 'We publish task IDs and per-task outcomes only (`docs/kaggle_env_gold_results.json`, "
                      "`docs/exp2_task_sets.json`, `docs/g6_screen_tasks.json`, `docs/g9_rollout_pool.json`)'; "
                      "docs/exp2_task_sets.json splits the 75 passing tasks into 'discovery' (48) and 'confirmation' (27); "
                      "docs/g6_screen_tasks.json lists 32 of them; docs/g9_rollout_pool.json lists 171 SWE-rebench ids"),
    },
    "decisions": [
        "no attempt is yielded: the README's 'per-task outcomes' are the reference-patch results; every agent result is "
        "a total (rule 'Aggregates only')",
        "the notebooks under kaggle_assets/ were checked for saved cell outputs: none of the 49 has any, so no per-task "
        "result can be read from them (their 75 task ids are the hard-coded task list)",
        "docs/STRATEGY.md names single tasks in passing (for example 'fastapi_15589 failed in run 6 but resolved in run "
        "7'); these remarks are not a per-task table and are not used",
    ],
    "excluded": [
        "docs/kaggle_env_gold_results.json: reference patch of all 129 public tasks run through verify_task in the "
        "Kaggle environment, 75 pass (control, see 'controls')",
        "docs/paper/tables.md Table 2 (31B / 12B rounds on the 75 tasks, 5-minute cap): base r1 25, base r2 28, G3 "
        "adapter 28, G11 adapter 22, 12B base 17 resolved of 75 - totals only",
        "docs/paper/tables_2.md Table 4 (E4B, 8 min / 28 calls): E4B base 10, 7, 8 (three prompts), E4B + LoRA (31B "
        "traces) 9, E4B + LoRA (Qwen3-Coder episodes) 3, on-policy distillation 6 of 60 run, the same with reasoning "
        "capped 9, E4B base reasoning capped 11, action-only distillation 6 of 44 run - totals only",
        "docs/paper/tables_2.md Table 5 and docs/paper/figure_data_2.json: number of tasks solved in exactly one of two "
        "rounds (base r1 vs r2: 7; against the temperature-1.0 round: 9-13); docs/paper/figure_data.json: tasks by "
        "number of the four 31B attempts that resolved them (0: 43, 1: 5, 2: 2, 3: 6, 4: 19) - aggregated only",
        "docs/STRATEGY.md: further rounds as totals (G17 preference adapter 26/75, temperature 1.0 29/75, thinking-budget "
        "arms 12 and 11 of 32 against 10 and 11, a sequential 24-task B2 run 9/24, Experiment 1 on 30 tasks)",
        "docs/g9_rollout_pool.json and the pool rounds (171 SWE-rebench tasks; 55, 55, 52 resolved): not the "
        "competition's development set, and totals only",
        "docs/paper/tables.md Table 1: public-leaderboard scores (hidden tasks)",
        "docs/exp2_task_sets.json, docs/g6_screen_tasks.json: task lists without outcomes",
    ],
    "dropped": {},
    "flags": ["aggregated_only", "no_per_task_outcomes", "controls_only"],
    "sampling": ("temperature 0.2, top_p 0.95, thinking budget 4096, max_output_tokens 8192 (submission*/configs/"
                 "sampling.yaml; docs/paper/DRAFT.md s2), no seed stated; the two base rounds are plain repeats at "
                 "temperature 0.2 ('base r1 (T 0.2)', 'base r2 (T 0.2)'); one extra round at temperature 1.0, top_p 0.95, "
                 "top_k 64 (docs/paper/DRAFT.md s3)"),
    "leaderboard": [
        "no system has runs in this adapter; the lab states these pairs (docs/paper/tables.md Table 1 and Table 2, "
        "docs/paper/DRAFT.md Table 1; public leaderboard = 58 tasks):",
        "5-minute prompt B1, 31B: public 0.13 and 0.15; local 25 and 28 of 75 (base r1, base r2, 5-minute cap)",
        "G3 SFT adapter on B1: public 0.08; local 28 of 75",
        "B2 (8 min / 28 tool calls), 31B: public 0.17 and 0.13 for two submissions of byte-identical files; local "
        "sequential run 9 of 24",
        "B3 (B2 + thinking 2048): public 0.15; baseline (starter prompt, 5 min cap): public 0.12; B4 exceeded 12 h",
        "E4B base, B2 prompt: public 0.06; local 7 of 75 (10 and 8 of 75 with two other prompts)",
    ],
    "controls": [
        {"path": "docs/kaggle_env_gold_results.json", "kind": "reference_patch",
         "environment": ("CPU-only Kaggle notebook (kaggle_assets/gold_check/), competition harness verify_task, official "
                         "image and wheelhouse, subprocess sandbox; 2026-10-04"),
         "tasks": 129, "resolved": 75,
         "note": ("by repository: fastapi 31/67, rich 43/48, requests 1/13, httpx 0/1; docs/STRATEGY.md adds that 52 pass "
                  "on the lab's own machine, 49 in both places")},
    ],
    "reported_numbers": [
        {"what": "public tasks whose reference patch passes in Kaggle's environment", "value": "75/129 (fastapi 31/67, rich 43/48, requests 1/13, httpx 0/1)",
         "where": "docs/STRATEGY.md 'Dev-pool RESULT (2026-10-04)'",
         "adapter": "not yielded; docs/kaggle_env_gold_results.json gives 75/129 with the same split", "match": True},
        {"what": "resolved of 75 per round, 31B and 12B", "value": "base r1 25, base r2 28, G3 adapter 28, G11 adapter 22, 12B base 17",
         "where": "docs/paper/tables.md Table 2", "adapter": "not available per task", "match": False},
        {"what": "resolved of 75 per round, E4B", "value": "10, 7, 8 (base, three prompts); 9, 3, 6 of 60, 9, 11, 6 of 44",
         "where": "docs/paper/tables_2.md Table 4", "adapter": "not available per task", "match": False},
    ],
}


def attempts(src):
    return
    yield
