"""Adapter for msingatullin/gemma-4-developer-agent ("AST-Guided Verification Suite for Gemma 4"). Not eligible."""

REPO = "msingatullin/gemma-4-developer-agent"

CARD = {
    "eligible": False,
    "reason": ("no attempt on the competition's public development set: the snapshot contains no task id of the form "
               "fastapi_/rich_/requests_/httpx_<number>. Its per-task outcomes are (a) single-call patch generations by Gemma 4 "
               "31B on 15 synthetic self-contained bug reproductions (task_01_flask_blueprint_dot ...), (b) a 6-task E4B "
               "'paired pilot' on SWE-bench ids that the lab's own STATUS.md calls a scripted simulation, not model inference, "
               "and (c) gold/base evaluator controls on five SWE-bench tasks; leaderboard scores are aggregate hidden-test numbers"),
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: '**License:** [Apache 2.0](LICENSE)'"},
    "models": [],
    "harness": None,
    "summary": ("Paper-track study of traceback compression: 150 single-shot Gemma 4 31B responses on 15 synthetic bug "
                "reproductions (two seeds), plus locator benchmarks and controls on SWE-bench tasks; no run on the "
                "competition's development tasks is published."),
    "evidence": {
        "model": "README.md: '**Model used for the empirical study:** `google/gemma-4-31B-it-qat-w4a16-ct`'",
        "harness": ("README.md: 'Generated patches were subsequently evaluated with pytest in an unprivileged Docker environment' "
                    "(own evaluator paper/e2e_gemma_experiment/run_container_execution_eval.py; not the competition harness)"),
        "systems": "README.md table: input variants A (full pytest output), B, B2, C, D (traceback treatments), 30 model calls each",
        "replicates": "README.md: 'evaluated with two seeds (`42`, `100`)' (synthetic tasks only; not used)",
        "task_sets": ("README.md: 'The 15 evaluated tasks are **synthetic, self-contained reproductions**; they are not genuine "
                      "upstream repository checkouts or SWE-bench issue-resolution runs'; paper/small_model_benchmark/MANIFEST.md: "
                      "'Curated from human-verified genuine repository issues (SWE-bench Verified & Lite)'"),
    },
    "decisions": [
        "ids such as psf__requests-1963 are SWE-bench instance ids, not competition tasks; they are not rewritten to requests_1963",
        "a byte search of every text file of the snapshot for (fastapi|rich|requests|httpx)[_-]<3-6 digits> finds only "
        "'Requests-1963' (the SWE-bench case) in two CPU reports",
    ],
    "excluded": [
        "paper/e2e_gemma_experiment/real_run_20261108/real_patch_execution_results.jsonl and real_b2_patch_execution_results.jsonl: "
        "150 single-call Gemma 4 31B patch generations on 15 synthetic tasks x 2 seeds x 5 traceback variants (patches passing "
        "pytest: A 30/30, B 25/30, B2 29/30, C 30/30, D 22/30); synthetic tasks, no agent loop",
        "paper/e2e_gemma_experiment/real_run_20261108/real_experiment_checkpoint.jsonl, real_b2_experiment_checkpoint.jsonl: the "
        "model responses behind those rows (same runs)",
        "paper/small_model_benchmark/PAIRED_E4B_PILOT_RESULTS.json: v15 1/6 vs v17 4/6 on six SWE-bench ids; STATUS.md: "
        "'`run_paired_e4b_eval.py` explicitly uses `simulate_v15_agent`/`simulate_v17_agent` ... This is **not real E4B "
        "inference**'; simulated, SWE-bench tasks",
        "paper/small_model_benchmark/LOCATOR_BENCHMARK_RESULTS.json, STAGE1_IN_SANDBOX_VERIFICATION.json: file-localisation "
        "metrics (recall@k, MRR) on 42 SWE-bench tasks; not outcomes",
        "paper/e2e_gemma_experiment/CPU_EVALUATOR_CONTROLS_20261110.json, real_repo_sample/evaluator_control_validation.json, "
        "CPU_UPSTREAM_RED_GREEN_20261110.json, real_repo_sample/cpu_verification_results.jsonl: controls without a model on five "
        "SWE-bench tasks (gold patch RESOLVED 5/5, unchanged base FAILED_F2P 5/5, 12/12 evaluator controls as expected)",
        "paper/e2e_gemma_experiment/frozen_prompts/, real_repo_sample/, tasks/, variants/, pilot/frozen_task_manifest.json, "
        "paper/small_model_benchmark/manifest.json: prompts, task definitions and manifests, no outcomes",
        "STATUS.md: public leaderboard scores (v11 0.17, v14 0.17 on 31B; v15 0.06, v16 0.05 on E4B): hidden-test aggregates",
    ],
    "dropped": {},
    "flags": ["no_dev_set_tasks", "simulated_results"],
    "reported_numbers": [
        {"what": "patches passing pytest per variant (synthetic tasks, excluded)", "value": "A 30/30, B 25/30, B2 29/30, C 30/30, D 22/30",
         "where": "README.md", "adapter": "not yielded (file counts: A 30/30, B 25/30, B2 29/30, C 30/30, D 22/30)", "match": True},
    ],
}


def attempts(src):
    return
    yield
