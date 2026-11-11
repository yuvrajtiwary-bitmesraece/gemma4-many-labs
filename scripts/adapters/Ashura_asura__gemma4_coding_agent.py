"""Adapter for Ashura-asura/gemma4-coding-agent ("Post-training Gemma 4 into an offline software-engineering agent
on a free T4").

Not eligible: the project never touches the competition's public development set.  Its tasks are SWE-bench
instances (django, sympy, sphinx, xarray, pylint, pytest, flask, requests; ids such as psf__requests-5087), run in
the lab's own sandbox and tool loop, and its one evaluation (24 held-out SWE-bench tasks, base E4B against a QLoRA
adapter) is reported as totals in the paper: 0 of 20 scored tasks resolved in both arms.  The JSONL files in data/
are task definitions and training trajectories, not results.
"""

REPO = "Ashura-asura/gemma4-coding-agent"

CARD = {
    "eligible": False,
    "reason": ("no development-set tasks: every task is a SWE-bench instance (data/holdout_tasks.jsonl 291, "
               "data/holdout_eval_subset.jsonl 24, data/rl_task_pool.jsonl 291; 'source': 'SWE-bench'), evaluated with "
               "the lab's own sandbox and harness, and the result (0/20 resolved for both the base Gemma 4 E4B and its "
               "QLoRA adapter) is published only as totals; no per-task outcome file is in the repository"),
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; DATA_LICENSES.md: '`LICENSE` "
                            "(Apache-2.0) was added by the audit as a **placeholder for the code**'; the same file says "
                            "the data files 'are derived from the SWE-bench test split' and that 'Each upstream project "
                            "keeps its own licence'. README.md has no licence section"},
    "models": [],
    "harness": "local",
    "summary": ("QLoRA SFT and a stopped GRPO attempt on Gemma 4 E4B with SWE-bench-derived tasks, evaluated on 24 "
                "held-out SWE-bench tasks on one Kaggle T4 with the lab's own sandbox and tool loop; a null result "
                "(0/20 in both arms), reported as totals."),
    "evidence": {
        "model": "paper/paper.md s4: 'Model | Gemma 4 E4B, 4-bit NF4 + LoRA'; README.md: 'Kaggle kernel v22 on E4B'",
        "harness": ("README.md: 'Phase 1. Sandbox + tool interface ... 5 tools + loop wired; eval harness runs end-to-end "
                    "with a null policy'; paper/paper.md s4: 'A task counts as resolved only if the submitted patch "
                    "applies to a fresh checkout, touches no test/harness file, and makes the task's tests pass' "
                    "(eval/run_eval.py, eval/scoring_guard.py) - the lab's own grader, not swegemma"),
        "systems": "paper/paper.md Table 1: 'Base Gemma 4 E4B, zero-shot', '+ QLoRA SFT', '+ GRPO (not evaluated)'",
        "replicates": ("none: paper/paper.md 'Single seed, single run. No error bars across seeds'; evaluation decoding "
                       "is greedy"),
        "task_sets": ("data/holdout_eval_subset.jsonl: 24 rows with 'task_id' like 'django__django-10390', "
                      "'psf__requests-5087' and 'source': 'SWE-bench'; README.md: 'The 24-task subset is sampled "
                      "deterministically: holdout rows are bucketed by repo and taken round-robin (4 django, 4 xarray, "
                      "4 pylint, 3 pytest, 3 sphinx, 3 sympy, 2 flask, 1 requests)'"),
    },
    "decisions": [
        "nothing is yielded: SWE-bench ids are outside the public development set (README rule: do not rewrite other "
        "id schemes), and no per-task result exists in the repository anyway",
        "the single 'requests' id that the screening matched is the SWE-bench instance psf__requests-5087, not a "
        "competition task",
    ],
    "excluded": [
        "data/holdout_tasks.jsonl (291), data/holdout_eval_subset.jsonl (24), data/rl_task_pool.jsonl (291): SWE-bench "
        "task definitions (issue, test patch, base commit), no outcomes",
        "data/sft_trajectories.jsonl, data/sft_trajectories.compact.jsonl: constructed tool-use training trajectories "
        "on SWE-bench tasks, not agent runs",
        "paper/paper.md Table 1 and README.md Status: base E4B 0/20 and QLoRA adapter 0/20 on the scored subset (4 of "
        "24 tasks failed environment setup; 12 base episodes ended in CUDA OOM) - totals only, SWE-bench tasks",
    ],
    "dropped": {},
    "flags": ["swe_bench_only", "no_dev_set_tasks", "no_per_task_outcomes"],
    "sampling": ("evaluation decoding is greedy, temperature 0.0 (configs/eval_config.yaml: 'temperature: 0.0  # greedy: "
                 "paired comparison needs one fixed outcome per task'); one run per arm, no seeds varied"),
    "leaderboard": "none",
    "controls": [],
    "reported_numbers": [
        {"what": "held-out SWE-bench subset, resolved per arm", "value": "base 0/20, QLoRA SFT 0/20 (24 attempted, 4 environment failures)",
         "where": "paper/paper.md Table 1; README.md Status", "adapter": "not yielded (not development-set tasks, totals only)",
         "match": False},
    ],
}


def attempts(src):
    return
    yield
