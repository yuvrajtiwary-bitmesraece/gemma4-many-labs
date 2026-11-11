"""Adapter for fjcarmona78/gemma4-agent-noise-study ("Most Agent Improvements Are Noise").

Not eligible: every released Gemma 4 attempt is on a task outside the competition's development set.
"""

REPO = "fjcarmona78/gemma4-agent-noise-study"

CARD = {
    "eligible": False,
    "reason": "no Gemma 4 attempt on a competition dev-set task is released: results/task_attempts.csv (4,362 Gemma 4 attempts, "
              "110 runs) covers only SWE-rebench and OOD-45 tasks (README.md: 'All 216 tasks come from public sources outside "
              "the competition data.'; none of its ids has the dev-set form); the only dev-set rows, results/teacher_attempts.csv "
              "set 'dev91' (91 rows), are a Qwen3.6-27B teacher run; the lab's own Gemma 4 runs on dev tasks (dev91_*, dev45_*, "
              "eval30_*) are only named in results/adapters.csv",
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: 'Apache "
                                                  "License 2.0 (see `LICENSE`).'"},
    "models": [],
    "harness": None,
    "summary": "Noise study of a Gemma 4 31B agent (and E4B variants): byte-identical Kaggle submissions, 38 complete local "
               "runs of two agent designs on 216 public non-competition tasks (SWE-rebench V1/V2, OOD-45), twenty "
               "interventions, failure anatomy of 995 failed attempts; other models run through the same harness.",
    "evidence": {
        "model": "README.md: 'The model is `google/gemma-4-31b-it-qat-w4a16-ct` served by vLLM 0.19.1'; results/runs.csv "
                 "variants e4b_bundle, e4b_testloop, e4b_distill_v1 are Gemma 4 E4B",
        "harness": "README.md: 'The harness driver `harness/run_eval.py` uses the competition's own `swegemma`, `adk-submission` "
                   "and `adk-eval-core` packages'; 'each task runs in its own upstream Docker image'",
        "systems": "results/task_attempts.csv columns `design` (bundle, safe, variant) and `variant`",
        "replicates": "README.md: '`run_index`: 1 for the first run of a configuration on a set, 2 for `_rep`, 3 for `_rep2`, "
                      "and so on.' (repeats exist, but on non-dev tasks)",
        "task_sets": "README.md task-set table: V2-74 and recent-33 and fresh-15 from SWE-rebench V2, rebench-dev (53) from "
                     "SWE-rebench V1, OOD-45 'a leakage-audited post-release split in the competition's task format'",
    },
    "decisions": [
        "SWE-rebench ids (owner__repo-number) and OOD-45 ids (e.g. apprise_1747) are not rewritten into dev-set ids; none of "
        "the 216 tasks is a competition dev task",
        "the 91 'dev91' rows are dev-set tasks (51 fastapi, 37 rich, 2 requests, 1 httpx) but were attempted by Qwen3.6-27B, a "
        "non-Gemma teacher model, so they are a control and not yielded",
    ],
    "excluded": [
        "results/task_attempts.csv, runs.csv, task_solve_rates.csv, tasks.csv: Gemma 4 31B / E4B attempts on SWE-rebench V1/V2 "
        "and OOD-45 tasks (sets V2-74, recent-33, rebench-dev, OOD-45, fresh-15) - outside the dev set",
        "results/teacher_attempts.csv: non-Gemma-4 models. Set dev91, run dev91_teacher_qwen36: Qwen3.6-27B resolves 38/91 dev "
        "tasks; the other rows are Mistral Large 4, Devstral Small 2 24B and Qwen3.6-27B on SWE-rebench / OOD-45 tasks",
        "results/kaggle_submissions.csv: 15 Kaggle submissions with public (hidden-test) scores, incl. byte-identical "
        "replicates: v4c_pinned_cap28_noedit_safe 0.18, 0.13, 0.13 and public_bav2_noedit_t7 0.17, 0.13",
        "results/interventions.csv: the paper's table of twenty interventions (aggregates over runs on non-dev sets and Kaggle)",
        "results/adapters.csv: the seven trained adapters; names Gemma 4 runs on dev tasks (step1C_v3_dev91_t4, dev91_v3_t5, "
        "dev91_v3_t5_rep, dev45_v3_rerun, dev91_v4b, eval30_lora_v1, eval30_lora_v1_rep ...) whose per-task results are not "
        "released (README.md: 'Not included: the competition's own data (rule 2.4b forbids redistribution)')",
    ],
    "dropped": {},
    "flags": ["no_dev_set_gemma_attempts"],
    "reported_numbers": [],
}


def attempts(src):
    return
    yield
