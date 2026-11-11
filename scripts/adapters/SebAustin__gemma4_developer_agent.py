"""Adapter for SebAustin/gemma4-developer-agent ("Audit the Pipeline Before Tuning the Agent").

Not eligible: the Gemma 4 runs are reported as totals only; the per-task files are controls.
"""

REPO = "SebAustin/gemma4-developer-agent"

CARD = {
    "eligible": False,
    "reason": "no per-task outcomes of Gemma 4 runs are published: the nine Gemma 4 31B development runs are reported in "
              "experiments/EXPERIMENTS.md as totals (0/30, 5/30, 6/30 on dev30; 20/40, 15/40 on sound40; 29/71, 28/71, "
              "24/71, 29/71 on sound71) with only the discordant task ids of some pairs, and their result folders "
              "(results/kaggle/...) are not in the repository; the per-task files that exist are reference-patch and "
              "empty-patch controls and two scripted-model replays of one task",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; docs/paper/writeup.md s6: 'The repository "
                            "github.com/SebAustin/gemma4-developer-agent (Apache-2.0)'"},
    "models": [],
    "harness": None,
    "summary": "Measurement-first entry: declarative Gemma 4 31B bundles (sample, v1-v8) run on Kaggle L4x4 with the "
               "official harness on frozen dev splits (dev30, sound40, sound71), plus reference-patch and empty-patch "
               "controls over all 129 public tasks. Only the controls and run totals are in the repository.",
    "evidence": {
        "model": "README.md: 'A declarative ADK bundle drives `gemma-4-31b-it-qat-w4a16-ct`'; "
                 "submissions/*/agent.yaml 'model: gemma-4-31b-it-qat-w4a16-ct'",
        "harness": "README.md: '`kaggle/eval_kernel/run_eval.py` Template for the Kaggle GPU eval run (vLLM + official "
                   "`Evaluator`)'; experiments/EXPERIMENTS.md: 'Serial dev runs on harness 0.2.11'",
        "systems": "experiments/EXPERIMENTS.md run tables: one row per bundle (sample_baseline, v1_scaffold, v2_fast, "
                   "v2_noskill, v5_budgetfit, v6_issue_recovery, v7_no_thoughts, v7_calls40) with bundle hash, split and "
                   "concurrency",
        "replicates": "none usable: experiments/EXPERIMENTS.md 'no identical-config rerun exists on `sound71`'; "
                      "docs/diagnosis/v1-regression.md describes three dev30 runs of v1_scaffold (0/30, 1/30, 5/30), the "
                      "first two invalid because the vLLM server hung; identical resubmissions exist only on the hidden "
                      "leaderboard (0.13 then 0.10)",
        "task_sets": "experiments/splits.json: full (129), dev30, sound40, sound71, smoke5, holdout_rich; "
                     "experiments/EXPERIMENTS.md: '`sound71`: every task whose reference fix passes on Kaggle and whose "
                     "empty patch fails (41 rich, 29 fastapi, 1 requests; it contains 38 of `sound40`)'",
    },
    "decisions": [
        "nothing is yielded: run totals and lists of discordant tasks do not determine the outcome of every task, and "
        "rows are not invented from totals (flag aggregated_only)",
        "the sample_baseline run on dev30 is reported as 0/30 ('all 30 timed out'), which would determine all 30 rows "
        "(y=0 for every dev30 task of experiments/splits.json); it is still not yielded because no per-task rows are "
        "published and the lab calls these concurrency-4 runs 'not comparable with serial `sound40` runs'",
        "experiments/replay_fastapi_14077/*/task_results.jsonl has the official per-task format but comes from 'a "
        "scripted model' that replays the reference fix, so it is a non-agent control",
    ],
    "excluded": [
        "experiments/oracle_kaggle.json: reference-patch control through the real harness in the Kaggle notebook "
        "(subprocess sandbox): resolves 73/129; the 56 failures are missing test dependency inline_snapshot 27, "
        "dirty_equals 7, rich._unicode_data 1, package version differences 13, network or certificates 8",
        "experiments/oracle_local.json: reference-patch control in local Docker with per-task wheels: resolves 105/129",
        "experiments/empty_local.json: empty-patch control (submit_patch with no edits) in local Docker, harness 0.2.11: "
        "resolves 2/129 (fastapi_14077, fastapi_14851), fails 127, errors 0",
        "experiments/replay_fastapi_14077/a_test_edit and b_with_guard (task_results.jsonl, trace_fastapi_14077.json): "
        "scripted-model replays of the reference fix plus a test edit on harness 0.2.7 (not resolved: 'Failed to apply "
        "test_patch'; resolved after reverting the test edit)",
        "experiments/EXPERIMENTS.md run tables (aggregates only, Gemma 4 31B, official harness on Kaggle): dev30 at "
        "concurrency 4: sample_baseline 0/30, v1_scaffold run 3 5/30, v2_fast 6/30; sound40 serial: v2_noskill 20/40, "
        "v2_fast 15/40 (both 14, noskill only 6, fast only 1, neither 19); sound71 serial: v5_budgetfit 29/71, "
        "v6_issue_recovery 28/71, v7_no_thoughts 24/71, v7_calls40 29/71",
        "experiments/EXPERIMENTS.md leaderboard table: hidden-test public leaderboard scores (0.06-0.17), excluded as "
        "hidden-test results",
        "experiments/splits.json: task lists only",
        "tests/: unit tests of the tooling",
    ],
    "dropped": {},
    "flags": ["aggregated_only"],
    "reported_numbers": [
        {"what": "reference fix resolved, Kaggle notebook (subprocess sandbox)", "value": "73/129 (dev30: 18/30)",
         "where": "docs/paper/writeup.md s3.2; experiments/EXPERIMENTS.md",
         "adapter": "not yielded (control); experiments/oracle_kaggle.json lists 73 resolvable of 129, 18 of dev30",
         "match": True},
        {"what": "reference fix resolved, local Docker with per-task wheels", "value": "105/129 (dev30: 25/30)",
         "where": "docs/paper/writeup.md s3.2; experiments/EXPERIMENTS.md",
         "adapter": "not yielded (control); experiments/oracle_local.json lists 105 resolvable of 129, 25 of dev30",
         "match": True},
        {"what": "public tasks that pass with an empty patch", "value": "2/129 (fastapi_14077, fastapi_14851)",
         "where": "docs/paper/writeup.md s3.3; experiments/EXPERIMENTS.md",
         "adapter": "not yielded (control); experiments/empty_local.json passes_with_empty_patch has these 2 of 129",
         "match": True},
        {"what": "sound71 = reference fix passes on Kaggle and empty patch fails", "value": "71 tasks (41 rich, 29 fastapi, "
                                                                                             "1 requests)",
         "where": "experiments/EXPERIMENTS.md",
         "adapter": "not yielded; experiments/splits.json sound71 equals oracle_kaggle resolvable minus the 2 empty-patch "
                    "passes (41 rich, 29 fastapi, 1 requests)", "match": True},
    ],
}


def attempts(src):
    return
    yield
