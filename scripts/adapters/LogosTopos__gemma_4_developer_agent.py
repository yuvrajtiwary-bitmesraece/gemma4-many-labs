"""Adapter for LogosTopos/gemma-4-developer-agent (harness analysis, localisation experiments, hand-off plan).

Not eligible: the snapshot contains no execution of a Gemma 4 agent with a
per-task resolved / not-resolved outcome.  The repository says so itself.
"""

REPO = "LogosTopos/gemma-4-developer-agent"

CARD = {
    "eligible": False,
    "reason": ("no per-task agent outcomes: the lab ran no real Gemma 4 solving experiment on the development set; "
               "the only Gemma 4 results are two hidden-test leaderboard scores (v2 0.05, v4 0.06), everything else is "
               "file-localisation metrics, a scripted-model replay on one task, reference-patch smoke tests and "
               "task/asset audits"),
    "licence": {
        "spdx": "MIT",
        "evidence": ("LICENSE: 'MIT License Copyright (c) 2026 LogosTopos Permission is hereby granted, free of charge, "
                     "to any person obtaining a copy of this software ...' followed by a NOTE 'This license covers only "
                     "the original content of this repository (analysis documents, scripts, and baseline agent "
                     "configuration). Competition data and the official evaluation framework ... are NOT distributed "
                     "here.' The appended NOTE is why GitHub reports NOASSERTION; the licence text itself is the "
                     "standard MIT text. README.md has no licence line."),
    },
    "models": [],
    "harness": None,
    "summary": ("Reverse-engineering notes on the competition harness, a data audit of the 129 public tasks, a "
                "deterministic file-localisation study (v3/v4 retrieval, Hit@k / recall on dev/validation/external "
                "splits) and four Kaggle submissions (v1 no score, v2 0.05, v4 0.06, an exact re-submission of a "
                "public 0.12 baseline still PENDING). No local Gemma inference was run."),
    "evidence": {
        "model": ("docs/RESEARCH_STATUS.md: target is the fixed 'Gemma 4 31B W4A16-CT'; templates/run_manifest.json "
                  "'model_id': 'gemma-4-31b-it-qat-w4a16-ct' (empty template, no run recorded)"),
        "harness": ("README.md: '没有本地真实 Gemma 解题实验；现有 ADK 工具回放使用脚本化模型，不能作为解题率证据。' "
                    "(no local real-Gemma solving experiment; the ADK tool replay uses a scripted model and is not "
                    "solve-rate evidence)"),
        "systems": ("README.md table: v1 submission/ (56521006, no score), v2 (56575713, 0.05), v3 (local candidate, "
                    "not submitted), v4 (56606835, 0.06), D0 Roman replica (56641493, PENDING) - leaderboard only"),
        "replicates": ("none. docs/RESEARCH_STATUS.md: '我们没有本地真实 Gemma 轨迹，没有同配置重复运行' "
                       "(we have no local real-Gemma trajectories and no repeated runs of one configuration)"),
        "task_sets": ("analysis/2026-09-26-data/audit/summary.json: tasks 129 (fastapi 67, rich 48, requests 13, "
                      "httpx 1); analysis/2026-09-27-localization/v4/splits.json: train 82 / validation 33 / "
                      "external_diagnostic 14, used for localisation only"),
    },
    "decisions": [
        "licence recorded as MIT from the LICENSE text although GitHub's detector says NOASSERTION (extra scope note appended to the standard text)",
        "the source would pass the licence rule; it is ineligible only because it holds no eligible attempts",
        "public leaderboard scores (0.05, 0.06) are hidden-test aggregates without task ids and are not converted into attempts",
        "experiment_matrix.json has 'status': 'prepared_not_model_evaluated' and templates/*.json are empty schemas: nothing to yield",
    ],
    "excluded": [
        "analysis/2026-09-28-public-baselines/submissions.json, analysis/2026-09-27-localization/runtime/submission_*_manifest.json, submission_status_evidence.json: hidden-test leaderboard scores only (v2 publicScore 0.05, v4 publicScore 0.06, v1 no score 'requested more CPU, GPU or TPU resources than are available', D0 Roman replica PENDING); no per-task outcomes",
        "analysis/2026-09-28-public-baselines/leaderboard.csv: export of the public Kaggle leaderboard (736 teams, other people's hidden-test scores)",
        "analysis/2026-09-27-localization/v3/localization_dev.jsonl, localization_heldout.jsonl and *_summary.json: file-localisation metrics (hit_at_k / recall_at_k per retrieval variant), not resolved outcomes",
        "analysis/2026-09-27-localization/v4/round{1,2,3}_train.jsonl, final_train.jsonl, final_validation.jsonl, final_external_diagnostic.jsonl, pairwise.json, *_summary.json: retrieval predictions (targets vs paths) for 19 retriever configurations, not agent runs",
        "analysis/2026-09-27-localization/runtime/submission_v{2,3,4}_runtime_check.json: non-agent control, 'kind': 'scripted model replay, actual ADK Runner + LiteLlm + swegemma tools + SubprocessManager; not Gemma inference' on the single task fastapi_14786 (before_failed_after_passed true with a pre-scripted edit)",
        "analysis/2026-09-26-data/audit/smoke_results.json: reference (gold) patch controls on 3 tasks (requests_7315, rich_3063, fastapi_14786): fail_to_pass_observed true for all 3, partial test replays, 'not the full public test_patch' for fastapi_14786",
        "analysis/2026-09-26-data/audit/*.json (asset_inventory, graph_aliases, graph_checks, pilot_snapshot_checks, git_worktree_probe, provenance, summary), analysis/task_index.csv, analysis/2026-09-26-data/splits.json, scripts/research_archive/v3/corpus_manifest.json: task metadata and asset audits",
        "analysis/2026-09-26-data/experiment_matrix.json: plan, 'status': 'prepared_not_model_evaluated'",
        "templates/task_result.json, templates/run_manifest.json: empty schema templates",
        "scripts/e2e_local_test.py + scripts/mock_llm_server.py: mock-LLM end-to-end test, no result file; docs/research/history/SUBMISSION_DIAGNOSIS.md reports it as one task, zero resolved and 'not evidence of real Gemma inference'",
    ],
    "dropped": {},
    "flags": ["no_per_task_outcomes", "leaderboard_only"],
    "reported_numbers": [
        {"what": "own submissions, public leaderboard score (hidden test, not per task)", "value": "v2 0.05, v4 0.06",
         "where": "README.md; analysis/2026-09-28-public-baselines/submissions.json", "adapter": "not yielded (leaderboard)",
         "match": True},
        {"what": "local real-Gemma solving runs", "value": "0",
         "where": "README.md, docs/RESEARCH_STATUS.md, docs/EXPERIMENT_HANDOFF.md", "adapter": "0 attempts", "match": True},
    ],
}


def attempts(src):
    return
    yield
