"""Adapter for emamira02/gemma4-swe-agent ("Anchored or Not: Localization and Re-ranking in an Offline Gemma 4 Agent").

Eligible data: results/agent_runs/<tag>.json, one file per dev-panel run of one frozen agent bundle
(per_task: resolved / gold / timeout).  Everything else in the repository is a localisation study,
derived tables of the same runs, sandbox checks of the editor skill, or leaderboard scores.
"""
import re

REPO = "emamira02/gemma4-swe-agent"

CARD = {
    "eligible": True,
    "reason": "",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; NOTICE: 'Licensed under the Apache "
                            "License, Version 2.0 (see LICENSE). Competition data is not part of this repository'; "
                            "README.md '## License Apache License 2.0 (`LICENSE`, `NOTICE`)'"},
    "models": ["31B"],
    "harness": "official",
    "summary": ("14 dev-panel runs (one per frozen agent bundle: public baseline B1, own candidates S1-S9, S12, S13, "
                "S16 as 'S16ctl', S17) of a single-LlmAgent 31B agent on 24-task panels (24 rich tasks, or 12 fastapi + "
                "12 rich; S13 on 14 selected tasks), scored in a Kaggle 4xL4 kernel with the competition's swegemma "
                "Evaluator (subprocess sandbox); each task also gets a reference-patch check in the same kernel."),
    "evidence": {
        "model": "kernels/agent_eval/build.py: MODEL = \"gemma-4-31b-it-qat-w4a16-ct\"; kernels/agent_eval/kernel-metadata.json "
                 "model_sources 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'; README.md 'Base model: "
                 "`gemma-4-31b-it-qat-w4a16-ct` only'",
        "harness": "kernels/agent_eval/build.py docstring: 'Generate the Kaggle GPU kernel that runs the agent bundle end to end "
                   "with the official harness on a dev panel ... The kernel mirrors the scorer: 4xL4, vLLM with the scorer's "
                   "settings'; code: 'from swegemma.evaluate import Evaluator', EvalConfig(..., sandbox=\"subprocess\", ...), "
                   "row['agent'] = {'resolved': bool(r.resolved), ...}",
        "systems": "candidates/<TAG>/MANIFEST.json zip_sha256 equals results/agent_runs/<TAG>.json config.sha256 for every "
                   "tag (S16ctl equals candidates/S16); RESUME.md queue table gives 'Change vs previous' per tag (prompt, "
                   "tools, budget or sampling differ between all tags); B1 is 'public Budget-Fit by verracodeguacas, "
                   "Apache 2.0, files extracted verbatim from its notebook' (RESUME.md, NOTICE)",
        "replicates": "none: 14 run files, 14 distinct bundle hashes. paper/PAPER.md 4.7: 'Configurations S1–S5 each ran "
                      "once, with the scorer's settings, on the same 24 rich tasks ... (21 or 22 per run: the environment is noisy)'; docs/gap-analysis-2026-10-29.md: "
                      "'Validation | 24-task mixed panel, 1 run, gold-solvable only'; RESUME.md: 'S16ctl (S16's frozen zip "
                      "19922aedd503 under its own tag, so S16's release gate is untouched)' is the only panel run of S16",
        "task_sets": "kernels/agent_eval/build.py: default panel = random.Random(seed 0).sample of 24 Textualize/rich tasks "
                     "(B1, S1-S8); RESUME.md: S9, S12, S16ctl, S17 on the 'mixed panel (build.py --mix Textualize/rich=12 "
                     "fastapi/fastapi=12 --pools results/gold_check/pools.json, seed 0)'; S13 'targeted panel of 14 tasks: "
                     "S12's 11 unresolved gold-solvable mixed-panel tasks + 3 S12-resolved controls'",
    },
    "decisions": [
        "system = frozen bundle tag; run = tag of the panel run; 'S16ctl' is the panel run of bundle S16 (identical sha256 19922aedd503...), so system 'S16', run 'S16ctl'",
        "attempts with gold == false are dropped as infrastructure-invalid: the reference patch itself fails in that run's sandbox, the lab's failure audit labels them GOLD_UNSOLVABLE and reports every rate over gold-solvable runs (23 attempts, all with resolved == false: rich_3296 and rich_3469 in B1 and S1-S8, rich_3782 in B1, S1, S2, S5, S8)",
        "S6 is kept in full although the lab calls its failure 'infrastructure' ('editor not found by run_command in 15 tasks'): the attempts ran and were scored, the defect is in the S6 bundle's own skill path, the lab reports S6 as 5/22 and includes it in its 13-run stability table; flag S6_editor_unreachable",
        "S13 is kept with the condition in the system label ('S13@targeted-panel'): its 14 tasks were selected from S12 outcomes (11 failures + 3 resolved controls)",
        "B1 (third-party public bundle run unmodified by this lab on its own panel) is a Gemma 4 agent run and is kept as system 'B1'",
        "harness = official: the competition's swegemma Evaluator scored the attempts, but in a Kaggle kernel with sandbox='subprocess' and the bundle's own eval_config budgets, not on the scoring server",
        "err = the 'timeout' flag ('timeout' in the harness error string of the agent phase); status also notes an empty final patch",
        "fastapi_14851 is reported resolved with patch_chars 0 and no source edit in S9, S12, S16ctl and S17 (failure_audit: category RESOLVED, wrote_source false): kept as y = 1 because the source reports it resolved, but the task evidently passes unchanged in this sandbox",
    ],
    "excluded": [
        "per-task 'gold' field of results/agent_runs/*.json and results/gold_check/gold_check/summary.json: reference-patch controls (agent phase replaced by the gold patch). In the panels the reference patch resolves 21-22 of the 24 rich tasks and 24/24 of the mixed panel; kernels/gold_check over 83 tasks resolves 34/83 (fastapi 31/67, requests 1/13, httpx 0/1, rich 2/2)",
        "results/failure_audit.json (by_tag per_task categories), results/agent_summary.json, paper/RESULTS_AGENT.md: derived tables of the same 13 panel runs (S16ctl not included there)",
        "results/serialization_audit.json: malformed tool-call audit of the same runs, not outcomes",
        "results/summary.json, results/sensitivity.json, results/label_audit.json, results/anchor_posthoc.json, results/container_policy.json, analysis/localization_results.json: symbol/file localisation metrics (Recall@k, MRR, hop distance) on 114-129 tasks, no agent runs",
        "results/llm_summary.json, paper/RESULTS_LLM.md: Gemma 4 31B used as a re-ranker / hypothesis generator (Hit@k, Recall@k), not a resolved outcome",
        "results/editor_check/**: unit-style checks of the heredoc editor inside the harness sandbox (23-28 cases), no model",
        "runs.csv, submit_queue.json: hidden-test public leaderboard scores per submitted tag (S1 0.15, S3 0.12, S5 0.12, S12 0.17, S14 0.15, S16 0.15, S18 0.17, PUB1 0.18), no per-task outcomes",
        "candidates/*/GATE.json, candidates/*/MANIFEST.json, kernels/agent_eval/panel.json: release-gate checks, bundle hashes and the task list of the last panel",
    ],
    "dropped": {"reference patch fails in the panel sandbox (gold == false, GOLD_UNSOLVABLE)": 23},
    "flags": ["task_subset_panels", "outcome_selected_task_subset:S13", "S6_editor_unreachable",
              "gold_unsolvable_attempts_dropped", "resolved_with_empty_patch:fastapi_14851"],
    "reported_numbers": [
        {"what": "resolved / gold-solvable per panel: B1, S1, S2, S3, S4, S5, S6, S7, S8, S9, S12, S13, S17",
         "value": "6/21, 4/21, 4/21, 4/22, 5/22, 5/21, 5/22, 7/22, 5/21, 12/24, 13/24, 3/14, 14/24",
         "where": "paper/RESULTS_AGENT.md table; results/agent_summary.json runs.*.panel",
         "adapter": "6/21, 4/21, 4/21, 4/22, 5/22, 5/21, 5/22, 7/22, 5/21, 12/24, 13/24, 3/14, 14/24", "match": True},
        {"what": "S16ctl panel", "value": "13/24", "where": "RESUME.md 'S16ctl panel COMPLETE 13/24'",
         "adapter": "13/24", "match": True},
        {"what": "pooled S1-S5 resolved / solvable runs", "value": "22/107", "where": "paper/PAPER.md 4.7 table",
         "adapter": "22/107", "match": True},
        {"what": "pooled S7, S8, S9, S12 resolved / solvable runs", "value": "37/91", "where": "paper/PAPER.md 4.7 table",
         "adapter": "37/91", "match": True},
        {"what": "mixed panels by repository (fastapi, rich): S9, S12, S17", "value": "9/12 + 3/12, 9/12 + 4/12, 10/12 + 4/12",
         "where": "paper/RESULTS_AGENT.md", "adapter": "9/12 + 3/12, 9/12 + 4/12, 10/12 + 4/12", "match": True},
        {"what": "outcome stability over 13 panel runs (without S16ctl), tasks with >= 2 gold-solvable runs",
         "value": "34 tasks: 13 always, 15 never, 6 mixed (fastapi_14419, rich_3063, rich_3471, rich_3506, rich_3772, rich_3777)",
         "where": "paper/RESULTS_AGENT.md; results/agent_summary.json stability",
         "adapter": "34 tasks: 13 always, 15 never, 6 mixed (same six ids)", "match": True},
    ],
}

RUN = re.compile(r"^results/agent_runs/([A-Za-z]+\d+)(ctl)?\.json$")
SYSTEM_LABEL = {"S13": "S13@targeted-panel"}


def attempts(src):
    for path, obj in src.json(RUN.pattern):
        tag, ctl = RUN.match(path).groups()
        run = tag + (ctl or "")
        system = SYSTEM_LABEL.get(tag, tag)
        for task, row in obj["per_task"].items():
            if not row.get("gold"):
                continue  # reference patch fails in this run's sandbox: counted in CARD["dropped"]
            resolved = bool(row.get("resolved"))
            timeout = bool(row.get("timeout"))
            if resolved:
                status = "resolved"
            elif timeout:
                status = "timeout"
            else:
                status = "failed"
            if not row.get("patch_chars"):
                status += ", empty patch"
            yield {
                "file": path,
                "system": system,
                "run": run,
                "task": task,
                "y": int(resolved),
                "model": "31B",
                "harness": "official",
                "err": timeout,
                "status": status,
            }
