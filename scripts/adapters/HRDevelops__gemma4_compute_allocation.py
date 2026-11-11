"""Adapter for HRDevelops/gemma4-compute-allocation ("More Seeds, More Steps, or a Better Pick?").

Eligible data: 25 seeded runs of one Gemma 4 31B agent (the public "Budget-Fit" agent) on the 118 development-set
tasks that passed the lab's own reference-patch / empty-patch check, in four budget regimes:

    results/baseline/seed{0..4}_task_results.jsonl           short  (8 min, 28 calls, 80 turns)
    results/baseline_holdout/seed{5..9}_task_results.jsonl   short, fresh seeds of the same configuration
    results/baseline_medium/seed{0..4}_task_results.jsonl    medium (16 min, 56 calls, 160 turns)
    results/baseline_long/seed{0..4}_task_results.jsonl      long   (40 min, 140 calls, 400 turns)
    results/long_thorough/seed{0..4}_task_results.jsonl      long + a prompt that states the budget

25 runs x 118 tasks = 2,950 attempts, the number the README states.  The per-run harness files are the rawest
representation; features.jsonl, short10_features.jsonl and matrix.csv repeat the same outcomes (checked: no
disagreement in any of the 2,950 + 1,180 + 590 cells) and are not read.
"""
import re

REPO = "HRDevelops/gemma4-compute-allocation"

CARD = {
    "eligible": True,
    "licence": {"spdx": "MIT",
                "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Husnain Rasool'; README.md '## License': 'Code and "
                            "results: MIT (see LICENSE). Competition tasks and data are not redistributed here and remain "
                            "under the competition's terms.'"},
    "models": ["31B"],
    "harness": "official",
    "summary": ("Budget-versus-samples study: the public 'Budget-Fit' Gemma 4 31B agent run with a fixed sampling seed per "
                "run on the 118 gold-checked development tasks under four budget regimes (short x10 seeds, medium x5, "
                "long x5, long with a budget-aware prompt x5) on a Slurm cluster (2x H100, Apptainer sandbox) through the "
                "swegemma Evaluator; 2,950 attempts."),
    "evidence": {
        "model": ("code/agents/*/agent.yaml: 'model: gemma-4-31b-it-qat-w4a16-ct'; paper/draft.md s2: 'The model is Gemma 4 "
                  "31B (QAT, w4a16), served by vLLM 0.19.1 with the competition's flags'"),
        "harness": ("README.md: '`seedN_task_results.jsonl` (official grader output per task)'; code/hpc/run_agent.py "
                    "docstring: 'Run a submission (agent) on tasks with the official swegemma Evaluator. Mirrors the starter "
                    "notebook ... Differences: Apptainer sandbox (apptainer_sandbox.py) and per-task time-consistent wheels "
                    "(task_env.py), as validated in Gate 1.'; paper/draft.md s2: 'We use two H100s per server (tensor "
                    "parallel 2) where Kaggle uses four L4s, and run 10 tasks concurrently.'"),
        "systems": ("README.md: '`code/agents/` The four agent configurations (budget and prompt are the only differences)' "
                    "and the run table: baseline = 'short: 8 min, 28 tool calls, 80 turns (competition default)', "
                    "baseline_medium = 'medium: 16 min, 56 calls, 160 turns', baseline_long = 'long: 40 min, 140 calls, 400 "
                    "turns', long_thorough = 'long, with a prompt that states the budget'; the code/agents/*/eval_config.yaml "
                    "files carry exactly these limits"),
        "replicates": ("README.md run table, column 'Seeds': baseline '0-4', baseline_holdout 'short, fresh seeds for the "
                       "pre-registered holdout' '5-9', baseline_medium / baseline_long / long_thorough '0-4'; "
                       "paper/draft.md Table 1: 'short (competition default) | 8 min | 28 | 80 | 10' seeds and 'Sampling "
                       "uses temperature 0.2, top-p 0.95, thinking budget 4,096, and a fixed seed per run'; "
                       "results/noise.txt: 'short: 10 seeds, mean 47.9'; code/hpc/run_agent.py: 'Per-seed copy of the "
                       "submission with the sampling seed set' (cfg_s[\"seed\"] = a.seed); file names seedN_task_results.jsonl"),
        "task_sets": ("paper/draft.md s2: 'we applied each task's reference (\"gold\") patch and an empty patch: gold "
                      "patches resolve 120/129; empty patches resolve 2/129. The 118 tasks where gold passes and empty fails "
                      "are our benchmark'; code/hpc/run_agent.py: '--tasks-csv ... gate1_soundness.csv; only status==sound "
                      "tasks run'. All 25 run files hold exactly these 118 ids (60 fastapi, 45 rich, 12 requests, 1 httpx)."),
    },
    "decisions": [
        "system = agent configuration directory under code/agents/ (baseline, baseline_medium, baseline_long, long_thorough)",
        "results/baseline (seeds 0-4) and results/baseline_holdout (seeds 5-9) are one system 'baseline': the lab calls the "
        "holdout 'short, fresh seeds', pools the ten seeds as the short regime ('short: 10 seeds') and has no separate agent "
        "configuration for it (four configurations, five result directories)",
        "run = '<result directory>/seedN'; the seed index is the sampling seed, so every run is a distinct execution",
        "long_thorough differs from baseline_long by one sentence of the prompt, hence a separate system (the lab's "
        "'long + budget-aware prompt' regime)",
        "harness 'official': the swegemma Evaluator scored every attempt; sandbox and package environment are the lab's "
        "(rootless Apptainer, per-task wheels chosen by release date, the repository's own wheel left out) - flagged "
        "lab_provisioned_sandbox",
        "the 118-task set was fixed by non-agent controls (reference patch passes and empty patch fails) before any agent "
        "run, so it is flagged tasks_restricted_to_gold_checked and NOT outcome_selected_task_subset",
        "all rows are kept: the lab counts every one of the 118 tasks per seed in its scores and marks none as not run; "
        "rows with test_exit_code 124/137/-1 (verification timeout or kill, no patch) are failures in the lab's own totals",
        "err = non-empty harness 'error' string (session timeout, tool-call budget or turns budget)",
        "the lab's 'invalid run' notion (patch changes no source file or a limit was hit) is an analysis label for patch "
        "selection, not an infrastructure failure; such rows stay y = 0",
    ],
    "excluded": [
        "results/*/features.jsonl (5 files, 2,950 rows) and results/short10_features.jsonl (1,180 rows): derived "
        "per-task-and-seed feature tables that repeat the outcomes of the seedN_task_results.jsonl files",
        "results/baseline/matrix.csv: 118 x 5 resolved matrix of baseline seeds 0-4, repeats the run files",
        "results/baseline/judge_scores.jsonl: Gemma 4 judging candidate patches (0-10 scores), not a test outcome",
        "results/gate1_soundness.csv: reference-patch and empty-patch controls on all 129 tasks (see 'controls'); its "
        "columns community_status / community2_gold are labels copied from two unnamed community audits "
        "(code/hpc/gate1_summary.py), not executions of this lab",
        "results/task_env_manifest.json: package versions chosen per task, no outcomes",
        "results/frontier.csv, frontier.txt, noise.txt, paired_tests.txt, prereg_holdout.txt, usage_by_regime.txt, "
        "baseline/selector_eval.txt, baseline_long/compare.txt, paper/figs/*.csv: aggregates and simulated best-of-k "
        "selection over the same runs",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["tasks_restricted_to_gold_checked", "lab_provisioned_sandbox", "non_kaggle_hardware_2xH100",
              "raised_budget_16min_56calls:baseline_medium", "raised_budget_40min_140calls:baseline_long,long_thorough"],
    "sampling": ("temperature 0.2, top_p 0.95, max_output_tokens 8192, thinking_budget 4096 for all four configurations "
                 "(code/agents/*/configs/sampling.yaml); one fixed sampling seed per run, equal to the seed index N of "
                 "seedN and written into the run's copy of sampling.yaml (code/hpc/run_agent.py); seeds 0-9 for baseline, "
                 "0-4 for the other three; 10 tasks concurrently per run (paper/draft.md s2)"),
    "leaderboard": ("none: the lab states no public-leaderboard score for any of its systems (the only mention is the "
                    "comment 'Budget of the public Budget-Fit agent (most stable public leaderboard config)' in "
                    "code/agents/*/eval_config.yaml)"),
    "controls": [
        {"path": "results/gate1_soundness.csv", "kind": "reference_patch", "column": "gold_resolved",
         "environment": ("official swegemma Evaluator (Docker code path) on the lab's Slurm cluster with a rootless "
                         "Apptainer sandbox and per-task time-consistent wheels, repository's own wheel left out "
                         "(code/hpc/gate1.py, task_env.py)"),
         "tasks": 129, "resolved": 120},
        {"path": "results/gate1_soundness.csv", "kind": "empty_patch", "column": "empty_resolved",
         "environment": ("official swegemma Evaluator (Docker code path) on the lab's Slurm cluster with a rootless "
                         "Apptainer sandbox and per-task time-consistent wheels, repository's own wheel left out "
                         "(code/hpc/gate1.py, task_env.py)"),
         "tasks": 129, "resolved": 2},
    ],
    "reported_numbers": [
        {"what": "agent runs (task x seed) in total", "value": "2,950 on 118 tasks", "where": "README.md, paper/draft.md Table 1",
         "adapter": "2,950 attempts = 25 runs x 118 tasks", "match": True},
        {"what": "short regime, 10 seeds: mean solved / SD across seeds", "value": "47.9 / 2.85", "where": "results/noise.txt",
         "adapter": "50, 50, 48, 45, 49, 49, 52, 47, 47, 42 -> mean 47.9, SD 2.85", "match": True},
        {"what": "short seeds 0-4: single-run mean and range", "value": "48.4, range 45-50",
         "where": "results/baseline_long/compare.txt, results/baseline/selector_eval.txt",
         "adapter": "50, 50, 48, 45, 49 -> 48.4", "match": True},
        {"what": "holdout short seeds 5-9: single-run mean", "value": "47.4", "where": "results/prereg_holdout.txt",
         "adapter": "49, 52, 47, 47, 42 -> 47.4", "match": True},
        {"what": "medium, 5 seeds: mean / SD", "value": "52.8 / 4.09", "where": "results/noise.txt",
         "adapter": "56, 46, 52, 55, 55 -> 52.8 / 4.09", "match": True},
        {"what": "long, 5 seeds: mean / SD", "value": "52.8 / 3.35", "where": "results/noise.txt",
         "adapter": "50, 57, 53, 55, 49 -> 52.8 / 3.35", "match": True},
        {"what": "long seeds 0-2: single-run mean and range", "value": "53.3, range 50-57",
         "where": "results/baseline_long/compare.txt", "adapter": "50, 57, 53 -> 53.3", "match": True},
        {"what": "long + budget-aware prompt, 5 seeds: mean / SD", "value": "54.8 / 1.10", "where": "results/noise.txt",
         "adapter": "56, 54, 54, 56, 54 -> 54.8 / 1.10", "match": True},
        {"what": "tasks that flip between seeds (short10 / medium / long / thorough)", "value": "31 / 34 / 37 / 31",
         "where": "results/noise.txt; paper/draft.md Fig. 1", "adapter": "31 / 34 / 37 / 31", "match": True},
        {"what": "tasks never solved (10 short seeds / 5 long seeds)", "value": "57 / 48", "where": "paper/draft.md s3",
         "adapter": "57 / 48", "match": True},
        {"what": "runs ending on a limit (short / medium / long / thorough)", "value": "106 / 31 / 14 / 14",
         "where": "results/usage_by_regime.txt ('limits' dicts: time+turns+calls)",
         "adapter": "err=True rows: 106 / 31 / 14 / 14", "match": True},
        {"what": "mean solved per repository, short regime", "value": "fastapi 22.0/60, httpx 0.8/1, requests 10.7/12, rich 14.4/45",
         "where": "results/usage_by_regime.txt", "adapter": "22.0, 0.8, 10.7, 14.4", "match": True},
        {"what": "controls: reference patch / empty patch resolved, sound tasks", "value": "120/129, 2/129, 118 sound",
         "where": "README.md, paper/draft.md s2", "adapter": "not yielded; results/gate1_soundness.csv gives 120, 2, 118",
         "match": True},
    ],
}

RUN = re.compile(r"^results/(baseline|baseline_holdout|baseline_medium|baseline_long|long_thorough)/"
                 r"(seed\d+)_task_results\.jsonl$")
SYSTEM = {"baseline": "baseline", "baseline_holdout": "baseline", "baseline_medium": "baseline_medium",
          "baseline_long": "baseline_long", "long_thorough": "long_thorough"}


def attempts(src):
    for path, rows in src.jsonl(RUN.pattern):
        run_dir, seed = RUN.match(path).groups()
        for row in rows:
            resolved = bool(row["resolved"])
            error = row.get("error")
            yield {
                "file": path,
                "system": SYSTEM[run_dir],
                "run": f"{run_dir}/{seed}",
                "task": row["instance_id"],
                "y": int(resolved),
                "model": "31B",
                "harness": "official",
                "err": bool(error),
                "status": error or ("resolved" if resolved else f"failed (test_exit_code {row.get('test_exit_code')})"),
            }
