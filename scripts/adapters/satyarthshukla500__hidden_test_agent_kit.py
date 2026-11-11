"""Adapter for satyarthshukla500/hidden-test-agent-kit ("No Test to Anchor On").

Eligible data: results/<tag>/buckets.json, one file per run of a Gemma 4 E4B agent configuration in a Kaggle 4xL4
notebook that drives the competition's swegemma Evaluator: 15 runs on the lab's 90-task development split, 2 runs on
its 39 held-out tasks and three 20-task "smoke" runs of retry-loop designs.  The rest of the repository is per-edit
trace decodes of the same runs, task labels and figures.
"""
import re

REPO = "satyarthshukla500/hidden-test-agent-kit"

# results/task_table.csv labels 29 of the 129 tasks (20 of the lab's 90 dev tasks) with a `sandbox_blocker`: the test
# patch needs a package missing from the offline wheel cache or grades through tests/test_requests.py, which errors at
# collection.  The lab keeps these tasks in every denominator it reports (x/90, x/39, x/20), so they are kept here.
# True would drop them (20 per dev run, 9 per held-out run, 2 per smoke run = 324 attempts, all y = 0).
DROP_SANDBOX_BLOCKED = False

CARD = {
    "eligible": True,
    "reason": "",
    "licence": {"spdx": "MIT",
                "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Satyarth Shukla Permission is hereby granted, free of "
                            "charge, to any person obtaining a copy of this software and associated documentation files'; "
                            "README.md last line 'MIT license.'; PAPER.md s10 '(MIT)'"},
    "models": ["E4B"],
    "harness": "official",
    "summary": ("20 runs of Gemma 4 E4B agents (15 configurations) scored by the competition's swegemma Evaluator in a "
                "Kaggle 4xL4 notebook, 4 tasks concurrent: 15 runs on the lab's 90-task dev split (12 configurations; p01 "
                "run twice, p03 three times), p03 and p07 once each on the lab's 39 held-out tasks, and three retry-loop "
                "designs (p08, p08b, p08c) once each on a 20-task smoke subset."),
    "evidence": {
        "model": "EXPERIMENTS.md: 'all runs on the 90-task dev split, gemma-4-e4b, 4 tasks concurrent, unless noted'; "
                 "submission/agent.yaml and variants/*/agent.yaml: 'model: gemma-4-e4b-it-qat-w4a16-ct'; PAPER.md s8: 'one model, one harness'",
        "harness": "scripts/make_kernel.py docstring: 'The kernel follows the official \"Getting Started\" notebook: install the "
                   "wheelhouse, start vLLM, run swegemma's Evaluator on the subprocess sandbox'; code: 'from swegemma.evaluate "
                   "import Evaluator', EvalConfig(..., sandbox='subprocess', ...), classify(): \"if rec.get('resolved') is True "
                   "...: return 'resolved'\"; README.md: '`resolved` all required tests pass'",
        "systems": "README.md: '`p*` runs other than `p08*` use the two-agent design (analyzer writes a brief, coder edits); `s*` runs "
                   "use the official sample prompt; `p08*` are retry loops; `heldout-*` are the same configuration on the 39 held-out "
                   "tasks; `*smoke` is a 20-task check'; EXPERIMENTS.md column 'change' gives one prompt / budget / sampling change per tag; "
                   "PAPER.md abstract: 'Fifteen runs of 12 configurations'",
        "replicates": "README.md: 'A letter suffix (p03b, p03c) is a repeat of the same configuration'; EXPERIMENTS.md: 'p01b | "
                      "p01 again, unchanged | 5/90 | noise run', 'p03b | p03 again, unchanged', 'p03c | p03 again, unchanged'; PAPER.md "
                      "s4 table: 'p03, p03b, p03c | 6,144 / 2,048 tokens; three runs of the same configuration | 7, 7, 8', 'p01, p01b | "
                      "... p01b is p01 unchanged | 7, 5'. (p08b and p08c are NOT repeats of p08: EXPERIMENTS.md gives each its own change, "
                      "variants/p08b and variants/p08c are separate configurations)",
        "task_sets": "README.md: '`make_splits.py` is seed 1234, 30% held out per repo, which gives 90 dev and 39 held-out' "
                     "(data/splits/dev.txt, heldout.txt; together the 129 public tasks). Smoke subset: EXPERIMENTS.md 'p08smoke | ... "
                     "7 min, 40 calls, 20 tasks'; the three smoke files hold the same 20 dev-split ids; no document says how the 20 were chosen",
    },
    "decisions": [
        "system = configuration tag; run = results/<tag> directory. p01 and p01b are two runs of system 'p01'; p03, p03b, p03c are three runs of system 'p03' (the lab calls them unchanged repeats)",
        "heldout-p03 and heldout-p07 are the same configurations on the lab's 39 held-out tasks (seeded random split, disjoint from the 90 dev tasks), so they are further runs of systems 'p03' and 'p07' on a different task set, not replicates on the same tasks",
        "the three 20-task smoke runs are kept as separate systems with the condition in the label ('p08@smoke20', 'p08b@smoke20', 'p08c@smoke20'). The selection rule for the 20 tasks is not documented, but 10 of the 20 are among the 16 dev tasks that any of the 13 earlier full runs had solved (a random 20 of 90 would hold 3.6; hypergeometric P(>=10) = 1.2e-4), so the subset is treated as chosen from earlier outcomes and flagged outcome_selected_task_subset (inferred from the data, not stated by the lab)",
        "sandbox-ungradable tasks are kept: the lab labels 20 of its 90 dev tasks (29 of 129) as not gradable in the offline sandbox (results/task_table.csv sandbox_blocker; PAPER.md s2 'none of the 20 was ever scored as solved, so at most 70 are gradable locally'), but reports every run over 90 / 39 / 20, and the label is a static rule on the test patch, not a per-attempt reference-patch check. DROP_SANDBOX_BLOCKED = True would drop 324 attempts (all y = 0)",
        "y = 1 iff bucket == 'resolved' (the Evaluator's resolved flag); wrong_fix, regression ('target still failing'), no_patch and budget_exhausted are failures",
        "err = bucket is one of budget_exhausted / timeout / apply_failed / agent_error (only budget_exhausted occurs, 4 attempts). The bucket rule tests 'no_patch' before 'timeout' and has no bucket for the turn cap, so err undercounts harness errors: NOTES.md records 'p03b errors: 8 timeouts, 7 turn cap, 68 none' while results/p03b/buckets.json has no error bucket",
        "buckets.json is used for all 20 runs; results/{p01,p01b,p02,p03}/raw.txt is the notebook's compact print of the same four runs with abbreviated ids (F11194 = fastapi_11194) from which those buckets.json files were built, so it is not yielded (checked: identical bucket per task in all four)",
        "harness = official: the competition's swegemma Evaluator scored the attempts, but in the lab's own Kaggle notebook with sandbox='subprocess', the submission's eval_config budgets (5 min / 28 calls / 24 turns; s01 1 min / 10 calls; p08* 6-7 min / 40 calls) and 4 tasks concurrent, not on the scoring server",
    ],
    "excluded": [
        "results/{p01,p01b,p02,p03}/raw.txt: second representation of four runs already yielded from buckets.json",
        "results/*/edits.csv, results/*/tasks.csv: one row per edit_file call / per-task edit counts decoded from traces (edit landed or not), not a resolved outcome",
        "results/task_table.csv, results/task_table.md, results/task_label_check.csv: one row per public task (test-patch labels, sandbox_blocker, and our_runs / our_runs_resolved counts aggregated over the same 17 runs); only sandbox_blocker is read, and only if DROP_SANDBOX_BLOCKED",
        "results/p00-base/brief_review.json: review of 32 failed p00-base briefs (right_file / wrong_file / empty), localisation not outcome",
        "runs without a per-task file (EXPERIMENTS.md): 'smoke' v1 (rich_3676 only, 0/1), 'p01-12b' (p01 on Gemma 4 12B, 0/12, stopped), 'p04' (temperature 0, stopped at 10 min), 'p08b' on the 90 dev tasks (not run)",
        "public leaderboard scores (58 hidden tasks): p03 0.06, p08b 0.06, s01 0.03, s04 0.03 (README.md, PAPER.md, NOTES.md): hidden-test results, no per-task outcomes",
        "controls: none. PAPER.md s8: 'no gold-patch or empty-patch run' (scripts/make_kernel.py has a --skip-agent-patch baseline mode, no result of it is in the repository)",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["outcome_selected_task_subset:p08smoke,p08bsmoke,p08csmoke", "task_selection_inferred_not_documented",
              "sandbox_ungradable_tasks_kept", "lab_defined_dev_heldout_split_90_39"],
    "sampling": ("temperature 0.2, top_p 0.9, top_k 40, fixed seed 7 for every p*/s02-s04 configuration, identical in all repeats "
                 "(submission/configs/sampling.yaml, variants/*/configs/sampling.yaml; PAPER.md s3 'Sampling: temperature 0.2, top_p 0.9, "
                 "top_k 40, seed 7'). Repeats therefore differ only by serving nondeterminism: PAPER.md s5.1 'Temperature 0.2 and a fixed seed "
                 "do not remove the variance: four tasks share one vLLM instance'. s01 uses the sample file (temperature 0.2, top_p 0.95, no "
                 "seed; variants/sample-e4b/configs/sampling.yaml); retry-loop attempts use temperature 0.7 (p08) or 0.5 (p08b, p08c), no seed "
                 "(variants/p08*/configs/attempt_sampling.yaml). A temperature-0 run (p04) was stopped after 10 min"),
    "leaderboard": [
        "p03: public leaderboard 0.06 (58 hidden tasks; lab reads it as 4/58) vs dev 7/90 (repeats 7/90, 8/90) - README.md, PAPER.md s4, NOTES.md 2026-11-08",
        "s01: public leaderboard 0.03 (read as 2/58) vs dev 5/90 - NOTES.md 2026-11-09, PAPER.md s4",
        "s04: public leaderboard 0.03 (read as 2/58) vs dev 8/90 - NOTES.md 2026-11-09 ('dev 8/90 did not carry to the hidden set'), PAPER.md s5.1",
        "p08b: public leaderboard 0.06 (read as 4/58); no 90-task dev run ('not run yet', EXPERIMENTS.md), only the 20-task smoke 5/20 - NOTES.md 2026-11-10",
    ],
    "reported_numbers": [
        {"what": "resolved per 90-task dev run: p00-base, p01, p01b, p02, p03, p03b, p03c, p05, p06, p07, p09, s01, s02, s03, s04",
         "value": "4, 7, 5, 6, 7, 7, 8, 5, 4, 3, 4, 5, 8, 8, 8 of 90", "where": "EXPERIMENTS.md table; PAPER.md s4 table",
         "adapter": "4, 7, 5, 6, 7, 7, 8, 5, 4, 3, 4, 5, 8, 8, 8 of 90", "match": True},
        {"what": "held-out runs: heldout-p03, heldout-p07", "value": "5/39, 3/39", "where": "EXPERIMENTS.md",
         "adapter": "5/39, 3/39", "match": True},
        {"what": "20-task smokes: p08smoke, p08bsmoke, p08csmoke", "value": "3/20, 5/20, 5/20", "where": "EXPERIMENTS.md; NOTES.md 2026-11-09",
         "adapter": "3/20, 5/20, 5/20", "match": True},
        {"what": "p08b / p08c smoke solved ids", "value": "p08b: rich_4077, fastapi_14786, rich_3905, requests_6644, rich_3454; p08c: same with rich_2725 instead of rich_3454",
         "where": "NOTES.md 2026-11-09", "adapter": "same ids", "match": True},
        {"what": "p01 vs p01b", "value": "7 and 5, disagreed on 8 tasks; overlap 2 of 10", "where": "PAPER.md s5.1; EXPERIMENTS.md; NOTES.md",
         "adapter": "7 and 5; 5 solved only by p01, 3 only by p01b (8 discordant); 2 solved by both of 10 solved by either", "match": True},
        {"what": "solve stability of the three p03 repeats", "value": "S = 2/14 (14 solved at least once, 2 every time)",
         "where": "PAPER.md s5.1; EXPERIMENTS.md p03c note", "adapter": "2/14", "match": True},
        {"what": "solve stability over the first 5 runs / over 8 two-agent runs", "value": "1/13 and 1/14", "where": "PAPER.md s5.1",
         "adapter": "1/13 (p00-base, p01, p02, p01b, p03) and 1/14 (plus p03b, p03c, p05)", "match": True},
        {"what": "union of solved dev tasks over all 15 full runs", "value": "16 of 90; none of the 20 sandbox-blocked tasks ever solved",
         "where": "PAPER.md s5.1, s2; figures/README.md", "adapter": "16 of 90; 0 resolved among sandbox-blocked tasks in any of the 20 runs", "match": True},
        {"what": "pooled s02-s04 vs p03 trio (270 tasks a side): no patch, wrong fix, resolved", "value": "33 vs 51, 134 vs 112, 24 vs 22",
         "where": "PAPER.md abstract, s5.3", "adapter": "33 vs 51, 134 vs 112, 24 vs 22", "match": True},
        {"what": "budget_exhausted bucket", "value": "2 tasks in s01 and 2 in the p08c smoke; no other error bucket occurs", "where": "README.md 'Bucket labels'",
         "adapter": "err=True: s01 2, p08csmoke 2, others 0", "match": True},
        {"what": "p00-base buckets", "value": "wf 41, tsf 27, empty 18", "where": "EXPERIMENTS.md", "adapter": "wrong_fix 41, regression 27, no_patch 18", "match": True},
        {"what": "per-task run counts in results/task_table.csv", "value": "our_runs 15 (dev) / 2 (held-out), our_runs_resolved per task",
         "where": "results/task_table.csv", "adapter": "identical for all 129 tasks (17 non-smoke runs)", "match": True},
    ],
}

RUN = re.compile(r"^results/([A-Za-z0-9-]+)/buckets\.json$")
BLOCKERS = r"^results/task_table\.csv$"
# tag -> system; tags not listed are their own system
SYSTEM = {
    "p01b": "p01",
    "p03b": "p03", "p03c": "p03", "heldout-p03": "p03",
    "heldout-p07": "p07",
    "p08smoke": "p08@smoke20", "p08bsmoke": "p08b@smoke20", "p08csmoke": "p08c@smoke20",
}
ERROR_BUCKETS = {"budget_exhausted", "timeout", "apply_failed", "agent_error"}


def attempts(src):
    blocked = set()
    if DROP_SANDBOX_BLOCKED:
        for _, rows in src.csv(BLOCKERS):
            blocked = {r["instance_id"] for r in rows if r["sandbox_blocker"].strip()}
    for path, obj in src.json(RUN.pattern):
        tag = RUN.match(path).group(1)
        for row in obj:
            if row["id"] in blocked:
                continue
            bucket = row["bucket"]
            yield {
                "file": path,
                "system": SYSTEM.get(tag, tag),
                "run": tag,
                "task": row["id"],
                "y": int(bucket == "resolved"),
                "model": "E4B",
                "harness": "official",
                "err": bucket in ERROR_BUCKETS,
                "status": f"{bucket}; calls={row.get('calls')}",
            }
