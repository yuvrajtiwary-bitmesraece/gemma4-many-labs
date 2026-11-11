"""Adapter for bsw0610/gemma4-swe-harness-audit ("What Actually Limits a Small-Model SWE Agent?").

Besides two localisation analyses (C1, C2: no outcomes) the repository publishes the per-run table of its
context-overflow experiment C3:

    results/c3_run_rows.csv             one row per (task, arm, rep): resolved + overflow / timeout / tool-budget flags
    results/c3_unjudgeable_tasks.csv    the 40 tasks whose verification is blocked by missing test dependencies

Two arms of the reference agent without LoRA adapter and sub-agent (base: max_output_tokens 16,384; out8k: 8,192)
on 120 development tasks, repetition 1 on all 120 and repetition 2 on the first 40: 320 rows.  The lab asks to leave
the 40 unjudgeable tasks out of resolved rates, so their 112 rows (all unresolved) are dropped here; 208 attempts
remain (80 + 80 + 24 + 24).
"""
import srcio

REPO = "bsw0610/gemma4-swe-harness-audit"

# The lab marks 40 of its 120 tasks as unjudgeable in its sandbox (pytest cannot pass whatever the patch, because the
# subprocess sandbox skips the test dependencies) and reports resolved rates on the other 80.  True follows the lab and
# the README rule on attempts that are invalid for infrastructure reasons; False yields all 320 rows (29/120, 34/120,
# 11/40, 11/40).
DROP_UNJUDGEABLE = True

CARD = {
    "eligible": True,
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License': 'Apache 2.0, see "
                            "`LICENSE`.'; README.md '## Data': 'It does include aggregate results and per-run metrics of our "
                            "own runs (`results/c3_run_rows.csv`, `results/c3_unjudgeable_tasks.csv`).'"},
    "models": ["31B"],
    "harness": "official",
    "summary": ("Harness audit whose experiment C3 runs the competition's sample agent (Gemma 4 31B, no LoRA adapter, no "
                "sub-agent) in two arms that differ only in max_output_tokens (16,384 against 8,192) on 120 randomly "
                "ordered development tasks, with a second repetition of both arms on the first 40 tasks; swegemma 0.2.7 "
                "with its subprocess sandbox on one Colab A100, 100 tool calls and 30 minutes per task."),
    "evidence": {
        "model": "colab/c3_run.py: MODEL_NAME = \"gemma-4-31b-it-qat-w4a16-ct\"; VllmConfig(..., enable_lora=False, tensor_parallel_size=1)",
        "harness": ("reports/c3_results.md: 'resolved = official swegemma verification in the Colab environment (harness env "
                    "+ pytest 8.3.4 from the competition wheels)'; colab/c3_run.py builds swegemma's Evaluator(EvalConfig(..., "
                    "sandbox=\"subprocess\", ...)); README.md: 'They ran on one A100 in Colab, not on the scoring hardware, so "
                    "official scores can differ.'"),
        "systems": ("reports/c3_results.md: 'base: sample_submission prompt/tools/sampling without the LoRA adapter and the "
                    "code_analyzer sub-agent (max_output_tokens 16,384). out8k: identical except max_output_tokens 8,192. "
                    "Budgets: 100 tool calls, 30 min, 300 s per command, 500 turns'"),
        "replicates": ("results/c3_run_rows.csv column 'rep' (1: 240 rows, 2: 80 rows); reports/c3_results.md '## Repetition "
                       "2 (calibration, 2026-11-04)': 'First 40 tasks of the main run, both arms, same harness (swegemma "
                       "0.2.7), same sampling'; README.md: '320 runs of the reference agent ... (120 tasks in two arms, 40 "
                       "of them repeated)'; analysis/c3_report.py: 'With a second repetition present, also prints "
                       "within-arm rep1-vs-rep2 discordance (run-to-run variation)'"),
        "task_sets": ("reports/c3_results.md: '120 tasks (positions 6-125 of `colab/c3_tasks.txt`, a seed-0 random order of "
                      "the 125 tasks with an existing gold code file; positions 1-5 were the pilot and are excluded)'; "
                      "'missing test dependencies blocked verification of 40 of the 120 tasks in this sandbox ... All 112 "
                      "runs on these 40 tasks (rep 1 and rep 2) are unresolved'; README.md: 'Leave out the 40 tasks in "
                      "`results/c3_unjudgeable_tasks.csv` when you compare resolved rates.'"),
    },
    "decisions": [
        "system = arm (base, out8k); run = '<arm>/rep<k>'; rep 2 is the lab's repetition of the same arm on the first 40 "
        "tasks of the fixed random order (not chosen from outcomes)",
        "the 40 tasks of results/c3_unjudgeable_tasks.csv are dropped (112 rows, all unresolved): the lab found after "
        "the runs that pytest cannot pass on them in its sandbox whatever the patch and reports its resolved rates on "
        "the other 80; the unjudgeable list comes from the runs' test output plus a static check, not from a "
        "reference-patch run. Set DROP_UNJUDGEABLE = False to yield all 320 rows",
        "between rep 1 (2026-10-07) and rep 2 (2026-11-04) the environment was rebuilt ('Within-arm change mixes "
        "sampling noise with any environment drift between runs'); the lab still treats rep 2 as a repetition, so both "
        "keep one system label",
        "fastapi_14851 is kept as recorded (resolved in all four runs with an empty patch; the lab notes its target "
        "tests pass on the unmodified snapshot)",
        "err = any of the row's overflow / timeout / tool_budget / other_error flags; 5 kept rows are resolved although "
        "the tool-budget flag is set (the four fastapi_14851 rows with an empty patch, and fastapi_14873 in out8k/rep2 "
        "whose working-tree patch passed)",
        "harness 'official': swegemma 0.2.7 verification with the subprocess sandbox, run by the lab in Colab on one "
        "A100 with a 30-minute, 100-call budget",
    ],
    "excluded": [
        "C2 pilot and pre-flight agent runs (5 pilot tasks x 2 arms, 3 pre-flight runs of a skill arm; "
        "reports/c2_results.md 'Agent runs'): only counts are reported ('0 of 8 skill-arm runs' called the skill), no "
        "per-task outcomes",
        "reports/c1_results.md, c1_manifest.json, harness_0_2_11_recheck.md, tool_mechanics.md, notebook.ipynb, "
        "analysis/, kaggle/: localisation reach of the code-graph tools and of a BM25 skill (hit@k against gold "
        "symbols), no agent outcomes",
        "results/c3_unjudgeable_tasks.csv: task list used to drop rows (see 'controls')",
        "trace-derived columns of results/c3_run_rows.csv (first_open_step, max_prompt, lost_edit, ...): not outcomes",
    ],
    "dropped": {"task marked unjudgeable in the lab's sandbox (missing test dependencies; lab leaves it out of resolved rates)": 112},
    "flags": ["sandbox_unsolvable_tasks_dropped", "non_kaggle_hardware_1xA100", "raised_budget_30min_100calls"],
    "sampling": ("temperature 0.2, top_p 0.95, thinking_budget 4096, max_output_tokens 16384 (base) or 8192 (out8k), "
                 "written by colab/c3_run.py into configs/sampling.yaml; no seed is set (README.md: 'The GPU runs are not "
                 "bit-reproducible'; reports/c3_results.md: 'run-to-run variation (temperature 0.2)'); repetition 2 uses "
                 "'same sampling'"),
    "leaderboard": "none (no leaderboard score is stated; README.md: 'official scores can differ')",
    "controls": [
        {"path": "results/c3_unjudgeable_tasks.csv", "kind": "flag_missing_dependency",
         "environment": ("swegemma 0.2.7 subprocess sandbox in Colab; static check of snapshots and test patches "
                         "(kaggle/unjudgeable/unjudgeable.py) confirmed by the runs' test output"),
         "tasks": 120, "flagged": 40,
         "note": ("31 fastapi tasks need inline_snapshot or dirty_equals, 1 the same hidden by an earlier error, 8 requests "
                  "tasks have a broken httpbin fixture; no reference-patch or empty-patch run was made (fastapi_14851 "
                  "passes with an empty patch in all four agent runs)")},
    ],
    "reported_numbers": [
        {"what": "rep 1 resolved, base / out8k, all 120 tasks and the 80 judgeable ones", "value": "29 / 34 (of 120 and of 80)",
         "where": "reports/c3_results.md 'Paired outcomes' and 'Judgeable tasks'; README.md C3",
         "adapter": "base/rep1 29/80, out8k/rep1 34/80 (29/120 and 34/120 with DROP_UNJUDGEABLE = False)", "match": True},
        {"what": "rep 1 context overflow / 30-min timeout / tool budget, base and out8k", "value": "26, 1, 9 and 0, 4, 12 of 120",
         "where": "reports/c3_results.md 'Paired outcomes'", "adapter": "same counts in the 240 rep-1 rows before dropping",
         "match": True},
        {"what": "resolved on the 40 repeated tasks", "value": "base rep 1 11, rep 2 11; out8k rep 1 12, rep 2 11",
         "where": "reports/c3_results.md 'Repetition 2'",
         "adapter": "base/rep2 11/24, out8k/rep2 11/24 after dropping; 11, 11, 12, 11 of 40 before", "match": True},
        {"what": "same arm, rep 1 against rep 2: tasks whose resolved flag changed", "value": "base 4/40 (2, 2); out8k 3/40 (1, 2)",
         "where": "reports/c3_results.md 'Repetition 2'", "adapter": "4 (2, 2) and 3 (1, 2)", "match": True},
        {"what": "runs on the 40 unjudgeable tasks", "value": "112, all unresolved", "where": "reports/c3_results.md 'Judgeable tasks'",
         "adapter": "112 rows dropped, 0 resolved", "match": True},
    ],
}


def attempts(src):
    unjudgeable = set()
    if DROP_UNJUDGEABLE:
        for _, rows in src.csv(r"^results/c3_unjudgeable_tasks\.csv$"):
            unjudgeable = {row["task"] for row in rows}
    for path, rows in src.csv(r"^results/c3_run_rows\.csv$"):
        for row in rows:
            if row["task"] in unjudgeable:
                continue  # counted in CARD["dropped"]
            resolved = srcio.truthy(row["resolved"])
            errors = [name for name in ("overflow", "timeout", "tool_budget", "other_error") if srcio.truthy(row[name])]
            notes = errors + (["empty patch"] if srcio.truthy(row["empty_patch"]) else [])
            status = ("resolved" if resolved else "failed") + (" (" + ", ".join(notes) + ")" if notes else "")
            yield {
                "file": path,
                "system": row["arm"],
                "run": f"{row['arm']}/rep{row['rep']}",
                "task": row["task"],
                "y": int(resolved),
                "model": "31B",
                "harness": "official",
                "err": bool(errors),
                "status": status,
            }
