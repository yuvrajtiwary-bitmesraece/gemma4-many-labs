"""Adapter for nishanthrjn/gemma4-agent-ablations
("Nothing Moved It: Replicated Ablations of a Gemma 4 Coding Agent").

Per-task verdicts exist for 8 local runs (results/task-resolved.json).  Four more
runs (p3 attempt 2, p4 attempt 1, p5 attempts 1 and 2) are published only as
exhaustive per-task flip lists against a run whose verdicts are known; they are
rebuilt from those lists and every rebuilt run is checked against the solved
count and the cross-comparisons the source reports.  Set
RECONSTRUCT_FROM_FLIP_LISTS = False to yield only the 8 runs of the per-task file.
"""

REPO = "nishanthrjn/gemma4-agent-ablations"

RECONSTRUCT_FROM_FLIP_LISTS = True

RESULTS = "experiments/gemma4_competition/results/"
RESOLVED = RESULTS + "task-resolved.json"
COMPARISON = RESULTS + "dev-16v-comparison.json"

CARD = {
    "eligible": True,
    "licence": {
        "spdx": "Apache-2.0",
        "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; NOTICE: 'Apache-2.0 applies to the authors' "
                    "own code and materials in this export.'; README.md: 'Apache-2.0 covers our own code; see LICENSE "
                    "and NOTICE.'",
    },
    "models": ["31B", "E4B"],
    "harness": "official",
    "summary": (
        "Prompt/budget ablations of a single-LlmAgent Gemma 4 31B agent (candidates p1..p10) and an E4B variant "
        "(s1, s2), each run on 'dev-16v' (16 locally verifiable dev tasks) on Kaggle L4x4 with the swegemma "
        "Evaluator; several candidates were run twice. Per-task verdicts are public for 8 runs, 4 more are "
        "recoverable from published flip lists, the remaining 10 are solved counts only."
    ),
    "evidence": {
        "model": "candidates/p5/agent.yaml: 'model: gemma-4-31b-it-qat-w4a16-ct'; candidates/s1|s2/agent.yaml: "
                 "'model: gemma-4-e4b-it-qat-w4a16-ct'; README.md: 'Gemma 4 31B model ... Use the E4B model instead "
                 "for S1/S2'; p1..p4 configs are not in the export: portable_m2.py 'MODEL = "
                 "\"gemma-4-31b-it-qat-w4a16-ct\"' and portable_p1.py guard 'P1 changed model/tools/sampling' "
                 "(P1 may differ from the 31B baseline only in prompt and eval_config); dev-16v-comparison.json "
                 "/p5_preregistered_decision/candidate: 'P3 with prompts/system.md changed only'",
        "harness": "local_eval/runtime.py: 'from swegemma.evaluate import Evaluator ... result = await "
                   "evaluator.evaluate_task(task, ...)' with sandbox='subprocess', concurrency=1 and the "
                   "candidate's eval_config budgets; portable_m2.py pins swegemma 0.2.11; README.md: 'Use Python 3.12 "
                   "Linux on Kaggle, L4x4, Internet Off'; dev-16v-comparison.json /context_diagnosis: 'Retained "
                   "task-bound official/task_results.jsonl inspected'",
        "systems": "README.md: 'candidates/: exact four-file candidates p5, p6, p7a, p7b, p8, p9a, p9b, p10, s1 and "
                   "s2'; results/task-resolved.json column 'candidate'; local_eval/experimental.py guards 'P6 must "
                   "change only the P5 prompt', 'S1 must change only the P5 model', 'S2 must change only the S1 "
                   "prompt'",
        "replicates": "p3: dev-16v-comparison.json /p3_same_config_replicate: '\"attempt\": 2, "
                      "\"comparison_attempt\": 1, \"configuration_relation\": \"SAME_CONFIG_REPLICATE_MEASURED\"'; "
                      "p5: /p5_preregistered_decision/solved_gate: 'Mean solved over two complete P5 dev-16v runs', "
                      "results/run-summary.csv gives both p5 rows the same zip_sha256 6db4109a..., and "
                      "local_eval/configs/p5-dev16v-attempt1.json / -attempt2.json differ only in the output "
                      "directory; README.md: 'Use a separate output and attempt config for every replicate.'",
        "task_sets": "local_eval/dev-16v.json: name 'dev-16v', 16 tasks (8 fastapi, 8 rich), 'controls_sha256'; "
                     "results/paper-evidence.json claim 'controls': dev30_verifiable 16, confirm20_verifiable 8, "
                     "unsound 19, env_broken 7, total 50; /runs/0/source: 'VERIFIABLE projection of dev-30' (p1)",
        "reconstruction": "dev-16v-comparison.json /eight_run_comparison/basis: 'Historical six-run task table plus "
                          "reported exhaustive flips supplies outcomes.'; /p5_preregistered_decision/flips: 'Report "
                          "each P5 attempt versus each P3 attempt and P5 attempt1 versus attempt2, per task'",
    },
    "decisions": [
        "system = candidate label; run = '<candidate>-attempt<k>' (p1b: '<candidate>-merged')",
        "p1b is one run: attempt 1 stopped with HARNESS_ERROR after 8 tasks and attempt 2 ran the other 8; the lab "
        "publishes and counts the merge as one run ('P1b attempt1+attempt2 merged as one run')",
        "p1 was executed on dev-30; only its 16 verifiable tasks are published ('VERIFIABLE projection of dev-30')",
        "p3-attempt2, p4-attempt1, p5-attempt1 and p5-attempt2 are REBUILT from published per-task flip lists "
        "(outcome_flips, gained_vs_*/lost_vs_* in dev-16v-comparison.json) applied to p3-attempt1 from "
        "task-resolved.json; the adapter raises if a rebuilt run disagrees with the reported solved count, with the "
        "second and third flip list the source gives for the same run, or with solved_in_both_p5_runs; these rows "
        "have status 'rebuilt from flip list' and file = dev-16v-comparison.json",
        "six_run_task_records in dev-16v-comparison.json repeat six runs of task-resolved.json (checked identical) "
        "and are not yielded",
        "err = None everywhere: task-resolved.json has no error field; timeout / context-error task lists exist "
        "only for some runs (p1b, p2, p3, p3g, p1c) in dev-16v-comparison.json and are not complete enough to code "
        "False for the other tasks",
        "p2 / fastapi_15800 is kept as y=0 (resolved=false in task-resolved.json) although the comparison record "
        "calls it 'remaining_unknown_task' pending error classification",
        "harness = official: the swegemma Evaluator scored every attempt; the lab's only change to the verdict is "
        "that unresolved runs with an external-module import error become UNKNOWN (analysis.verification_outcome), "
        "which cannot turn a failure into a success",
    ],
    "excluded": [
        "results/run-summary.csv + dev-16v-comparison.json run records without per-task verdicts (solved counts "
        "of 16 only): p6-attempt1 7, p7a-attempt1 11, p7a-attempt2 7 (same ZIP), p7b-attempt1 10, p8-attempt1 7, "
        "p9a 8 and 7, p9b 7 and 9, p10-attempt2 7 -> aggregated only, no runs invented",
        "dev-16v-comparison.json /six_run_task_records: duplicate of p1, p1b, p1c, p2, p3, p3g in task-resolved.json",
        "results/paper-evidence.json /s1_smoke5 (s1 on smoke-5: 1 verifiable task, 0 resolved): smoke test, "
        "no per-task table",
        "results/submissions.json: public-leaderboard (hidden test) scores - m2 0.05, b2 0.12, p1 0.10, p3 0.15 and "
        "0.12 (identical files), p5 0.13 and 0.15 (identical ZIP), p7a 0.12, s1 (E4B) 0.05",
        "controls (gold / no-op control runs behind the 'verifiable' label): the per-task control file "
        "local_eval/evidence/controls-05706c6a.json is not in the export; only totals are given (50 tasks: 24 "
        "verifiable, 19 unsound, 7 env_broken)",
        "localization study (results/localization-study-attempt1): not in the export",
        "experiments/gemma4_competition/tests, local_eval/test_*.py: synthetic fixtures ('Synthetic test fixtures "
        "are authored test examples, not competition patches')",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["reconstructed_from_flip_lists", "merged_attempts", "partly_aggregated_only"],
    "reported_numbers": [
        {"what": "six-run table p1, p1b, p1c, p2, p3, p3g resolved of 16", "value": "8, 8, 7, 8, 8, 7",
         "where": "results/paper-evidence.json claim six_run_table_and_task_categories; results/run-summary.csv",
         "adapter": "8, 8, 7, 8, 8, 7", "match": True},
        {"what": "s1 and s2 (E4B) resolved of 16", "value": "3, 3",
         "where": "README.md 'S1 local 3/16'; dev-16v-comparison.json /s1_dev16v_attempt1, /s2_dev16v_attempt1",
         "adapter": "3, 3 (same resolved_tasks lists)", "match": True},
        {"what": "p3 same-config replicate", "value": "8/16 then 6/16, 2 outcome flips",
         "where": "dev-16v-comparison.json /p3_same_config_replicate", "adapter": "8/16, 6/16 (rebuilt)", "match": True},
        {"what": "p5 two runs", "value": "9/16 + 9/16, 8 tasks solved in both, 2 flips",
         "where": "README.md 'P5 local 9/16 + 9/16'; dev-16v-comparison.json /p5_adoption_decision",
         "adapter": "9/16, 9/16, 8 in both (rebuilt)", "match": True},
        {"what": "p4 attempt 1", "value": "8/16, 4 paired flips vs p3 attempt 1, first success on rich_4079",
         "where": "dev-16v-comparison.json /p4_attempt1, /p4_adoption_decision",
         "adapter": "8/16; rich_4079 unsolved in the 7 earlier runs (rebuilt)", "match": True},
        {"what": "always / variable / never solved tasks in the six-run table", "value": "5 / 5 / 6 tasks",
         "where": "dev-16v-comparison.json /always_solved, /variable, /never_solved",
         "adapter": "5 / 5 / 6 (same task lists)", "match": True},
    ],
}

E4B = {"s1", "s2"}


def _record(path, candidate, run, task, resolved, status):
    return {
        "file": path,
        "system": candidate,
        "run": run,
        "task": task,
        "y": int(bool(resolved)),
        "model": "E4B" if candidate in E4B else "31B",
        "harness": "official",
        "err": None,
        "status": status,
    }


def _apply(base, gained, lost, what):
    """Outcomes of a run described as 'base run plus gained minus lost'."""
    for task in gained:
        if base.get(task) is not False:
            raise ValueError(f"{what}: gained task {task} is not an unresolved task of the base run")
    for task in lost:
        if base.get(task) is not True:
            raise ValueError(f"{what}: lost task {task} is not a resolved task of the base run")
    out = dict(base)
    out.update({task: True for task in gained})
    out.update({task: False for task in lost})
    return out


def _expect(condition, what):
    if not condition:
        raise ValueError(f"{REPO}: flip-list reconstruction is inconsistent: {what}")


def _rebuild(comparison, p3a1):
    """{run label: (candidate, outcomes)} for the four runs published as flip lists."""
    rep = comparison["p3_same_config_replicate"]
    flips = [f["task_id"] for f in rep["outcome_flips"]]
    _expect(rep["candidate"] == "p3" and rep["attempt"] == 2 and rep["comparison_attempt"] == 1, "p3 replicate id")
    _expect(len(flips) == rep["outcome_flip_count"] and set(flips) <= set(p3a1), "p3 attempt2 flip list")
    p3a2 = {task: (not y if task in flips else y) for task, y in p3a1.items()}
    _expect(sum(p3a2.values()) == rep["resolved"] and len(p3a2) == rep["verifiable"], "p3 attempt2 solved count")

    r = comparison["p4_attempt1"]
    p4a1 = _apply(p3a1, r["gained_vs_p3_attempt1"], r["lost_vs_p3_attempt1"], "p4 attempt1")
    _expect(sum(p4a1.values()) == r["resolved"] and len(p4a1) == r["verifiable"], "p4 attempt1 solved count")
    _expect(len(r["gained_vs_p3_attempt1"]) + len(r["lost_vs_p3_attempt1"])
            == comparison["p4_adoption_decision"]["paired_outcome_flip_count"], "p4 attempt1 flip count")

    r = comparison["p5_attempt1"]
    p5a1 = _apply(p3a1, r["gained_vs_p3_attempt1"], r["lost_vs_p3_attempt1"], "p5 attempt1")
    _expect(sum(p5a1.values()) == r["resolved"] and len(p5a1) == r["verifiable"], "p5 attempt1 solved count")
    _expect(p5a1 == _apply(p3a2, r["gained_vs_p3_attempt2"], r["lost_vs_p3_attempt2"], "p5 attempt1 vs p3 attempt2"),
            "p5 attempt1 vs p3 attempt2")

    r = comparison["p5_attempt2"]
    p5a2 = _apply(p5a1, r["gained_vs_p5_attempt1"], r["lost_vs_p5_attempt1"], "p5 attempt2")
    _expect(sum(p5a2.values()) == r["resolved"] and len(p5a2) == r["verifiable"], "p5 attempt2 solved count")
    _expect(p5a2 == _apply(p3a1, r["gained_vs_p3_attempt1"], r["lost_vs_p3_attempt1"], "p5 attempt2 vs p3 attempt1"),
            "p5 attempt2 vs p3 attempt1")
    _expect(p5a2 == _apply(p3a2, r["gained_vs_p3_attempt2"], r["lost_vs_p3_attempt2"], "p5 attempt2 vs p3 attempt2"),
            "p5 attempt2 vs p3 attempt2")
    decision = comparison["p5_adoption_decision"]
    _expect(sum(a and b for a, b in zip(p5a1.values(), p5a2.values())) == decision["solved_in_both_p5_runs"],
            "tasks solved in both p5 runs")
    _expect(sum(a != b for a, b in zip(p5a1.values(), p5a2.values())) == decision["paired_outcome_flip_count"],
            "p5 paired flip count")

    return {
        "p3-attempt2": ("p3", p3a2),
        "p4-attempt1": ("p4", p4a1),
        "p5-attempt1": ("p5", p5a1),
        "p5-attempt2": ("p5", p5a2),
    }


def attempts(src):
    files = dict(src.json(r"^experiments/gemma4_competition/results/(task-resolved|dev-16v-comparison)\.json$"))
    known = {}
    for row in files[RESOLVED]:
        candidate, attempt = row["candidate"], row["attempt"]
        run = f"{candidate}-merged" if attempt == "merged" else f"{candidate}-attempt{attempt}"
        if not isinstance(row["resolved"], bool):
            raise ValueError(f"{REPO}: non-boolean verdict in {RESOLVED}: {row}")
        known.setdefault(run, {})[row["task_id"]] = row["resolved"]
        yield _record(RESOLVED, candidate, run, row["task_id"], row["resolved"],
                      "resolved" if row["resolved"] else "not resolved")

    comparison = files[COMPARISON]
    for record in comparison["six_run_task_records"]:  # duplicate table: must agree, is not yielded
        run = "p1b-merged" if record["candidate"] == "p1b" else f"{record['candidate']}-attempt1"
        if {t["task_id"]: t["resolved"] for t in record["tasks"]} != known[run]:
            raise ValueError(f"{REPO}: six_run_task_records disagree with {RESOLVED} for {run}")

    if not RECONSTRUCT_FROM_FLIP_LISTS:
        return
    for run, (candidate, outcomes) in _rebuild(comparison, known["p3-attempt1"]).items():
        for task, resolved in outcomes.items():
            yield _record(COMPARISON, candidate, run, task, resolved,
                          ("resolved" if resolved else "not resolved") + "; rebuilt from flip list")
