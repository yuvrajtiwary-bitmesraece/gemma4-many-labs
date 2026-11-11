"""Adapter for ShreySharma07/gemma4-swe-agent ("Found, Not Fixed: Counterfactual Diagnostics of Partial Repairs in a
Quantized Code Agent"; team Shrey Sharma + Atul Baliyan).

Eligible data: twelve executions of Gemma 4 31B agents on public development-set tasks, all scored by the competition's
swegemma Evaluator (subprocess sandbox, one task at a time) in Kaggle 4xL4 notebooks:

  results/runs/{r1_baseline_v1,r2_v2_sed,r3_v5_graph}/task_results.jsonl   three early agents, 129 tasks each
  kaggle/eval_l4/regrade_kernel/gemma4-r1-regrade.ipynb                    embedded payload (base64 tar.gz in the cell
                                                                           variable NEC_B64): per_task.jsonl of the five
                                                                           repeats of v7_noedit on the 54 "valid" tasks
  data/runs/*new60/**/analysis/{per_task,original_per_task}.jsonl          v7, v7 + LoRA s3b, v14, v16 on the 60 "new" tasks

The rest of the repository is rollouts on synthetic "factory" tasks, LoRA training, counterfactual re-grading of agent
patches, retrieval studies and literature notes.  The payload is decoded in memory; nothing is unpacked or executed.
"""
import base64
import io
import json
import re
import tarfile

REPO = "ShreySharma07/gemma4-swe-agent"

# Step D (v8_conv_lora, a collapsed LoRA adapter) resolved 0 of the 114 valid tasks in one pass.  No per-task file is
# published, but 0/114 fixes every per-task outcome.  False follows the README rule (no rows from aggregates); True yields
# 114 rows with y = 0 for the ids of data/runs/eval_sweep_v2/valid_tasks.json (system 'v8_conv_lora@lora-conv').
INCLUDE_ALL_ZERO_STEP_D = False

CARD = {
    "eligible": True,
    "reason": "",
    "licence": {"spdx": "MIT",
                "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Shrey Sharma Permission is hereby granted, free of charge, "
                            "to any person obtaining a copy of this software and associated documentation files'; README.md "
                            "'License: MIT.'; research_deliverables/kaggle_writeup_fields.md 'Code, analysis scripts and "
                            "aggregate results (MIT)'"},
    "models": ["31B"],
    "harness": "official",
    "summary": ("Gemma 4 31B (QAT W4A16) agents of one team (two Kaggle accounts) scored with the official swegemma Evaluator "
                "in Kaggle 4xL4 notebooks, one task at a time: three early agents on all 129 public tasks (r1-r3), the v7 "
                "agent five times on the 54 tasks whose reference patch verified in the lab's scorer-faithful sandbox, and "
                "v7, v7 + LoRA 's3b', v14 and v16 once each on 60 further tasks made verifiable with test-only overlays."),
    "evidence": {
        "model": "README.md: 'Post-training Gemma 4 31B (QAT W4A16) into an offline software-engineering agent'; "
                 "submissions/*/agent.yaml: 'model: gemma-4-31b-it-qat-w4a16-ct'; research_deliverables/paper_draft.md s2: "
                 "'Gemma-4-31B-it QAT W4A16 is served by vLLM 0.19.1 (TP=4, 4xL4), using the official harness and tool set'",
        "harness": "docs/experiments.md: 'Local = scorer-faithful run on Kaggle L4x4 (`run_official_l4.py`, subprocess sandbox), "
                   "129 public tasks unless noted'; kaggle/archive/run_official_l4_v1.py: 'official swegemma Evaluator, subprocess "
                   "sandbox, tasks run ONE AT A TIME under the submission's eval_config.yaml'; kaggle/eval_l4/evalkit/runner.py: "
                   "'the official swegemma Evaluator, one task at a time, in budgeted chunks' (EvalConfig(..., sandbox=\"subprocess\", "
                   "... concurrency=self.s.concurrency ...), default concurrency 1 = scorer)",
        "systems": "submissions/README.md table (one row per submission folder with its one change: baseline_v1 'organisers' sample "
                   "agent with raised budgets', v2_sed 'no `read_file`; grep/sed reads', v5_graph 'v4 + code-map skill', v7_noedit "
                   "'v2 minus `edit_file`', v14_thinking 'v13 + thinking ON (budget 4096, include_thoughts) + public budgets', "
                   "v16_small_output 'v14 + one prompt section'); docs/post_training_guide.md: '`submissions/v12_s3b` = v7 + "
                   "`adapter: s3b`'; data/runs/*/summary.json 'name' gives the submission of each run",
        "replicates": "docs/experiments.md row rs1: 'Step 2 rejection sampling: v7 x 5 reps on the 54 valid tasks (check-l4 v10, "
                      "9.7 h, `--valid-only --repeats 5`) | 16/16/16/16/18 of 54'; data/runs/rs_v7_valid/summary.json: 'name': "
                      "'v7_noedit', 'reps': {'rep0': {'resolved': 16, 'n': 54}, ..., 'rep4': {'resolved': 18, 'n': 54}}; the payload "
                      "members are rs_v7_valid/x/eval_v7_noedit/rep0..rep4/analysis/per_task.jsonl. No other configuration was run "
                      "twice on the same tasks",
        "task_sets": "docs/experiments.md r6: 'Valid set (sweep, CPU): 54/129 tasks where gold passes and empty fails (fastapi 9, "
                     "rich 39, requests 5, httpx 1 ...)' (data/runs/eval_sweep/valid_tasks.json); sw2: 'overlays kernel ... "
                     "test-only extras for every task ... per-task pins ... then gold/empty sweep | 114/129 valid (was 54; none "
                     "lost)' (data/runs/eval_sweep_v2/valid_tasks.json; the 60 added ids are new_tasks.json, 'new-60'); r1-r3 ran "
                     "on all 129",
    },
    "decisions": [
        "system = submission folder (agent configuration); run = one evaluator pass (run directory + rep index)",
        "the five rs_v7_valid passes are replicates of system 'v7_noedit' (explicit rep0..rep4 of one notebook run with --repeats 5); the v7 pass on the new-60 tasks is the same submission on a disjoint task set and gets the same system label (run 'eval_v7_new60/rep0')",
        "the rs_v7_valid per-task rows exist only inside kaggle/eval_l4/regrade_kernel/gemma4-r1-regrade.ipynb: its cell variable NEC_B64 is a base64 tar.gz ('agent patches + per-task rows + old-54 LOO results', kaggle/eval_l4/make_eval_notebook.py) that the lab embedded for its re-grading kernel. The adapter decodes it in memory and reads the five per_task.jsonl members; `file` is the notebook path. The same payload also holds copies of the v7 and v14 new-60 per_task.jsonl, which are NOT read from there (identical to the plain files: same outcome and duration on 60/60 tasks)",
        "v14 new-60 is one execution with four representations (data/runs/atul_eval_v14_new60/.../per_task.jsonl, the payload copy 'eval_v14_new60_peek', .../atul_eval_v16_new60/.../v14_per_task.jsonl, docs/optimization_v14/trace_audit.json: identical outcomes and durations); only the first is yielded. v16 new-60 likewise (original_per_task.jsonl yielded, v16_per_task.jsonl is a later audit of the same run)",
        "r1-r3: task_results.jsonl (harness output) is yielded; per_task.jsonl (r1, r2) and behaviour_per_task.json (r3) are trace-analyzer tables of the same runs with identical `resolved`",
        "r1 (baseline_v1) is kept on all 129 tasks although the lab says '32 fastapi tasks could not pass in our local environment whatever the patch' (missing inline_snapshot / dirty_equals; docs/findings.md) and also reports 15/97 verifiable: the r1 files carry no per-task env flag (34 fastapi rows have test_exit_code 2), so the 32 cannot be identified and dropped; flag r1_env_unverifiable_tasks_kept. r2 and r3 ran after the dependency fix ('all 129 verifiable')",
        "r1-r3 used the lab's first runner (kaggle/archive/run_official_l4_v1.py, harness swegemma 0.2.7, old Kaggle image, context compaction every 5 events / 14336 tokens instead of the scorer's 15 / 32768: docs/experiments.md 'Fidelity notes'); later runs use the scorer-faithful evalkit runner (swegemma 0.2.11, Python 3.13 scorer-site sandboxes). The lab itself calls r2's 26 'not comparable' with later numbers ('fastapi 2/67 vs 16/67 is environment')",
        "v12_s3b is v7 with a LoRA adapter trained on converted public teacher trajectories; kept with the condition in the label ('v12_s3b@lora-s3b') and flag fine_tuned",
        "old-54 and new-60 are the tasks that passed the lab's reference-patch / empty-patch sweep, i.e. selected on control outcomes, not on agent outcomes; flag tasks_restricted_to_gold_checked",
        "y = harness `resolved`; err = non-empty harness error (`error` in task_results.jsonl, `harness_error` in per_task.jsonl: session timeout, tool-call or turn budget, context-window overflow, no submit_patch call, test patch does not apply after the agent edited test files). test_exit_code -1 (tests not run after an agent-phase error) is an agent failure, not an infrastructure drop",
        "Step D (v8_conv_lora, 0/114, all patches empty) has no per-task file; not yielded by default (INCLUDE_ALL_ZERO_STEP_D)",
    ],
    "excluded": [
        "results/runs/{fs1,r2_s0,r2_s1}/task_results.jsonl, results/runs/{r3_s0,r3_s1}/factory_rollout_report.json, results/runs/g1/l4_v1_report.json: v14 (and v14 + LoRA rs2) rollouts on the lab's synthetic 'factory' tasks (mutated bugs in boltons, sqlparse, more-itertools, toolz, pathspec ...; ids fac_*, f1_*, f2_*), outside the development set",
        "results/runs/factory_v0|v1|v2/verify_gold.jsonl, verify_empty.jsonl, task_meta.jsonl: reference-patch / empty-patch verification of the synthetic factory tasks",
        "aggregated only, no per-task rows in the snapshot: r6 = v7_noedit on all 129 tasks (19/129; data/runs/eval_v7_full/summary.json; run.log shows one task per chunk), v17_superagent on new-60 (27/60, docs/experiments.md), Step D = v8_conv_lora on the 114 valid tasks (0/114, docs/step_d_d0_20261016_report.md), Step B = v8_lora_smoke on 8 tasks (2/8), eval_v7_chunk0_aborted (aborted run)",
        "results/runs/{r1_baseline_v1,r2_v2_sed}/per_task.jsonl, results/runs/r3_v5_graph/behaviour_per_task.json, data/runs/atul_eval_v16_new60/**/v14_per_task.jsonl and v16_per_task.jsonl, docs/optimization_v14/trace_audit.json, and the payload copies of the v7 / v14 new-60 per_task.jsonl: further tables of runs already yielded",
        "results/runs/r1/results.jsonl: re-grading on CPU of 40 agent patches of runs already yielded ('orig') and of 25 patches with stray files stripped ('stripped'); not new agent executions (the lab: 'Originals re-graded identically to their runs (40/40)')",
        "results/runs/a3a4/results.jsonl, results/runs/a12/results.jsonl, results/runs/a12v/results.jsonl, swaps.json, swaps_v2.json, cases.json: counterfactual patches (reference patch minus one function group, agent patch plus reference hunks, agent function swapped into the reference tree) and their gold / empty controls; non-agent runs",
        "run_traces/traces/*.json (= r2_v2_sed) and traces2/*.json (= r3_v5_graph): raw ATIF traces of two yielded runs, no outcome field",
        "data/runs/atul_gate, data/runs/gate_v2, data/runs/gate/prefixes.json: single-decision 'behaviour gate' samples on 30 trajectory prefixes (base vs adapters), not task outcomes",
        "results/runs/{e0a,e0b,a8,a8b,a1,a13,mutation_metrics,rs1,rs2,lc1,lp1,lp2,lt1,pa1,t1,p0_lora_serving}, research_deliverables/atul_t1_lora_analysis_2026-11-04: retrieval hit rates (incl. 120 SWE-rebench tasks), regressions, trace metrics, training-set and LoRA weight/timing reports",
        "public leaderboard scores in submissions/README.md and docs/experiments.md (hidden tasks)",
        "data/runs/*.npz, *.tar.gz, kaggle/train/**: training windows (incl. 200 public OpenHands runs of other models) and training notebooks",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["fine_tuned", "tasks_restricted_to_gold_checked", "runs_from_embedded_notebook_payload", "aggregated_only",
              "r1_env_unverifiable_tasks_kept", "r1_r3_early_runner_compaction_mismatch", "two_kaggle_accounts_one_team"],
    "sampling": ("temperature 0.2, top_p 0.95, top_k 40, no seed in any submissions/*/configs/sampling.yaml (thinking off for "
                 "baseline_v1, v2, v5, v7, v12; thinking budget 4096 for v14, v16). The five v7 repeats are plain re-runs of the "
                 "same notebook (`--repeats 5`), tasks one at a time; docs/experiments.md r10 speaks of 'agent randomness at "
                 "temperature 0.2'. `seed: 0` in the runners only fixes the task order"),
    "leaderboard": [
        "baseline_v1: LB 0.00 vs local 15/129 (15/97 verifiable) - docs/experiments.md r1, submissions/README.md",
        "v2_sed: LB 0.08 (docs/experiments.md r5 lists 'v2 0.08/0.08/0.08/0.06') vs local 26/129 - docs/experiments.md r2",
        "v5_graph: LB 0.08 vs local 33/129 - docs/experiments.md r3 ('our local set no longer predicts the LB', docs/findings.md)",
        "v7_noedit: LB 0.15 (ref 57042679) vs local 16, 16, 16, 16, 18 of 54 (five repeats), 23/60 on new-60, 19/129 on all tasks (aggregated) - docs/experiments.md r5, r6, rs1; docs/handoff_20261027.md",
        "v14_thinking: LB 0.17 (ref 57425260) vs local new-60 30/60 - submissions/README.md, docs/handoff_20261027.md",
        "no per-task data here: v9 0.12, v10b 0.10, v11 0.10, v13 0.10, v15b 0.08, v17 0.18 (v17 local new-60 27/60, aggregated); v16 and v12_s3b were not submitted",
    ],
    "controls": [
        {"path": "kaggle/eval_l4/regrade_kernel/gemma4-r1-regrade.ipynb", "member": "NEC_B64 payload: xg/results.jsonl, mode 'gold'",
         "kind": "reference_patch", "tasks": 54, "resolved": 54,
         "environment": "Kaggle CPU kernel, swegemma scorer-faithful verification (subprocess sandbox, scorer site, no overlays), the 54 gold-passing tasks of the exec-graph instrument (53 of the old-54 valid tasks + fastapi_14873)"},
        {"path": "kaggle/eval_l4/regrade_kernel/gemma4-r1-regrade.ipynb", "member": "NEC_B64 payload: xg/results.jsonl, mode 'empty'",
         "kind": "empty_patch", "tasks": 54, "resolved": 1,
         "environment": "same kernel and tasks; fastapi_14873 passes with no patch"},
        {"path": "results/runs/a3a4/results.jsonl", "member": "mode 'gold'", "kind": "reference_patch", "tasks": 22, "resolved": 22,
         "environment": "Kaggle CPU kernel, official Evaluator with a fake agent, scorer site + test overlays / version pins, offline; the 22 new-60 tasks whose reference patch touches >= 2 function groups"},
        {"path": "results/runs/a3a4/results.jsonl", "member": "mode 'empty'", "kind": "empty_patch", "tasks": 22, "resolved": 0,
         "environment": "same kernel and tasks"},
        {"path": "results/runs/a12/results.jsonl", "member": "mode 'gold_rerun'", "kind": "reference_patch", "tasks": 29, "resolved": 29,
         "environment": "same CPU stack with overlays; the 29 tasks of the site-swap population (14 old-54, 15 new-60)"},
        {"path": "results/runs/a12v/results.jsonl", "member": "mode 'gold_rerun'", "kind": "reference_patch", "tasks": 29, "resolved": 29,
         "environment": "same stack and tasks, separate kernel run three days later"},
        {"path": "data/runs/eval_sweep/valid_tasks.json", "member": "list of ids", "kind": "reference_patch", "tasks": 129, "resolved": 54,
         "environment": "Kaggle CPU sweep in the scorer-faithful sandbox without overlays; the list holds the tasks where the reference patch passes AND the empty patch fails (the two outcomes are not published separately)"},
        {"path": "data/runs/eval_sweep_v2/valid_tasks.json", "member": "list of ids", "kind": "reference_patch", "tasks": 129, "resolved": 114,
         "environment": "same sweep with test-only overlays and per-task Starlette / Pygments pins; reference passes AND empty fails (fastapi 59/67, rich 45/48, requests 9/13, httpx 1/1)"},
    ],
    "reported_numbers": [
        {"what": "r1 baseline_v1, r2 v2_sed, r3 v5_graph resolved", "value": "15/129, 26/129, 33/129",
         "where": "docs/experiments.md rows r1-r3; submissions/README.md 'Local (129)'", "adapter": "15/129, 26/129, 33/129", "match": True},
        {"what": "r1 on verifiable tasks", "value": "15/97 (32 fastapi tasks env-unverifiable)", "where": "docs/findings.md; docs/experiments.md r1",
         "adapter": "not reproducible: no per-task env flag for r1 (15/129 over all rows; 34 fastapi rows with test_exit_code 2)", "match": False},
        {"what": "r1 vs r2 paired", "value": "both 13, baseline only 2, v2 only 8 on the 97 common verifiable tasks; v2 solved 5 of the 32 env-broken tasks",
         "where": "docs/findings.md 'Run 2'", "adapter": "over 129: both 13, r1 only 2, r2 only 13 (= 8 + 5)", "match": True},
        {"what": "r2 vs r3", "value": "gained 12, lost 5; by repo fastapi 16 -> 21, rich 9 -> 11, requests 1 -> 1; 21 timeouts in r3",
         "where": "docs/findings.md 'Run 3'; docs/experiments.md r3", "adapter": "gained 12, lost 5; fastapi 16 -> 21, rich 9 -> 11, requests 1 -> 1; 21 session-timeout errors",
         "match": True},
        {"what": "v2 and v5 on the 54 valid tasks", "value": "v5 14/54, v2 12/54", "where": "docs/experiments.md r6",
         "adapter": "r3 14/54, r2 12/54 on the ids of data/runs/eval_sweep/valid_tasks.json", "match": True},
        {"what": "v7 x 5 on the 54 valid tasks", "value": "16/16/16/16/18 of 54; pass@5 = 21/54, pass^5 = 9/54; solves per task 0:33, 1:2, 2:1, 3:3, 4:6, 5:9; 82 solved trajectories",
         "where": "docs/experiments.md row rs1; data/runs/rs_v7_valid/summary.json",
         "adapter": "16, 16, 16, 16, 18 of 54; any 21, all 9; histogram 0:33, 1:2, 2:1, 3:3, 4:6, 5:9; 82 resolved attempts", "match": True},
        {"what": "v7 new-60", "value": "23/60", "where": "data/runs/eval_v7_new60/summary.json; docs/atul_codex_v7_new60.md", "adapter": "23/60", "match": True},
        {"what": "v14 new-60 vs v7", "value": "30/60 vs 23/60: 22 both, 8 v14-only, 1 v7-only, 29 neither; exact McNemar p = 0.0390625; 8 context errors",
         "where": "docs/experiments.md row v14-new60", "adapter": "30 vs 23: 22 both, 8 v14-only, 1 v7-only, 29 neither; p = 0.0390625; 8 context-window errors", "match": True},
        {"what": "v16 new-60 vs v14", "value": "27/60 vs 30/60: 26 both, 1 v16-only, 4 v14-only, p = 0.375; overflows 8 -> 5",
         "where": "submissions/README.md; docs/post_training_guide.md", "adapter": "27 vs 30: 26 both, 1 v16-only, 4 v14-only; p = 0.375; 5 context-window errors", "match": True},
        {"what": "v12_s3b new-60 vs v7", "value": "9/60 vs 23/60: 9 both, 0 S3b-only, 14 v7-only, 37 neither; exact McNemar p = 0.00012207",
         "where": "research_deliverables/atul_independent_research_audit_2026-10-27.md; docs/post_training_guide.md",
         "adapter": "9 vs 23: 9 both, 0 s3b-only, 14 v7-only, 37 neither; p = 0.00012207", "match": True},
    ],
}

RAW = {  # harness task_results.jsonl of the three 129-task runs
    "results/runs/r1_baseline_v1/task_results.jsonl": ("baseline_v1", "r1_baseline_v1"),
    "results/runs/r2_v2_sed/task_results.jsonl": ("v2_sed", "r2_v2_sed"),
    "results/runs/r3_v5_graph/task_results.jsonl": ("v5_graph", "r3_v5_graph"),
}
PER_TASK = {  # trace-analyzer per-task rows (harness `resolved` and `harness_error`) of the new-60 runs
    "data/runs/eval_v7_new60/x/eval_v7_noedit/rep0/analysis/per_task.jsonl": ("v7_noedit", "eval_v7_new60/rep0"),
    "data/runs/eval_s3b_new60/x/eval_v12_s3b/rep0/analysis/per_task.jsonl": ("v12_s3b@lora-s3b", "eval_s3b_new60/rep0"),
    "data/runs/atul_eval_v14_new60/eval_v14_thinking/rep0/analysis/per_task.jsonl": ("v14_thinking", "atul_eval_v14_new60/rep0"),
    "data/runs/atul_eval_v16_new60/eval_v16_small_output/rep0/analysis/original_per_task.jsonl":
        ("v16_small_output", "atul_eval_v16_new60/rep0"),
}
NOTEBOOK = "kaggle/eval_l4/regrade_kernel/gemma4-r1-regrade.ipynb"
VALID114 = "data/runs/eval_sweep_v2/valid_tasks.json"
PAYLOAD = re.compile(r"^NEC_B64\s*=\s*'([A-Za-z0-9+/=]+)'", re.M)
REP = re.compile(r"^rs_v7_valid/x/eval_v7_noedit/(rep\d)/analysis/per_task\.jsonl$")
WANTED = "^(" + "|".join(re.escape(p) for p in [*RAW, *PER_TASK, NOTEBOOK, VALID114]) + ")$"


def _rows(raw):
    out = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _record(path, system, run, row, error):
    resolved = bool(row["resolved"])
    status = "resolved" if resolved else (str(error)[:48] if error else "failed")
    if "test_exit_code" in row:
        status += f"; exit={row['test_exit_code']}"
    return {
        "file": path,
        "system": system,
        "run": run,
        "task": row["instance_id"],
        "y": int(resolved),
        "model": "31B",
        "harness": "official",
        "err": bool(error),
        "status": status,
    }


def _embedded_repeats(raw):
    """(rep label, rows) for the five v7 repeats stored in the notebook's NEC_B64 payload, in archive order."""
    notebook = json.loads(raw.decode("utf-8", "replace"))
    for cell in notebook.get("cells", []):
        match = PAYLOAD.search("".join(cell.get("source", [])))
        if not match:
            continue
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(match.group(1))), mode="r:gz") as tf:
            for member in tf:
                rep = REP.match(member.name)
                if rep and member.isfile():
                    yield rep.group(1), _rows(tf.extractfile(member).read())


def attempts(src):
    valid114 = []
    for path, raw in src.read_many(WANTED):
        if path in RAW:
            system, run = RAW[path]
            for row in _rows(raw):
                yield _record(path, system, run, row, row.get("error"))
        elif path in PER_TASK:
            system, run = PER_TASK[path]
            for row in _rows(raw):
                yield _record(path, system, run, row, row.get("harness_error"))
        elif path == NOTEBOOK:
            for rep, rows in _embedded_repeats(raw):
                for row in rows:
                    yield _record(path, "v7_noedit", f"rs_v7_valid/{rep}", row, row.get("harness_error"))
        elif path == VALID114:
            valid114 = json.loads(raw.decode("utf-8", "replace"))
    if INCLUDE_ALL_ZERO_STEP_D:
        for task in valid114:
            yield {"file": "docs/step_d_d0_20261016_report.md", "system": "v8_conv_lora@lora-conv", "run": "atul_step_d_20261016/rep0",
                   "task": task, "y": 0, "model": "31B", "harness": "official", "err": None,
                   "status": "reconstructed: 0/114 resolved, all patches empty"}
