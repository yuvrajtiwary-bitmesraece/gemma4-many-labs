"""Adapter for renorai/gemma4-memory-study ("One Run Is Not Enough", Patrick Passos, Renor).

Eligible data: results/run_rows.jsonl, one row per (backend, arm, run, task) of a pre-registered paired comparison of
a Gemma 4 31B agent with and without re-injection of its explorer's last finding ("recap" vs "none") on 98 public
tasks, run with the competition's swegemma harness on three backends (Modal H100: 16 tasks, Kaggle 4xL4: 32 tasks,
AWS L40S: 50 tasks).  nb-out/fake/rows.jsonl is synthetic output of the lab's fake harness and is not used.
"""

REPO = "renorai/gemma4-memory-study"

# The lab pools one arm over the three backends (pre-registered primary: "Run 1 pooled over 48 pairs (Modal 16 +
# Kaggle 32), with the bootstrap stratified by backend"), and each task runs on exactly one backend, so system = arm.
# True would give every backend its own system ('none@kaggle', ...).
SYSTEM_PER_BACKEND = False

CARD = {
    "eligible": True,
    "reason": "",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: 'This study is licensed under the "
                            "Apache License 2.0; see `LICENSE`'"},
    "models": ["31B"],
    "harness": "official",
    "summary": ("Pre-registered paired test of a coder + explorer Gemma 4 31B agent (thinking 4096, 100 tool calls, 60 min) "
                "with ('recap') and without ('none') automatic re-injection of the explorer's latest finding after context "
                "compaction: 98 gold-valid public tasks, each on one of three backends; both arms once everywhere, a second "
                "run of both arms on Kaggle (32 tasks), a second 'none' run on Modal (16 tasks) and a 7-task partial second "
                "'recap' run on Modal."),
    "evidence": {
        "model": "paper/WRITEUP.md s2.1: 'The model is Gemma 4 31B (QAT, w4a16, thinking budget 4096) in the competition's "
                 "declarative ADK harness'; submission/agent.yaml: 'model: gemma-4-31b-it-qat-w4a16-ct'",
        "harness": "paper/WRITEUP.md s2.1: 'Scoring uses the harness's `verify_task` in a per-task Python 3.13 environment built "
                   "from its Docker setup steps'; s5: 'Agents ran under the subprocess sandbox, not the official scorer'; "
                   "kernels/scorer_sandbox.py: 'patch_harness() makes swegemma's Evaluator use DockerSandbox and runs post_setup() "
                   "after every install_test_dependencies call'; results/build_rows.py: '\"success\": int(bool(r[\"resolved\"]))' "
                   "read from each results directory's task_results.jsonl",
        "systems": "paper/WRITEUP.md s2.2: 'none: the baseline submission. Its build is byte-identical across every pin', 'recap: "
                   "after every compaction, the coder's instruction carries two session-state slots ...'; results/README.md: "
                   "'One row per (backend, arm, run, task)'",
        "replicates": "paper/WRITEUP.md s2.4: 'A/A calibration. We compared the no-memory arm against itself, run 2 against run 1 "
                      "(Table 1). Of the 48 tasks, 9 changed outcome (18.75%) with no treatment at all'; DESIGN.md s10 2026-11-08: "
                      "'31B only, 48 tasks x 2 runs x {`none`, `node`}'; results/README.md table: 'Modal (H100) | runs 1-2 | run 1 "
                      "(16 tasks) and run 2 (7 tasks, partial)', 'Kaggle (4x L4) | runs 1-2 | runs 1-2'; column `run` (1 or 2) in "
                      "results/run_rows.jsonl",
        "task_sets": "paper/WRITEUP.md s2.4: 'only 37 of 129 tasks passed with their gold patch ... raised it to 105 ... The 48-task "
                     "subset was drawn once (seed 20261107) from the 105 valid tasks, stratified by repository. The 50 extension tasks "
                     "are the remaining valid tasks, minus 7 used for tuning, smoke and throughput checks'; DESIGN.md s10 "
                     "2026-11-09T00:27Z: '`split_every_third` sends seed ranks 3, 6, 9, ... to Modal; the rest run on Kaggle' "
                     "(results/extension_tasks.txt, results/phase_r_tasks.txt)",
    },
    "decisions": [
        "system = arm ('none', 'recap'); run = backend/arm/run index. The backend is serving infrastructure for the same frozen build, the lab pools an arm over backends in its registered analyses, and no task ran on two backends, so same-task repeats are always the lab's run 1 / run 2 on one backend. The lab does note backend differences (leaked thought text in 5 and 3 of 32 Kaggle no-memory runs against 0 of 16 on Modal: 'a different serving stack is the likeliest explanation'); SYSTEM_PER_BACKEND = True separates them",
        "replicates: kaggle none run1/run2, kaggle recap run1/run2 and modal none run1/run2 (explicit run index, the lab's A/A calibration). AWS has one run per arm",
        "the 7 Modal recap run-2 rows (exploratory_partial = true; the task-runs that finished before the vLLM server hung, 'a biased, short-task subset', all requests tasks) are kept as a partial run with the condition in the system label ('recap@partial'); the lab gates them out of every registered figure",
        "Modal recap run 1 is a composite: 6 task-runs from the execution that hung plus the 10 missing tasks rerun 'once, from scratch' at the same pin (DESIGN.md s10 2026-11-09T04:10Z); the lab treats it as the complete run 1 and so does the adapter. The 19 task-runs in flight at the hang are 'not run, not failures' and are not in the file (9 run-2 tasks were never rerun)",
        "harness = official: swegemma eval and its verify_task score every attempt, but in the lab's scorer-like sandbox (clean Python 3.13 venv per task, competition wheels plus PyPI test dependencies and per-snapshot version pins) on Modal, Kaggle and AWS; flag lab_provisioned_sandbox",
        "task sets are restricted to the 105 tasks the lab's reference-patch check found valid (seeded draw of 48, then the other 50 minus 7 used for tuning); selected on control outcomes, not on agent outcomes",
        "err = context_overflow or cap_calls or cap_time or cap_turns (ContextWindowExceeded in the harness error, 100-tool-call cap, 60-minute cap); submit_loop, leaked_thought and plumbing_failed are behaviour flags of the lab, not harness errors",
    ],
    "excluded": [
        "nb-out/fake/rows.jsonl: 1,032 rows with model 'FAKE-gemma-4-31b-synthetic' written by fake_harness/ (arms none, node, shuffled, recap; 3 runs); synthetic demo data for the notebook, not model runs",
        "results/analysis.json, results/pooled_run1_v2.json|md, results/aa_calibration.json, nb-out/paper-tables.md: aggregates computed from run_rows.jsonl",
        "Phase R replicate runs (aws-l40s, run = 100 + k; pre-registered in DESIGN.md s10 2026-11-09T20:05Z, 'reported separately'): no rows in this snapshot",
        "not in the repository: the 16-task no-memory throughput run on Kaggle ('these are not results'), recap smoke / check runs on 2 tasks, 12B runs (never completed), the arms node / text / shuffled (not run on real tasks)",
        "controls: per-task validity results (reference patch and base commit) exist only as prose in DESIGN.md s10 (2026-11-09T00:24Z), see CARD['controls']",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["partial_runs", "tasks_restricted_to_gold_checked", "lab_provisioned_sandbox", "three_serving_backends",
              "composite_runs"],
    "sampling": ("no temperature or seed set: PREREG.md s3 'thinking_budget 4096, include_thoughts true, max_output_tokens 16384, and "
                 "the harness's defaults for every other sampling parameter' (submission/agent.yaml). Run 2 is an unchanged rerun of "
                 "run 1 (same build and environment hashes); concurrency 8 on Kaggle, both Modal no-memory runs at once (16 in flight)"),
    "leaderboard": "none (no leaderboard score of this agent is stated; DESIGN.md quotes other teams' public numbers only)",
    "controls": [
        {"path": "DESIGN.md", "member": "s10 entry 2026-11-09T00:24Z 'validity funnel' (prose, ids listed)", "kind": "reference_patch",
         "tasks": 129, "resolved": 105,
         "environment": "lab's scorer-like sandbox, envcheck v3 (clean Python 3.13 venv per task, competition wheels, PyPI test dependencies, per-snapshot versions); 105 valid = reference patch resolves and the base commit fails cleanly; of the 24 others 11 'gold not resolved' (fastapi 14356, 14186, 15745, 15763, 15785; requests 6757, 6589, 6629, 6592; rich 3486, 3472), 11 fastapi 'base errors besides failures', 2 'tests pass on base' (fastapi 14851, 14077). The 105 valid ids = results/phase_r_tasks.txt (98) + the 7 tuning tasks named in the 2026-11-09T10:47Z entry. Earlier stages, counts only: 37 valid in the public environment, 68 after adding test dependencies"},
    ],
    "reported_numbers": [
        {"what": "rows per backend / arm / run", "value": "Modal: none runs 1-2 (16 each), recap run 1 (16) and run 2 (7, partial); Kaggle: both arms runs 1-2 (32 each); AWS: 50 per arm, one run",
         "where": "results/README.md; paper/WRITEUP.md abstract", "adapter": "16, 16, 16, 7; 32 x 4; 50, 50 = 283 rows, 98 tasks", "match": True},
        {"what": "primary, run 1, 48 pairs: recap / none resolved", "value": "29/48 vs 25/48 (recap-only 5, none-only 1; exact McNemar p = 0.22)",
         "where": "paper/WRITEUP.md Table 2; results/analysis.json primary_resolve_run1", "adapter": "29/48 vs 25/48; 5 / 1; p = 0.21875", "match": True},
        {"what": "per backend, run 1: recap / none", "value": "Modal 12/11 of 16 (2 / 1); Kaggle 17/14 of 32 (3 / 0); AWS 18/23 of 50 (1 / 6, p = 0.125)",
         "where": "paper/WRITEUP.md Table 2", "adapter": "Modal 12/11 (2 / 1); Kaggle 17/14 (3 / 0); AWS 18/23 (1 / 6, p = 0.125)", "match": True},
        {"what": "combined 98 pairs, run 1", "value": "47/98 vs 48/98 (6 / 7), p = 1.0", "where": "paper/WRITEUP.md Table 2",
         "adapter": "47 vs 48; 6 / 7; p = 1.0", "match": True},
        {"what": "A/A, no memory run 1 / run 2 resolved", "value": "Kaggle 14 / 16 (b=4, c=2); Modal 11 / 12 (b=2, c=1); 9 of 48 tasks changed outcome (18.75%)",
         "where": "paper/WRITEUP.md Table 1; results/aa_calibration.json", "adapter": "Kaggle 14 / 16 (4 gained, 2 lost); Modal 11 / 12 (2 gained, 1 lost); 9 of 48 flipped", "match": True},
        {"what": "overflows, no memory run 1 / run 2", "value": "Kaggle 5 / 4, Modal 1 / 0; extension run 1: 12 of 50 (none), 14 of 50 (recap)",
         "where": "paper/WRITEUP.md Table 1, s3.1", "adapter": "context_overflow: Kaggle 5 / 4, Modal 1 / 0; AWS 12 and 14", "match": True},
        {"what": "Kaggle recap run 2", "value": "32 rows, sensitivity only", "where": "results/analysis.json task_runs_used",
         "adapter": "17/32", "match": True},
        {"what": "Modal no-memory baseline", "value": "23 of 32 runs (72%) on its 16 tasks", "where": "DESIGN.md s10 2026-11-09T01:35Z; paper/WRITEUP.md s5",
         "adapter": "11 + 12 = 23 of 32", "match": True},
    ],
}

ROWS = r"^results/run_rows\.jsonl$"


def attempts(src):
    for path, rows in src.jsonl(ROWS):
        for row in rows:
            backend, arm, run = row["backend"], row["arm"], row["run"]
            system = f"{arm}@{backend}" if SYSTEM_PER_BACKEND else arm
            if row.get("exploratory_partial"):
                system += "@partial"
            flags = [k for k in ("context_overflow", "cap_calls", "cap_time", "cap_turns") if row.get(k)]
            y = int(row["success"])
            yield {
                "file": path,
                "system": system,
                "run": f"{backend}/{arm}/run{run}",
                "task": row["task_id"],
                "y": y,
                "model": "31B",
                "harness": "official",
                "err": bool(flags),
                "status": ("resolved" if y else "failed") + ("; " + ",".join(flags) if flags else ""),
            }
