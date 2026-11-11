"""Adapter for sundarshahi/gemma4-agents-rarely-lose-a-fix ("Run the Real Tests: Scoring every intermediate step of
345 Gemma 4 coding-agent runs").

Eligible data: one live run of the official sample agent on Gemma 4 E4B over the 129 public tasks.  The 298 Gemma 4 31B
task-runs in results/run-tn are replays of the public traces of TechNaom/gemma4-agentic-swe, a source that is already in
the pool, and are not yielded.
"""

REPO = "sundarshahi/gemma4-agents-rarely-lose-a-fix"

# The lab's own audit (reference patch must pass, empty patch must fail) leaves 72 of the 129 tasks, and the lab reports
# every result on those 72.  True follows the lab (README rule: attempts the source marks as invalid for infrastructure
# reasons are dropped); False would yield all 129 rows of the E4B run (11/129).
DROP_AUDIT_INVALID = True

CARD = {
    "eligible": True,
    "reason": "",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License' / 'Apache-2.0.'; "
                            "README.md '## Data policy': '`results/` holds agent-produced diffs and verdicts, which the hosts "
                            "confirmed may be published as derived labels (competition discussion 749803)'"},
    "models": ["E4B"],
    "harness": "official",
    "summary": "One live run of the official sample submission with the model swapped to Gemma 4 E4B (100 tool calls, 100 "
               "turns, 10 minutes per task, vLLM on Kaggle 4x L4, 8 tasks in flight) on all 129 public tasks, scored by the "
               "swegemma Evaluator; the lab reports it on the 72 tasks its gold/empty audit found valid (9 of 72).",
    "evidence": {
        "model": "results/run-e4b/run_meta.json: \"model\": \"gemma-4-e4b-it-qat-w4a16-ct\", \"run_id\": \"e4b-s1\"; "
                 "paper/writeup.md s3.2: 'Gemma 4 E4B, run live. We ran the official sample submission with only the model "
                 "name changed.'",
        "harness": "kernels/run_body.py: 'from swegemma.evaluate import Evaluator' ... EvalConfig(..., sandbox='subprocess', "
                   "...) ... 'result = asyncio.run(Evaluator(cfg).run())' and final.jsonl is written from "
                   "result.task_results ('resolved': bool(r.resolved), 'error': r.error); paper/writeup.md s8: 'Each stage "
                   "is a Kaggle notebook on the official harness'",
        "systems": "paper/writeup.md s3.2: 'It has the six workspace tools, the three graph tools, a code-analyzer sub-agent, "
                   "temperature 0.2 and a 4,096-token thinking budget. The sample's own budget of 10 tool calls and one "
                   "minute per task is too short to study, so we allowed 100 tool calls, 100 turns and 10 minutes. It ran "
                   "with vLLM on four L4 GPUs.'; kernels/run_body.py removes the sample's dummy LoRA adapters",
        "replicates": "none: a single run (run id e4b-s1). kernels/make_kernels.py also defines 'g4-regret-run-31b-s1', but "
                      "'31B live run was stopped (GPU quota)' and it is not published",
        "task_sets": "paper/writeup.md s3.1: 'The public development set has 129 tasks. We ran the official verifier on each "
                     "one twice, once with the reference fix (it should pass) and once with no change (it should fail). 72 "
                     "tasks behave correctly: 43 from rich and 29 from fastapi. ... Every result below uses only these 72.'; "
                     "analysis/analyze.py: 'valid = {i for i in g if g[i] and e.get(i) is False}'",
    },
    "decisions": [
        "one system 'e4b-sample-agent' (the unmodified official sample agent apart from the model name, removed dummy "
        "adapters and raised budgets), one run 'e4b-s1'",
        "the 57 attempts on tasks outside the lab's 72 audit-valid tasks are dropped (54 tasks whose reference patch "
        "fails in the lab's Kaggle subprocess sandbox, 3 whose tests pass with an empty patch); the two E4B 'resolved' "
        "rows among them are on empty-pass tasks (fastapi_14077, requests_6644); set DROP_AUDIT_INVALID = False to yield "
        "129 rows (11/129)",
        "results/run-tn (298 task-runs, 17 runs tn-abl1-k0 ... tn-pilot2-n5, Gemma 4 31B) is NOT yielded: these are "
        "another lab's executions (paper/writeup.md s3.2: 'Our own 31B run was stopped when the weekly GPU quota ran out. "
        "Instead we used 298 public 31B runs from Papasani [2], covering 17 agent configurations'; README.md: 'The replayed "
        "trajectories are from TechNaom/gemma4-agentic-swe (Apache-2.0, commit 0758ce1)'). Each of the 17 runs has the same "
        "task set and the same outcome on every task as a runs.jsonl of the TechNaom snapshot already in the pool",
        "the flag possible_duplicate_of concerns only results/run-tn; the yielded E4B run is the lab's own execution",
        "y = `resolved` of results/run-e4b/final.jsonl (the Evaluator's own verdict of the live run); the per-state "
        "re-verifications in results/verify-live-0 are not attempts",
        "err = the evaluator's error string is non-empty (context-window overflow raised through litellm, session "
        "timeout); these are agent failures (y = 0), not infrastructure drops",
    ],
    "excluded": [
        "results/run-tn/final.jsonl, states.jsonl, replay_fidelity.jsonl: replays of TechNaom/gemma4-agentic-swe traces "
        "(same executions as that source)",
        "results/audit/verdicts.jsonl: reference-patch and empty-patch controls (see controls)",
        "results/verify-*/verdicts.jsonl: official-verifier verdicts on intermediate workspace states (1,431 distinct "
        "states), not final outcomes of attempts",
        "results/run-e4b/states.jsonl: workspace diff after every mutating tool call, no outcome",
        "results/summary.json, results/checks.json, results/reach.json: derived tables",
        "results/gold_hunks.json: files and line ranges of the reference patches",
    ],
    "dropped": {
        "task not audit-valid: reference patch fails in the lab's sandbox": 54,
        "task not audit-valid: tests pass with an empty patch (fastapi_14077, fastapi_14851, requests_6644)": 3,
    },
    "flags": ["sandbox_unsolvable_tasks_dropped",
              "possible_duplicate_of:TechNaom/gemma4-agentic-swe (results/run-tn only; excluded from attempts)"],
    "sampling": "temperature 0.2, thinking budget 4,096 tokens (the sample submission's settings, paper/writeup.md s3.2); "
                "no seed is set in kernels/run_body.py; a single run, no repeats",
    "leaderboard": "none",
    "controls": [
        {"path": "results/audit/verdicts.jsonl", "kind": "reference_patch",
         "environment": "official verify_task (swegemma.harness.verification) in a Kaggle CPU kernel, SubprocessManager "
                        "sandbox, competition wheels (kernels/verify_body.py mode 'audit')",
         "tasks": 129, "resolved": 75},
        {"path": "results/audit/verdicts.jsonl", "kind": "empty_patch",
         "environment": "same kernel and verifier, empty agent patch",
         "tasks": 129, "resolved": 3},
    ],
    "reported_numbers": [
        {"what": "E4B live run, solved of the valid tasks", "value": "9 of 72",
         "where": "paper/writeup.md s4.3 table; results/summary.json report['e4b-s1'] (n_tasks 72, final_resolved 9)",
         "adapter": "9/72", "match": True},
        {"what": "audit: valid tasks and their repositories", "value": "72 (43 rich, 29 fastapi)",
         "where": "paper/writeup.md s3.1; results/summary.json valid_tasks",
         "adapter": "gold passes 75/129, empty passes 3/129, valid 72 (43 rich, 29 fastapi); same 72 ids as summary.json",
         "match": True},
        {"what": "E4B: re-verified final version agrees with the logged result", "value": "66 of 66",
         "where": "paper/writeup.md s3.4; results/summary.json pooled['E4B live'] label_agree / label_disagree",
         "adapter": "66 agree, 0 disagree (6 of the 72 runs have no non-empty state)", "match": True},
        {"what": "E4B live run over all 129 rows (not reported by the lab)", "value": "-",
         "where": "results/run-e4b/final.jsonl", "adapter": "11/129 before the drop; 15 context-window errors, 2 timeouts",
         "match": True},
    ],
}

AUDIT = r"^results/audit/verdicts\.jsonl$"
FINAL = r"^results/run-e4b/final\.jsonl$"


def _valid(src):
    gold, empty = {}, {}
    for _, rows in src.jsonl(AUDIT):
        for r in rows:
            {"gold": gold, "empty": empty}[r["kind"]][r["iid"]] = r["resolved"]
    return {i for i, ok in gold.items() if ok is True and empty.get(i) is False}


def attempts(src):
    valid = _valid(src) if DROP_AUDIT_INVALID else None
    for path, rows in src.jsonl(FINAL):
        for row in rows:
            if valid is not None and row["iid"] not in valid:
                continue  # counted in CARD["dropped"]
            resolved = row["resolved"] is True
            error = row.get("error")
            yield {
                "file": path,
                "system": "e4b-sample-agent",
                "run": "e4b-s1",
                "task": row["iid"],
                "y": int(resolved),
                "model": "E4B",
                "harness": "official",
                "err": bool(error),
                "status": error or ("resolved" if resolved else "failed"),
            }
