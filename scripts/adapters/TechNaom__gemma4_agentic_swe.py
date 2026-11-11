"""Adapter for TechNaom/gemma4-agentic-swe ("Tools Unused, Tests Fooled: What actually helps a Gemma 4 coding
agent, and can we trust the score?")."""
import re

REPO = "TechNaom/gemma4-agentic-swe"

CARD = {
    "eligible": True,
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; "
                                                  "README.md '## License [Apache 2.0](LICENSE)'"},
    "models": ["31B"],
    "harness": "official",
    "summary": "A tool ablation on Kaggle 4x L4: 11 ADK agent configurations (k0-k5 with zero to three code-graph "
               "tools, n1-n5 the same with a 'use the tools' prompt) in two 10-task pilots and a 68-task comparison "
               "of k0, k4 and n4, restricted to the 68 dev tasks the lab's gold/empty audit found valid.",
    "evidence": {
        "model": "experiments/*/<run>/manifest.json: \"model\": \"gemma-4-31b-it-qat-w4a16-ct via local vLLM\"; "
                 "paper/paper.md s2: 'served with vLLM on a Kaggle notebook with 4x NVIDIA L4'",
        "harness": "paper/paper.md s2: 'The official `swegemma` harness and a Google ADK agent configuration ... "
                   "Per task the budget is 15 minutes, 60 tool calls and 200 turns'; src/g4swe/run_eval.py builds "
                   "swegemma.evaluate.Evaluator and writes one runs.jsonl row per task from its result",
        "systems": "paper/paper.md s4.1 table of configurations k0, k1-k3, k4, k5, n1-n5; run directories are named "
                   "<timestamp>_agent_<config>; experiments/g4swe-pilot2-202610240552/FINDINGS.md: 'k0 and k4 "
                   "re-run, plus nudge configs n1-n5 ... All with `max_output_tokens: 4096` (was 16384). Same 10 "
                   "pilot tasks, same model and budgets as pilot 1'",
        "replicates": "k0 (pilot 2 run and 68-task run): paper/paper.md s4.4 'The 10-task pilot suggested 80% for "
                      "k0; on all 68 valid tasks it is 50%. This was not luck in the pilot runs: in the full "
                      "ablation k0 solved the same 8 pilot tasks again'. k4 (pilot 2 run and 68-task run): same "
                      "configuration name in paper s4.2 and s4.3 ('k4 (all three, no nudge)', 'offering the tools "
                      "(k4 vs k0)'); scripts/build_notebook.py: 'nudge configs n1-n5, with k0/k4 re-run under the "
                      "same 4096-token output cap'. Pilot 1 is NOT a repeat of pilot 2 (output cap 16384 vs 4096)",
        "task_sets": "experiments/audit-2026-10-15/README.md: 'All agent experiments in this project use the 68 "
                     "valid tasks'; experiments/valid_tasks.txt (68), experiments/pilot_tasks.txt (10), "
                     "experiments/abl3_k4_tasks.txt (the 35 tasks k4 had not reached)",
    },
    "decisions": [
        "system = configuration name (k0 ... n5) for pilot 2 and the 68-task ablation, which share the committed "
        "agents/<config> (max_output_tokens 4096)",
        "pilot 1 (g4swe-pilot-202610231941) ran the k configurations with max_output_tokens 16384, a different "
        "sampling setting: separate systems '<config>@out16384', although FINDINGS.md of pilot 2 calls the k0 "
        "difference 'Run-to-run variation'",
        "experiments/first-agent-run-2026-10-24 (k0, 2 tasks, 30-minute budget in its manifest, started before "
        "pilot 1) is its own system 'k0@first-run'",
        "k4 on the 68 tasks was produced in two sessions: 33 tasks in abl1, then 'The other 35 k4 tasks: abl3' "
        "(SUMMARY.md of the full ablation). The lab reports them as one k4 row (68 tasks, 32 resolved); they are "
        "yielded as two shards of one run '20261024-204434+20261031-050847_agent_k4'",
        "n4 on the 68-task list 'stopped at 32 tasks when it reached a hard four-hour wall-clock cap' (paper s4.3): "
        "kept as system 'n4@partial', separate from the complete 10-task pilot-2 run of n4",
        "experiments/g4swe-ablation-full-20261031/<run> are byte-identical copies of the abl3 and abl4 run "
        "directories: yielded with the same run label, the harmonizer keeps the first copy",
        "run = run directory name (<timestamp>_agent_<config>)",
        "err = the harness `error` field is not null (session timeout, tool-call or turn budget, context overflow, "
        "no submit); status keeps the test exit code otherwise",
        "the manifest field \"seed\": 0 only seeds task sampling in run_eval.py and is the same everywhere; it is "
        "not a repeat index",
    ],
    "excluded": [
        "experiments/audit-2026-10-15/gold_runs.jsonl: reference-patch control, resolves 73/129 (fastapi 31/67, "
        "rich 39/48, requests 3/13, httpx 0/1); 34 of the 56 failures are missing test libraries "
        "(inline_snapshot 27, dirty_equals 7)",
        "experiments/audit-2026-10-15/empty_runs.jsonl: empty-patch control, resolves 5/129 (requests_7427, "
        "requests_7315, requests_7309, fastapi_14851, fastapi_14077)",
        "experiments/faults-2026-10-23/runs.jsonl, mutants.jsonl: 163 planted-bug mutants of gold patches "
        "(mode 'patch', no agent); 35 marked resolved, 24 judged genuine bugs by hand review",
        "experiments/*/SUMMARY.md, FINDINGS.md, notebook.log, abl*-notebook.log: tables and logs derived from the "
        "runs.jsonl files",
        "experiments/*/<run>/traces, patches, logs, test_outputs: per-task artefacts of the yielded runs, no "
        "additional outcomes",
        "demo/: toy demo, not a competition task",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["partial_runs"],
    "reported_numbers": [
        {"what": "68-task ablation: k0, k4, n4 resolved", "value": "34/68, 32/68, 14/32",
         "where": "paper/paper.md s4.3; experiments/g4swe-ablation-full-20261031/SUMMARY.md",
         "adapter": "34/68, 32/68 (16/33 + 16/35), 14/32", "match": True},
        {"what": "k4 vs k0 paired (68 tasks): both / only k0 / only k4 / neither", "value": "30 / 4 / 2 / 32",
         "where": "paper/paper.md s4.3", "adapter": "30 / 4 / 2 / 32", "match": True},
        {"what": "n4 vs k0 paired (32 tasks): both / only k0 / only n4 / neither", "value": "13 / 3 / 1 / 15",
         "where": "paper/paper.md s4.3", "adapter": "13 / 3 / 1 / 15", "match": True},
        {"what": "pilot 1: k0..k5 resolved of 10", "value": "7, 7, 6, 7, 6, 7",
         "where": "experiments/g4swe-pilot-202610231941/SUMMARY.md", "adapter": "7, 7, 6, 7, 6, 7", "match": True},
        {"what": "pilot 2: k0, k4, n1..n5 resolved of 10", "value": "8, 7, 6, 6, 6, 8, 6",
         "where": "paper/paper.md s4.2; experiments/g4swe-pilot2-202610240552/SUMMARY.md",
         "adapter": "8, 7, 6, 6, 6, 8, 6", "match": True},
        {"what": "k0 in the 68-task run on the 10 pilot tasks and on the other 58",
         "value": "'solved the same 8 pilot tasks again, but only 26 of the other 58'",
         "where": "paper/paper.md s4.4",
         "adapter": "8/10 and 26/58, but not the same 8: fastapi_14616 is solved only in pilot 2, rich_3944 only "
                    "in the 68-task run",
         "match": False},
        {"what": "first agent run (k0, 2 tasks)", "value": "2/2",
         "where": "experiments/first-agent-run-2026-10-24/README.md", "adapter": "2/2", "match": True},
    ],
}

RUN = re.compile(r"^experiments/(g4swe-[^/]+)/(\d{8}-\d{6})_agent_(\w+)/runs\.jsonl$")
FIRST = "experiments/first-agent-run-2026-10-24/runs.jsonl"
FILES = r"^experiments/(g4swe-[^/]+/\d{8}-\d{6}_agent_\w+|first-agent-run-2026-10-24)/runs\.jsonl$"
# k4 on the 68-task list: 33 tasks in abl1, the remaining 35 a week later in abl3.
K4_SHARDS = {"20261024-204434_agent_k4", "20261031-050847_agent_k4"}
K4_RUN = "20261024-204434+20261031-050847_agent_k4"
# n4 on the 68-task list stopped after 32 tasks (four-hour cap).
PARTIAL = {"20261031-102612_agent_n4"}


def attempts(src):
    for path, rows in src.jsonl(FILES):
        if path == FIRST:
            system, run = "k0@first-run", "20261023-193056_agent_k0"
        else:
            experiment, stamp, config = RUN.match(path).groups()
            run = f"{stamp}_agent_{config}"
            if experiment.startswith("g4swe-pilot-"):
                system = f"{config}@out16384"
            elif run in PARTIAL:
                system = f"{config}@partial"
            else:
                system = config
            if run in K4_SHARDS:
                run = K4_RUN
        for row in rows:
            if row.get("mode") != "agent":
                continue
            error = row.get("error")
            resolved = bool(row["resolved"])
            yield {
                "file": path,
                "system": system,
                "run": run,
                "task": row["instance_id"],
                "y": int(resolved),
                "model": "31B",
                "harness": "official",
                "err": bool(error),
                "status": error or ("resolved" if resolved else f"failed (test exit {row.get('test_exit_code')})"),
            }
