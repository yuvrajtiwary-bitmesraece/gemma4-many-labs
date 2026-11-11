"""Adapter for sigongjoa/gemma4-e2b-edit-failure ("Where a 2B Coding Agent Breaks")."""
import re

REPO = "sigongjoa/gemma4-e2b-edit-failure"

CARD = {
    "eligible": True,
    "reason": "",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License': "
                            "'Apache License 2.0 (`LICENSE`)'"},
    "models": ["E2B"],
    "harness": "official",
    "summary": "Gemma 4 E2B (QAT q4_0 GGUF on Ollama, one RTX 3050 8 GB) run with the competition's swegemma harness "
               "on a laptop: conditions A and D on 28 dev tasks, conditions B-K on 13 rich tasks (0 resolved in "
               "every 13-task run; 1/28 for A).",
    "evidence": {
        "model": "README.md: 'Gemma 4 E2B (QAT q4_0 GGUF, Ollama, one RTX 3050 8 GB) runs the competition's `swegemma` "
                 "harness'; agent_configs/condition_*/agent.yaml 'model: gemma-4-e2b-it-qat-w4a16-ct'; every trace step "
                 "has model_name 'hf.co/google/gemma-4-E2B-it-qat-q4_0-gguf:latest'",
        "harness": "scripts/run_eval.py: 'Runs swegemma with sandboxes that mimic the official Docker setup' "
                   "(`from swegemma.cli import main`, --sandbox subprocess); RESULTS.md: 'Resolved = hidden tests pass "
                   "in our local sandbox'; runs/*/task_results.jsonl carry instance_id / resolved / test_exit_code",
        "systems": "RESULTS.md '## Conditions' table (A, D, B, C, E, F, F2, G, H, I, J, K with the runs of each); "
                   "scripts/skill_runs.sh: 'cp -r $SRC/agent_configs/condition_$COND run/sub'",
        "replicates": "scripts/skill_runs.sh: 'runs PLAN entries \"COND:REP\" in order on the eval split, same limits as "
                      "pair.sh' with 'OUT=run/sk_${COND}_r$REP'; RESULTS.md Conditions table: 'D | ... | `pair_D` (28), "
                      "`sk_D_r2`, `sk_D_r3`', 'H | ... | `sk_H_r1`-`r3`', 'K | ... | `sk_K_r1`, `sk_K_r3`'; result "
                      "table rows 'D (2 runs)', 'H (3 runs)', 'K (2 runs)'; 'below H's run-to-run spread (11 points)'",
        "task_sets": "RESULTS.md: '52 public tasks sampled with seed 42; 28 whose reference patch passes in our sandbox "
                     "(rich 19, requests 5, fastapi 3, httpx 1)'; '`rich` tasks created after 2024-08-26 (13, "
                     "`scripts/eval13.txt`) are the evaluation set for all configuration comparisons'",
    },
    "decisions": [
        "outcomes are read from runs/<run>/task_results.jsonl (one per run, written by swegemma); the 253 "
        "runs/<run>/traces/trace_<task>.json files describe the same 253 attempts and are not read",
        "system = condition letter; run = run directory name",
        "pair_D (28 tasks) and sk_D_r2 / sk_D_r3 (13 of the same tasks) are one system 'D': RESULTS.md lists the three "
        "as the runs of condition D, the sk_ repeat index starts at 2, and the system prompt recorded in the traces is "
        "byte-identical in the three runs",
        "sk_B_r2 stopped after 11 of 13 tasks (no rows for rich_3470 and rich_3471; RESULTS.md's 13-task table reports "
        "only one B run): kept as an interrupted run under its own system label 'B@partial' (flag partial_runs)",
        "sk_K_r2 is dropped: README.md 'excluded from the paper (9 of 13 tasks timed out while the machine was short of "
        "memory)'; 7 of its rows have 0 LLM calls",
        "single-task trials (sk_*_rsmoke, one task each) are yielded under separate systems '<COND>@smoke': the system "
        "prompt in sk_B_rsmoke differs from sk_B_r1/r2, so they are not treated as repeats; the harmonizer's "
        "minimum-tasks rule removes them",
        "scripts/compare.py flags rich_3480 and rich_3471 in pair_D ('memory event 11-02 15:00, condition D only') but "
        "RESULTS.md reports D on all 28 tasks; both attempts are kept (y=0)",
        "harness = official because swegemma itself runs and scores each attempt, but in subprocess sandboxes on the "
        "lab's machine (not Docker / Kaggle) and with local limits of 7 minutes, 30 tool calls, 60 turns",
        "condition K replaces the built-in edit_file matcher (scripts/fuzzy_edit.py, 'local diagnostic only'); it is "
        "still a Gemma 4 agent scored by the harness and is kept as system 'K'",
        "err = non-empty harness error string (session timeout, tool-call budget, no submit_patch call)",
    ],
    "excluded": [
        "runs/*/traces/trace_<task>.json (253 files): trajectories of the attempts already yielded from task_results.jsonl",
        "runs/*/summary.json: per-run totals of the same task_results.jsonl",
        "runs/sk_K_r2: excluded by the lab (memory pressure), see dropped",
        "reference-patch control (scripts/run_eval.py GOLD=1): no per-task file published; RESULTS.md says the reference "
        "patch passes for 28 of the 52 sampled public tasks in the lab's sandbox, and only those 28 were used",
        "agent_configs/tools_graph, agent_configs/tools_search: configurations without published runs",
    ],
    "dropped": {"sk_K_r2: run invalidated by the lab (machine short of memory)": 13},
    "flags": ["partial_runs", "task_subset_where_reference_patch_passes_locally", "subprocess_sandbox_not_docker",
              "modified_builtin_tool_in_condition_K"],
    "reported_numbers": [
        {"what": "pair_A resolved on 28 tasks", "value": "1/28 (rich_3006)", "where": "RESULTS.md; runs/pair_A/summary.json",
         "adapter": "1/28 (rich_3006)", "match": True},
        {"what": "pair_D resolved on 28 tasks", "value": "0/28", "where": "RESULTS.md; runs/pair_D/summary.json",
         "adapter": "0/28", "match": True},
        {"what": "13 evaluation tasks, resolved per condition: D (2 runs), B, C, G, H (3 runs), I, J, F2, K (2 runs)",
         "value": "0 in every run", "where": "RESULTS.md '## 13 evaluation tasks'",
         "adapter": "0/13 in sk_D_r2, sk_D_r3, sk_B_r1, sk_C_r1, sk_G_r1, sk_H_r1-3, sk_I_r1, sk_J_r1, sk_F2_r1, "
                    "sk_K_r1, sk_K_r3", "match": True},
        {"what": "task-runs in the stage funnel (D x2, B, C, G, H x3, I, J on the 13 tasks)", "value": "130, resolved 0",
         "where": "RESULTS.md '## Stage funnel'", "adapter": "10 runs x 13 = 130, resolved 0", "match": True},
        {"what": "composition of the 28-task set", "value": "rich 19, requests 5, fastapi 3, httpx 1",
         "where": "RESULTS.md '## Task sets'", "adapter": "rich 19, requests 5, fastapi 3, httpx 1", "match": True},
        {"what": "13 evaluation tasks, patch / empty per run: D (2 runs), B, C, G, H (3 runs), I, J, F2, K (2 runs)",
         "value": "D 8/5, 10/3; B 11/2; C 11/2; G 10/3; H 11/2, 8/5, 11/2; I 9/4; J 8/5; F2 1/12; K 11/2, 11/2",
         "where": "RESULTS.md '## 13 evaluation tasks'",
         "adapter": "rows with agent_patch_size > 0 / == 0: sk_D_r2 8/5, sk_D_r3 10/3, sk_B_r1 11/2, sk_C_r1 11/2, "
                    "sk_G_r1 10/3, sk_H_r1 11/2, sk_H_r2 8/5, sk_H_r3 11/2, sk_I_r1 9/4, sk_J_r1 8/5, sk_F2_r1 1/12, "
                    "sk_K_r1 11/2, sk_K_r3 11/2 (this fixes which run directory is which table entry)", "match": True},
        {"what": "runs/*/summary.json total_tasks / resolved", "value": "28/1, 28/0, 13/0 (x15), 11/0 (sk_B_r2), 1/0 (x4)",
         "where": "runs/<run>/summary.json", "adapter": "same counts from task_results.jsonl in all 21 runs "
                                                        "(sk_K_r2 read but dropped)", "match": True},
    ],
}

RUN = re.compile(r"^runs/((pair|sk)_([A-Z]\d?)(?:_r(\w+))?)/task_results\.jsonl$")
LAB_EXCLUDED = {"sk_K_r2"}
PARTIAL = {"sk_B_r2"}


def attempts(src):
    for path, rows in src.jsonl(RUN.pattern):
        run, _, cond, rep = RUN.match(path).groups()
        if run in LAB_EXCLUDED:
            continue
        if rep == "smoke":
            system = f"{cond}@smoke"
        elif run in PARTIAL:
            system = f"{cond}@partial"
        else:
            system = cond
        for row in rows:
            error = row.get("error")
            resolved = bool(row["resolved"])
            yield {
                "file": path,
                "system": system,
                "run": run,
                "task": row["instance_id"],
                "y": int(resolved),
                "model": "E2B",
                "harness": "official",
                "err": bool(error),
                "status": error or ("resolved" if resolved else f"failed (test_exit_code {row.get('test_exit_code')})"),
            }
