"""Adapter for sumitmehta1025/given-a-map ("Given a Map, the Agent Doesn't Look")."""

REPO = "sumitmehta1025/given-a-map"

PAPER = "paper/paper.md"

CARD = {
    "eligible": True,
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: "
                                                  "'## License' / 'Apache 2.0.'"},
    "models": ["31B"],
    "harness": "official",
    "summary": "Pre-registered ablation of the harness's code-graph tools: 4 agent setups (no graph tools, tools available, "
               "official sample prompt, tools required) x 26 dev tasks checked solvable x 3 runs = 312 attempts on Kaggle 4x L4.",
    "evidence": {
        "model": "README.md: 'Harness: swegemma 0.2.11. Model: `gemma-4-31b-it-qat-w4a16-ct`, temperature 0.2. Budget per "
                 "attempt: 4 minutes and 100 tool calls.'",
        "harness": PAPER + " 3.1: 'We use the competition's harness (`swegemma` 0.2.11), unchanged, with its `subprocess` "
                           "sandbox, on Kaggle's 4× L4 machines.'",
        "systems": PAPER + " 3.3 table: 'A: grep only', 'B: grep + graph', 'C: official sample', 'D: graph required'; README.md: "
                           "'`setups/*.diff` | The 4 agent setups, as diffs against the official sample agent'",
        "replicates": PAPER + " 3.4: 'Each setup ran all 26 tasks three times, one task at a time, with setups alternating in "
                              "each run.'; analysis/output/runs1-3/attempts.csv column `run` (1, 2, 3); README.md: 'We ran 4 "
                              "setups × 26 tasks × 3 runs, for 312 attempts.'",
        "task_sets": PAPER + " 3.2: 'We kept only solvable tasks and picked 13 multi-file and 13 single-file fixes (FastAPI 13, "
                             "Rich 13), stratified by repository, with a fixed seed. ... The same 26 tasks are used for every "
                             "setup and run.'",
    },
    "decisions": [
        "system = setup, named after the lab's setup files: 'Grep only' = A_grep_only, 'Graph + grep' = B_grep_graph, "
        "'Graph + grep, prompted' = C_graph_prompted (paper: 'C: official sample ... the official sample, unchanged'; plan: "
        "'Exactly the official sample, with our 4-min / 100-call budget'), 'Graph required' = D_graph_required",
        "run = setup x run index (12 runs of 26 tasks); the run index is the lab's repeat number",
        "C's run 1 ran in a separate session after A/B's run 1 and D's three runs ran a week later ('on the same harness, "
        "model, tasks, budget and machine type'); they stay runs 1-3 of their setup, as in the lab's tables",
        "y = `resolved`; err = `hit_limit` is 'Yes' or `category` is 'Crash or tool error' (both columns are filled only for "
        "failed attempts)",
        "attempts that hung twice are recorded by the run script as errors and labelled 'Crash or tool error'; the released "
        "table does not separate them from other crashes and the lab counts them as failures, so nothing is dropped",
        "paper 3.1: the model is 'served by vLLM with the sample's LoRA adapters' in every setup; these are the official "
        "sample's adapters, not a lab fine-tune, so no fine_tuned flag",
    ],
    "excluded": [
        "analysis/output/runs1-3/sheet_h1_h2.csv: the same 312 attempts in another layout (derived from attempts.csv)",
        "analysis/output/runs1-3/sheet_h3.csv, sheet_h4.csv: failure categories and graph queries, no outcomes",
        "analysis/output/week5/per_task.csv, resolution.csv, results.md: aggregates built from attempts.csv",
        "analysis/output/task_features.csv: control, not agent runs - per-task features and exclusion reasons for all 129 tasks; "
        "54 are marked 'not solvable in the sandbox' (paper 3.2: '54 of the 104 remaining tasks cannot pass in the sandbox even "
        "with the reference fix, mostly because of missing test dependencies. Requests and HTTPX had no solvable tasks.')",
        "pilot: experiment-plan change log Oct 21: run 1's 'first 8 attempts (old list) archived in "
        "`runs/2026-10-20-pilot-old30/` and not used' - not in the repository",
    ],
    "dropped": {},
    "flags": [],
    "reported_numbers": [
        {"what": "A: grep only, runs 1-3", "value": "8, 11, 10 of 26 (29/78)", "where": PAPER + " Table 1",
         "adapter": "8, 11, 10 of 26", "match": True},
        {"what": "B: grep + graph, runs 1-3", "value": "10, 9, 10 of 26 (29/78)", "where": PAPER + " Table 1",
         "adapter": "10, 9, 10 of 26", "match": True},
        {"what": "C: official sample, runs 1-3", "value": "7, 8, 11 of 26 (26/78)", "where": PAPER + " Table 1",
         "adapter": "7, 8, 11 of 26", "match": True},
        {"what": "D: graph required, runs 1-3", "value": "9, 7, 11 of 26 (27/78)", "where": PAPER + " Table 1",
         "adapter": "9, 7, 11 of 26", "match": True},
        {"what": "attempts and failures", "value": "312 attempts, 201 failures", "where": PAPER + " abstract and 4.3",
         "adapter": "312 attempts, 201 with y = 0", "match": True},
    ],
}

SETUPS = {
    "Grep only": "A_grep_only",
    "Graph + grep": "B_grep_graph",
    "Graph + grep, prompted": "C_graph_prompted",
    "Graph required": "D_graph_required",
}


def attempts(src):
    for path, rows in src.csv(r"^analysis/output/runs1-3/attempts\.csv$"):
        for row in rows:
            system = SETUPS.get(row["setup"].strip(), row["setup"].strip())
            resolved = row["resolved"].strip() == "1"
            category = (row.get("category") or "").strip()
            limit = (row.get("hit_limit") or "").strip() == "Yes"
            yield {
                "file": path,
                "system": system,
                "run": f"{system}-run{row['run'].strip()}",
                "task": row["task"].strip(),
                "y": int(resolved),
                "model": "31B",
                "harness": "official",
                "err": limit or category == "Crash or tool error",
                "status": "resolved" if resolved else (category or "failed") + ("; hit limit" if limit else ""),
            }
