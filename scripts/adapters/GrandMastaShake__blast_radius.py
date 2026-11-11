"""Adapter for GrandMastaShake/blast-radius ("Blast Radius" and "The Edit Gate", two paper-track entries).

Not eligible: the per-task tables are edit-set prediction and test-selection metrics of graph walks (no model in
the loop).  The only outcome-like field is an aggregated "ever resolved" flag over 182 agent runs whose per-run
rows are not in the repository.
"""

REPO = "GrandMastaShake/blast-radius"

CARD = {
    "eligible": False,
    "reason": ("no per-run agent outcomes: the data are file/test retrieval metrics (recall@k, full coverage, hit@k) of "
               "a code-graph walk versus keyword and BM25 baselines; the single outcome-like column, "
               "ever_resolved_by_ember in data/ripple_129.csv, is an aggregate ('resolved in any arm of any campaign', "
               "182 runs) without per-run rows, and the model behind that agent is not stated here"),
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License Apache License 2.0 "
                            "(see `LICENSE`). Gold-file labels derive from Himanshu Kumar Jha's Apache 2.0 dataset'"},
    "models": [],
    "harness": None,
    "summary": ("CPU-only benchmark on the 129 public tasks of (a) 'Blast Radius', a forward graph walk from issue text to "
                "the files a fix touches, against keyword search and file BM25 (a negative result: keyword search wins "
                "recall@8 68.4 vs 54.4), and (b) 'The Edit Gate', a reverse walk from an edit to the test files at risk. "
                "No agent was run for these tables."),
    "evidence": {
        "model": "none in the tables; scripts/blast_radius.py and scripts/edit_gate.py are 'Standard library only' graph "
                 "walks. 'Ember' is the author's agent, mentioned only as the origin of the keyword baseline ('Keyword "
                 "search (the agent's grep)') and of the aggregate flag; its model is not named in this repository",
        "harness": "none: scores are recall against gold files / test_patch files (README.md 'Edit-set prediction on the 129 "
                   "public tasks, percent'; 'Scored on the 129 public tasks against the test files each task's test_patch touches')",
        "systems": "none yielded",
        "replicates": "not applicable. data/bench/ ('Run 2, the one in the paper') and data/bench/run1_seeds_v1/ ('Run 1, with "
                      "the first seed rules') are two versions of a deterministic retrieval benchmark, not agent repeats",
        "task_sets": "scripts/make_ripple_table.py: V5_25 and V6_33 list the 25 and 33 tasks of the author's earlier agent "
                     "campaigns (37 distinct); '# Tasks marked resolved in any arm of any campaign (campaign_rows.csv, 182 "
                     "runs).' EVER_RESOLVED = 5 task ids; campaign_rows.csv is not in the repository",
    },
    "decisions": [
        "ever_resolved_by_ember is not converted into attempts: it pools an unknown number of runs and arms per task (aggregated_only), and 21 of the 37 campaign tasks were in both campaigns",
        "blast:full@8, keyword:full@8, gate:hit@k etc. are 0/1 per task but measure retrieval coverage, not a resolved outcome",
    ],
    "excluded": [
        "data/ripple_129.csv: task metadata (gold files, ripple kind) plus campaign membership (in_v5_25, in_v6_33) and the aggregate ever_resolved_by_ember: 5 of the 37 campaign tasks (fastapi_14851, fastapi_15589, requests_6644, rich_3718, rich_4077) were resolved in at least one of 182 agent runs; no per-run rows",
        "data/bench/bench_per_task.csv, data/bench/run1_seeds_v1/bench_per_task.csv, bench_summary.json, report.md, run.log: edit-set prediction metrics (recall@3, recall@8, full coverage@8, ripple recall@8) for keyword, BM25 and graph-walk rankers; no model",
        "data/gate/gate_per_task.csv, gate_summary.json, report.md: test-file selection metrics (hit@k, recall@k) for the edit gate, naming and grep baselines and decoy edits; no model",
        "data/gate/shipped_log.csv, shipped_summary.json, shipped_by_repo.txt: the same test-selection study on the competition's shipped graphs and embeddings (Kaggle CPU notebook)",
        "data/jha_tasks_index.json, data/jha_bench_summary.json: third-party gold-file index and retrieval baselines (HIMANSHUKUMARJHA/gemma4-graph-retrieval-benchmark, Apache 2.0)",
        "scripts/smoke_issues.json: 'Stand-in issue texts for five rippling tasks, written for smoke tests. Not competition text, not paper numbers.' (test fixture)",
    ],
    "dropped": {},
    "flags": ["aggregated_only", "no_per_task_outcomes"],
    "reported_numbers": [
        {"what": "agent runs with per-task outcomes in this repository", "value": "0 (retrieval metrics only; 182 campaign runs are cited but not included)",
         "where": "README.md; scripts/make_ripple_table.py", "adapter": "0 attempts", "match": True},
    ],
}


def attempts(src):
    return
    yield
