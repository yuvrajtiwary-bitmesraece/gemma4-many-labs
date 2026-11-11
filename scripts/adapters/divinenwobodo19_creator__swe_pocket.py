"""Adapter for divinenwobodo19-creator/swe-pocket ("SWE-Pocket: A Size-Stratified Evaluation Resource for Local
Coding Agents").

Not eligible: a task-metadata resource.  No model or agent was run; every row is task metadata, call-graph
structure, test names or upstream provenance.
"""

REPO = "divinenwobodo19-creator/swe-pocket"

CARD = {
    "eligible": False,
    "reason": ("no agent outcomes: the source states 'No model was run. This is a resource and a structural analysis, "
               "not a leaderboard.' All per-task rows are task metadata (gold-patch size, test names, call-graph "
               "structure, upstream provenance)"),
    "licence": {
        "spdx": "Apache-2.0",
        "evidence": ("LICENSE: 'Apache License Version 2.0, January 2004' (code: scripts/, notebooks/, rebuild.sh); "
                     "LICENSE-DATA: 'The data files, figures and paper text in this repository (everything outside "
                     "scripts/, notebooks/ and rebuild.sh) are licensed under the Creative Commons Attribution 4.0 "
                     "International licence (CC-BY-4.0)'; README.md '## Licence - Code ...: Apache-2.0 ... - Data and "
                     "paper ...: CC-BY-4.0'. The data tables would be reused under CC-BY-4.0."),
    },
    "models": [],
    "harness": None,
    "summary": ("A size-stratified task resource: structural and gold-patch metrics for the 129 public competition "
                "tasks ('large' stratum) plus 24 tasks mined from four small libraries ('compact' stratum, ids "
                "packaging_*, prettytable_*, parso_*, python-dotenv_*). No model baselines."),
    "evidence": {
        "model": "none. README.md: 'No model was run.'; DATASHEET.md: 'What is not included: no model predictions, no "
                 "agent trajectories, no baseline resolve rates. Model baselines were deliberately out of scope'",
        "harness": "none. paper/swe-pocket.md s4.5: 'We make claims about structure, not agents ... (vii) We report no "
                   "model baselines'; s5: 'We report no agent accuracy and no failure taxonomy.'",
        "systems": "none",
        "replicates": "none",
        "task_sets": "README.md: '153 Python tasks across a large stratum (the 129 tasks released with the Gemma 4 "
                     "Developer Agent competition) and a compact stratum (24 tasks we mined from four small libraries)'",
    },
    "decisions": [
        "ineligible for lack of outcomes, not for licence: code is Apache-2.0 and data CC-BY-4.0, both permit reuse",
        "has_hints, f2p_count, p2p_count, gold_churn_* and the graph metrics are task descriptors (potential covariates), not outcomes",
    ],
    "excluded": [
        "data/task_metadata.csv, resources/task_metadata.json, data/joined_structure_gold.csv: one row per dev task with problem length, gold files, gold/test churn, F2P/P2P counts and graph metrics (task metadata)",
        "data/test_sets.csv, resources/test_sets.json: FAIL_TO_PASS / PASS_TO_PASS test names parsed from each test_patch (588 rows, 121 tasks with at least one name); a heuristic, 'the ground-truth pass/fail classification needs the harness run' (resources/test_sets.md)",
        "data/stratification.csv, resources/stratification.json, resources/strata_table.json, resources/size_vs_difficulty.json: call-graph size metrics per graph (127 graphs) and correlations with gold-patch size",
        "resources/provenance.csv, resources/provenance.json: upstream commit / pull-request provenance per graph",
        "resources/compact_tasks.json, data/compact_joined.*, data/compact_structure.csv, data/compact_graphs/*.json: the 24 mined compact tasks (outside the development set) and their graphs",
        "VALIDATION.md: reference (gold) patch replay of the 24 compact tasks only: 21/24 fail at base and pass with the gold patch; parso_227, parso_233, parso_241 already pass at base. Non-agent control on non-dev tasks",
    ],
    "dropped": {},
    "flags": ["no_per_task_outcomes", "task_metadata_only"],
    "reported_numbers": [
        {"what": "model / agent runs", "value": "0 ('No model was run')", "where": "README.md; DATASHEET.md; paper/swe-pocket.md s4.5, s5",
         "adapter": "0 attempts", "match": True},
    ],
}


def attempts(src):
    return
    yield
