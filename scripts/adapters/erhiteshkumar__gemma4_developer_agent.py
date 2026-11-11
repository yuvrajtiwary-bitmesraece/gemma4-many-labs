"""Adapter for erhiteshkumar/gemma4-developer-agent ("Rules in the tools, not the prompt: what limits Gemma 4 as a
coding agent under a fixed budget"); scaffolds and tooling, results only as totals."""

REPO = "erhiteshkumar/gemma4-developer-agent"

CARD = {
    "eligible": False,
    "reason": "aggregates only: run outputs are not published (README.md 'The repository holds no competition data: "
              "no task statements, patches, tests or agent traces'; 'run outputs are excluded by `.gitignore`'); "
              "local results exist only as per-run totals in docs/PAPER_DRAFT.md s5.1 and as narrative in "
              "docs/HARNESS_RUNBOOK.md",
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md "
                                                  "'## License Apache License 2.0 ([LICENSE](LICENSE))'"},
    "models": [],
    "harness": None,
    "summary": "About 30 YAML agent scaffolds (baseline, single agents with an explorer, staged analyst/fixer/"
               "verifier workflows, E4B variants) and the tooling to run the organizers' harness offline on Kaggle "
               "4x L4. Each candidate was run twice on the 26 gradeable TRAIN tasks (52 attempts); only the totals "
               "are public.",
    "evidence": {
        "model": "docs/PAPER_DRAFT.md s2: 'Gemma 4 31B instruction-tuned, quantised (gemma-4-31b-it-qat-w4a16-ct) "
                 "... E4B (gemma-4-e4b-it-qat-w4a16-ct)'",
        "harness": "docs/PAPER_DRAFT.md s2: 'Our harness runs the organizers' code offline on Kaggle's 4x L4 "
                   "notebooks with pinned wheels and checkpoints; a session runs each candidate twice on TRAIN (52 "
                   "attempts) and keeps the traces'; README.md 'swegemma 0.2.11'",
        "systems": "submissions/<name>/ (baseline, v2 ... v29-e4b); none has per-task results in the repository",
        "replicates": "stated only as totals: docs/PAPER_DRAFT.md s5.1 'the same unchanged submission resolved 26 "
                      "and then 30 of 52 in two sessions' (v16)",
        "task_sets": "configs/splits.json: fastapi TRAIN 23 / TUNE 12 / LOCKED-PUBLIC 32, rich 17 / 8 / 23, "
                     "requests 13 and httpx 1 REPO-TRANSFER; docs/PAPER_DRAFT.md s2: 'TRAIN split (26 tasks "
                     "gradeable in Kaggle's subprocess sandbox: 10 FastAPI, 16 rich)'",
    },
    "decisions": [
        "no per-run rows exist, so nothing is yielded (README rule 'Aggregates only'); the totals are listed in "
        "`excluded`",
        "docs/HARNESS_RUNBOOK.md names a few tasks per run in prose (e.g. smoke-001: 'Resolved 1/5 (rich_2725)'); "
        "these fragments are not complete per-run tables and are not transcribed",
    ],
    "excluded": [
        "docs/PAPER_DRAFT.md s5.1 table (public score | Kaggle L4 TRAIN resolved of 52 = 2 runs x 26 tasks): "
        "baseline 0.06 | -; v7 0.05-0.10 | 22; v15 0.08 | 23; v16 0.10 | 26 and 30; v20 0.15 | 27; v24 0.10 | 27; "
        "v26 timed out | 24; v27 - | 26; v28-e4b 0.05 | 8; text: v18 27, v29-e4b 4, v20-e4b 2 of 52",
        "docs/HARNESS_RUNBOOK.md: lab notebook with totals per session (e.g. five submissions on 26 tasks: baseline "
        "6, v2-focused 9, v3-focused 9, v2-explorer 12, v3-explorer 10; 144 collection attempts, 59 resolved)",
        "controls, docs/HARNESS_RUNBOOK.md: reference patch passes 18 of 40 TRAIN tasks in Docker with swegemma "
        "0.2.7 and 26 of 40 (10/23 FastAPI, 16/17 rich) in the subprocess sandbox; 'No patch: passed, all 40 TRAIN "
        "tasks fail'; 'rich_3468 also passes with an empty patch'",
        "configs/task_labels.json: three task labels (rich_3105, rich_3942 underspecified; fastapi_14479 "
        "statement_conflicts_with_tests), not outcomes",
        "tests/: CPU unit tests",
    ],
    "dropped": {},
    "flags": ["aggregated_only"],
    "reported_numbers": [],
}


def attempts(src):
    return
    yield
