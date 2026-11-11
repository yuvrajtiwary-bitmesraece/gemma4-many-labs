"""Adapter for Strider221B/gemma4_dev_agent ("SweGemma-Agent", a post-training and submission pipeline; no results)."""

REPO = "Strider221B/gemma4_dev_agent"

CARD = {
    "eligible": False,
    "reason": "no per-task outcomes: the only data file with task ids (data/files.csv, 524 rows, columns name,size) "
              "is the listing of the competition's own files (snapshots, graphs, embeddings, wheels), and no run "
              "result of any agent is published",
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md "
                                                  "s9 'This repository is licensed under the [Apache License "
                                                  "2.0](LICENSE)'"},
    "models": [],
    "harness": None,
    "summary": "Code and design documents for a planned pipeline (synthetic trajectories, QLoRA SFT, GRPO/DPO, "
               "repository-grouped cross-validation, submission packaging) around gemma-4-31b-it-qat-w4a16-ct. "
               "The repository holds code, tests, plans and competition documentation, but no evaluation output.",
    "evidence": {
        "model": "README.md s1: 'adapts Google's Gemma 4 (`gemma-4-31b-it-qat-w4a16-ct`) into an autonomous "
                 "software engineering developer agent' (intent only; no run is recorded)",
        "harness": "none recorded; configs/eval_config.yaml describes a planned 4-fold CV with output_dir "
                   "'/kaggle/working/cv_results', which is not in the repository",
        "systems": "none",
        "replicates": "none",
        "task_sets": "notes.md: 'kaggle competitions files -c gemma-4-developer-agent --page-size 600 --format csv "
                     "> files.csv' is how data/files.csv was made: 129 snapshots/<id>.tgz, 127 graphs, 127 "
                     "embeddings, 124 wheels, 10 sample_submission files, tasks.jsonl, docker files",
    },
    "decisions": [
        "data/files.csv is a directory listing, not results: its 196 loosely matching 'ids' are file names such as "
        "snapshots/fastapi_11194.tgz and embeddings/fastapi_<commit sha>.npz",
        "the three notebooks (documentation/sample_code/*.ipynb, notebooks/train_notebook.ipynb) carry no cell "
        "outputs and no task ids; the four task ids in configs/eval_config.yaml are a 'quick_eval' list",
    ],
    "excluded": [
        "data/files.csv: listing of the competition data files (name, size), no outcomes",
        "data/HARNESS_README.md, documentation/competition_details/*: copies of competition documentation",
        "documentation/sample_code/high_scorers/*.ipynb: a third party's public notebook, stored without outputs",
        "tests/, src/data/mock_data_factory.py, src/evaluation/mock_sandbox.py: unit tests and mocks",
    ],
    "dropped": {},
    "flags": ["no_results"],
    "reported_numbers": [],
}


def attempts(src):
    return
    yield
