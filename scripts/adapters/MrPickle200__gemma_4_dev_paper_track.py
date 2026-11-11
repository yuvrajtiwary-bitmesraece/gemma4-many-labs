"""Adapter for MrPickle200/gemma-4-dev-paper-track. Not eligible: SWE-bench Verified tasks only."""

REPO = "MrPickle200/gemma-4-dev-paper-track"

CARD = {
    "eligible": False,
    "reason": "no development-set attempts: every run is on a 50-task subset of SWE-bench Verified (sympy 29, "
              "scikit-learn 13, pytest 5, psf/requests 3; ids such as sympy__sympy-15599, psf__requests-1724) scored "
              "with the SWE-bench Docker harness; the three psf__requests-<n> ids are SWE-bench instances, not "
              "competition tasks, and no competition task id or competition harness appears anywhere in the repository",
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'"},
    "models": [],
    "harness": None,
    "summary": "Micro-model harness ablation (arms B0, M1-M4, M4-SFT) for Gemma 4 E2B and E4B (Q4_0 GGUF on llama.cpp, "
               "RTX 4050 6 GB) on 'SWE-bench Verified Mini (N=50)'.",
    "evidence": {
        "model": "README.md: 'Models-Gemma 4 E2B | E4B (QAT Q4_0)'; run files "
                 "experiments/<date>-<name>/runs/<instance>__gemma-4-E2B|E4B[-M4-sft]__<arm>__<timestamp>.jsonl",
        "harness": "README.md s3: 'All 500+ cells were evaluated end-to-end inside isolated Docker containers using the "
                   "official SWE-bench Verified harness' (scripts/run_swebench_eval.py)",
        "systems": "README.md s2 'Harness Ablation Hierarchy': B0, M1, M2, M3, M4, M4-SFT, each for E2B and E4B",
        "replicates": "none claimed: PROGRESS.md 'With n=50, one seed, one run per cell'; several traces per cell exist "
                      "because infrastructure failures were re-run ('one trace per cell, newest wins, infra failures out')",
        "task_sets": "PROGRESS.md: 'Loaded `princeton-nlp/SWE-bench_Verified` (500 instances, split `test`) ... repo in "
                     "{sympy/sympy, scikit-learn/scikit-learn, pytest-dev/pytest, psf/requests}, gold patch touches "
                     "exactly 1 file, patch < 50 changed lines ... 117 of 500 instances pass; sampled 50 with `seed=0`'; "
                     "data/subset_50.json",
    },
    "decisions": [
        "all 1,986 run/patch file names were parsed: 50 distinct instance ids, all in SWE-bench form "
        "(<owner>__<repo>-<number>); none is a competition development-set id",
        "nothing is yielded; SWE-bench ids are not rewritten into the competition id scheme",
    ],
    "excluded": [
        "experiments/*/runs/*.jsonl (999 trace files), experiments/*/patches/*.patch, experiments/*/predictions/*.jsonl, "
        "experiments/*/reports/*.json, experiments/*/results.csv, experiments/*/swebench_results.csv: SWE-bench Verified "
        "tasks, outside the public development set. Resolved of 50 per arm (swebench_results.csv, same as README.md): "
        "E2B B0 3, M1 4, M2 5, M3 3, M4 4, M4-SFT 0; E4B B0 3, M1 6, M2 8, M3 8, M4 9, M4-SFT 0",
        "experiments/20261017-original-e2b, experiments/20261021-fixes-smoke-e2b: earlier and smoke runs on the same "
        "SWE-bench tasks, without resolved/not-resolved scoring",
        "data/subset_50.json, cache/: task definitions and retrieval caches for the SWE-bench subset",
        "tests/: unit tests of the harness",
    ],
    "dropped": {},
    "flags": ["swe_bench_only"],
    "reported_numbers": [
        {"what": "resolved of 50 SWE-bench Verified tasks per arm, E2B: B0, M1, M2, M3, M4, M4-SFT", "value": "3, 4, 5, 3, 4, 0",
         "where": "README.md s3; experiments/*/swebench_results.csv", "adapter": "not yielded (not development-set tasks)",
         "match": True},
        {"what": "resolved of 50 SWE-bench Verified tasks per arm, E4B: B0, M1, M2, M3, M4, M4-SFT", "value": "3, 6, 8, 8, 9, 0",
         "where": "README.md s3; experiments/*/swebench_results.csv", "adapter": "not yielded (not development-set tasks)",
         "match": True},
    ],
}


def attempts(src):
    return
    yield
