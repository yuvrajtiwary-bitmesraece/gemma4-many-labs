"""Adapter for Axentalan-VI/gemma-4-developer-agent
("Calibrate the Yardstick First: Measurement Discipline for Agentic SWE Benchmarks").

Not eligible, for two independent reasons (see CARD): the only licence statement
covers the writeup, not the repository's ledger, and the lab's own audit found that
its rig could not grade about three quarters of the fastapi rows - and every ledger
run is fastapi-only.  Nothing is yielded.  The repository matters as related work
(leaderboard granularity, gold/null calibration, power analysis), not as data.
"""

REPO = "Axentalan-VI/gemma-4-developer-agent"

CARD = {
    "eligible": False,
    "reason": (
        "no licence for the data: the repository has no LICENSE file and the README's only licence statement "
        "covers the writeup ('The writeup is released under CC BY 4.0'), not the ledger or code. Independently, the "
        "per-task `resolved` flags are not verified outcomes: all 13 ledger runs are fastapi-only and the lab "
        "reports that a starlette/fastapi mismatch in its own rig aborted 73% (clamp runs) and 76% (cap60 runs) of "
        "the test logs at collection ('three quarters of these tasks cannot be graded at all, whatever the patch "
        "says'); the affected rows cannot be identified because the pytest output is not committed"
    ),
    "licence": {
        "spdx": "NONE",
        "evidence": (
            "no LICENSE / COPYING file in the snapshot. README.md lines 134-139, in full: '## Licence and citation / "
            "The writeup is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you use the "
            "protocol, cite the writeup rather than this repository - it is the version of record.' The licence is "
            "granted for the writeup (paper/MANUSCRIPT.md, paper/SUBMISSION.md, the Kaggle write-up) and the "
            "sentence itself separates the writeup from 'this repository'; ledger/, bundles/, src/ and scripts/ "
            "carry no licence. EXPERIMENTS.md 'PAPER - submitted' table: 'licence | CC BY 4.0' (a field of the "
            "write-up); scripts/push_dataset.py stages the lab's own Kaggle dataset with "
            "'\"licenses\": [{\"name\": \"unknown\"}]'."
        ),
    },
    "models": [],
    "harness": None,
    "summary": (
        "Measurement-methodology study. 13 local Kaggle-GPU runs (388 rows) of gemma-4-31b-it-qat-w4a16-ct agent "
        "bundles (public baseline v00 and variants v03, v05, v06, v08) on 12, 36 or 60 fastapi tasks, used to show "
        "that binary resolution carried no information in the lab's own rig; plus gold/null calibration of all 129 "
        "tasks and an analysis of the public leaderboard's granularity."
    ),
    "evidence": {
        "model": "bundles/v00_public_baseline/agent.yaml: 'model: gemma-4-31b-it-qat-w4a16-ct' (ledger column "
                 "model_served is empty in all 388 rows)",
        "harness": "local re-run of the swegemma harness in the lab's Kaggle GPU kernels "
                   "(kernels/17_cap60_ab); EXPERIMENTS.md 'P009 RESULT': '26 of 34 `cap60_v00` test logs carry "
                   "`TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` - 76%. That is the "
                   "mechanical reason `resolved` is 1/36 with zero discordance: three quarters of these tasks cannot "
                   "be graded at all, whatever the patch says.'; T011 table: 'log carries the starlette mismatch | "
                   "34 / 46 | 41 / 57' (clamp_v00 / clamp_v08); paper/MANUSCRIPT.md Limitations: 'two tasks "
                   "produced - in both arms - patches whose change bodies were byte-identical to the reference "
                   "solution, and all four were scored unresolved'",
        "systems": "ledger columns variant + bundle_hash; README.md: 'Every row carries the `bundle_hash` of the "
                   "configuration that produced it ... repeated runs of one configuration pool instead of competing'",
        "replicates": "n/a (nothing yielded). For the record: v00_public_baseline (bundle 6f3c38bf...) has 8 runs, "
                      "but under different budgets (10/30/60-minute caps, 30/100 tool calls, concurrency 2/12); only "
                      "graph_v00 and clamp_v00 share task list (clean_dev.txt), cap (30 min) and seed 1001",
        "task_sets": "data/tasks/probe12.txt (12 fastapi), clean_dev36.txt (36 fastapi), clean_dev.txt (60 fastapi); "
                     "clean_n.txt = 102 measurable tasks (60 fastapi + 42 rich) after gold/null calibration",
    },
    "decisions": [
        "licence scope read literally: 'The writeup is released under CC BY 4.0' licenses the paper, and the "
        "analysis plan requires a licence for the source's data, so the ledger is not used",
        "even with a licence the rows would be unusable: the lab marks the fastapi verification in its rig as an "
        "environment fault (not an agent failure) for ~3/4 of rows in the clamp and cap60 runs, per run but not "
        "per row; the earlier graph runs resolve exactly the same three tasks (fastapi_14372, fastapi_14873, "
        "fastapi_15589), the pattern of the same fault; 10-12 rows per 60-task run have gold_exact=True and "
        "resolved=False",
        "gold_hit / gold_exact are localisation surrogates, not outcomes, and would not be used in any case",
    ],
    "excluded": [
        "ledger/runs/<run>/task_results.jsonl (13 files, 388 rows; all seed 1001, surrogate false, 31B, fastapi "
        "only) - resolved/n as recorded: probe_c12 2/12 and probe_c2 0/4 (v00, 30 calls, 10 min), paired100_v00 "
        "0/12, paired100_v03 0/12, resolve60_v00 0/12, analyzer_v00 0/12, analyzer_v05 0/12, graph_v00 3/60, "
        "graph_v06 3/60, clamp_v00 2/60, clamp_v08 3/60 (30-minute cap), cap60_v00 1/36, cap60_v08 1/36 (60-minute "
        "cap): not licensed, and not verified outcomes (see reason)",
        "ledger/runs/*/patches/*.patch (29 agent diffs), ledger/runs/*/runspec.json: no outcomes",
        "gold / null controls (kernels/01_gold_calibration, EXPERIMENTS.md rows C001-C004, no model): accepted "
        "run C004 - gold patch resolves 105/129, empty patch resolves 3/129, measurable set 102 "
        "(data/tasks/clean_n.txt lists the 102 ids; the 27 others are 7 fastapi, 13 requests, 6 rich, 1 httpx); "
        "first run C001/C003 gave gold 74/129, null 3/129. Later re-calibrations in the lab's rig: fastapi gold "
        "16/67 (P022, 2026-11-09) against 62/67 at C004; rich gold 39/48 (P011). Per-task verdicts are not "
        "committed, only the clean list",
        "data/leaderboard/public_lb_2026-10-13.json, public_lb_2026-10-19.json: public-leaderboard (hidden test) "
        "histograms; own scores v00_public_baseline 0.13, v01_hygiene 0.08, v04_lean 0.10 "
        "(bundles/*/MANIFEST.yaml, paper/MANUSCRIPT.md) - hidden-test results",
        "EXPERIMENTS.md runs P012-P020 (rich tasks, E4B / 31B 2x2, k=2 replications): described in prose only, "
        "their ledgers are not in this export ('Agent configurations that the paper does not cite' are omitted)",
    ],
    "dropped": {},
    "flags": ["no_licence_for_data", "harness_environment_fault"],
    "reported_numbers": [
        {"what": "cap60_v00 vs cap60_v08 resolved (36 paired fastapi tasks)", "value": "1/36 and 1/36, zero discordant pairs",
         "where": "README.md; paper/MANUSCRIPT.md; EXPERIMENTS.md 'P009 RESULT'",
         "adapter": "not yielded; ledger rows give 1/36 and 1/36, the same task (fastapi_14873) in both arms",
         "match": True},
        {"what": "session timeouts cap60_v00 vs cap60_v08", "value": "11 -> 0",
         "where": "README.md; paper/MANUSCRIPT.md",
         "adapter": "not yielded; ledger 'error' has 11 and 0 'Agent exceeded session timeout (60.0 min)' rows",
         "match": True},
        {"what": "gold_hit cap60_v00 vs cap60_v08", "value": "18 -> 24",
         "where": "paper/MANUSCRIPT.md 'What the protocol bought'",
         "adapter": "not yielded; ledger gold_hit sums are 18 and 24", "match": True},
    ],
}


def attempts(src):
    return
    yield
