"""Adapter for yosh95/g4-developer-agent-paper ("Every Tool Call Is a Token You Cannot Buy Back")."""

REPO = "yosh95/g4-developer-agent-paper"

PAPER = "paper/writeup_draft.md"

CARD = {
    "eligible": True,
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE (682 bytes, short form, hence GitHub's NOASSERTION): 'Apache License "
                                                  "Version 2.0, January 2004 ... Licensed under the Apache License, Version 2.0 "
                                                  "(the \"License\")'; README.md: 'Apache-2.0 (see `LICENSE`).'"},
    "models": ["31B"],
    "harness": "official",
    "summary": "Five agent bundles (v5, v6_single, v8_anzcap, v8_noanz, v9_verify) run through the swegemma harness on a "
               "CPU-only host against a hosted Gemma 4 31B endpoint: four on an 18-task set, two on a 55-task set that "
               "contains the 18 (v8_noanz on both).",
    "evidence": {
        "model": "README.md: 'a single small coding agent (Gemma 4 31B, INT4 QAT)'; replay/local_eval.py: "
                 "'Point --base-url at https://ollama.com/v1 with --model gemma4:31b to use the free Ollama Cloud quota' "
                 "(the arm tags are '@ollama' and '@ollama55')",
        "harness": "replay/local_eval.py: 'Run the REAL swegemma harness locally against a submission bundle, using "
                   "OpenRouter's google/gemma-4-31b-it as the model backend instead of a local vLLM server' and 'each sandbox "
                   "is provisioned the same way replay.py does it (validated: oracle 18/18)'",
        "systems": "paper/task_outcomes.csv column `arm` = '<bundle>@<tag>'; replay/local_eval.py: '--arm ... bundle dir name', "
                   "'--tag ... suffix for the results dir, e.g. 'ollama'', results dir = arm + '@' + tag",
        "replicates": "README.md: 'task_outcomes.csv       per-task outcome for 5 configurations (182 rows)' while the file has "
                      "six arm labels; only 'v8_noanz@ollama' (18 tasks) and 'v8_noanz@ollama55' (55 tasks, the 18 among them) "
                      "share a bundle name; paper/spin_stats.json calls them 'v8_noanz@18' and 'v8_noanz@55'",
        "task_sets": PAPER + " section 2: '55 tasks from the competition's public task file (37 fastapi, 9 rich, 8 requests, "
                             "1 httpx) ... Tasks whose snapshot is not reachable offline are excluded.'; 3.3: 'On the same 18 "
                             "tasks: single agent 12/18, delegated analyzer 6/18'",
    },
    "decisions": [
        "system = bundle name (the part of `arm` before '@'); the tag names the endpoint and task set ('ollama' = 18 tasks, "
        "'ollama55' = 55 tasks); run = the full arm label",
        "v8_noanz has two runs (18 and 55 tasks; the 18 tasks are attempted in both): one system because the lab counts five "
        "configurations for six arm labels. There is no seed or repeat index, and whether the bundle's budget was identical in "
        "both runs is not documented (the 55-task run reports 'Agent exceeded turns budget (150 turns)', the 18-task run no "
        "budget error)",
        "harness = official: swegemma's Evaluator and verification scored every attempt, but off Kaggle - subprocess sandbox "
        "provisioned by the lab with unpinned test dependencies, model served by a hosted endpoint instead of the QAT build "
        "under vLLM; the reference-patch replay was checked on the 18-task set only",
        "y = `resolved`; `failure_mode` 'wrong_patch' and 'empty_patch' are plain failures (err False); any other value is a "
        "harness error string (turn budget, session timeout, unknown tool, test patch did not apply) -> err True, y = 0",
        "'Failed to apply test_patch' (3 rows) and 'Tool 'run_command' not found' (7 rows, v8_anzcap) stay failures, as the lab "
        "counts them (analyzer arm 6/18)",
    ],
    "excluded": [
        "paper/spin_stats.json: per-configuration tool-call statistics (aggregates)",
        "control, no per-task rows: " + PAPER + " section 2 'We verified the reconstruction by replaying the ground-truth patch on "
        "18 tasks: 18/18 resolve.'",
        "budget arms, aggregate only: " + PAPER + " 3.4 'Widening it (5 → 10 minutes, 100 → 200 calls) left accuracy unchanged "
        "across three arms (11, 11, 11 on the same 18 tasks)' - these arms are not in task_outcomes.csv",
        "hidden-test leaderboard: " + PAPER + " 3.8 '19/55 = 34.5 % of public tasks and 3/58 = 5.2 % of hidden ones'; 'an "
        "earlier configuration of ours reached k = 4 (0.06)'",
        "localisation metric: 'gold file in top-5 for 14/18 tasks, top-1 for 9/18' (not a resolved outcome)",
    ],
    "dropped": {},
    "flags": ["hosted_endpoint_model", "lab_provisioned_sandbox"],
    "reported_numbers": [
        {"what": "v9 on the 55 tasks", "value": "19/55 (34.5 %)", "where": "README.md; " + PAPER + " 3.8",
         "adapter": "v9_verify@ollama55 19/55", "match": True},
        {"what": "single agent vs delegated analyzer on the 18 tasks", "value": "12/18 vs 6/18 (McNemar exact p = 0.070)",
         "where": PAPER + " 3.3", "adapter": "v8_noanz@ollama 12/18, v8_anzcap@ollama 6/18 (discordant 7 vs 1)", "match": True},
        {"what": "accuracy before and after the stop-loss rule on the 18 tasks", "value": "11 -> 12", "where": PAPER + " 3.2",
         "adapter": "v5@ollama 11/18, v8_noanz@ollama 12/18", "match": True},
        {"what": "v9 per repository", "value": "requests 7/8, fastapi 10/37, rich 2/9", "where": PAPER + " 3.7",
         "adapter": "requests 7/8, fastapi 10/37, rich 2/9 (httpx 0/1)", "match": True},
        {"what": "v9 failures", "value": "8 of 36 budget exhausted, 13 zero-byte patches", "where": PAPER + " 3.4",
         "adapter": "36 failures, 8 with 'exceeded turns budget', 13 with patch_bytes 0", "match": True},
        {"what": "three budget arms on the 18 tasks", "value": "11, 11, 11",
         "where": PAPER + " 3.4; paper/SUBMITTED_2026-10-05.md says it was checked against task_outcomes.csv",
         "adapter": "arms on the 18 tasks give 11 (v5), 9 (v6_single), 6 (v8_anzcap), 12 (v8_noanz), and 9 / 10 for the two "
                    "55-task runs restricted to them; no three arms with 11", "match": False},
        {"what": "rows, configurations, tasks", "value": "182 rows, 5 configurations, 55 tasks", "where": "README.md",
         "adapter": "182 attempts, 5 systems (6 runs), 55 tasks", "match": True},
    ],
}

PLAIN_FAILURES = {"wrong_patch", "empty_patch"}


def attempts(src):
    for path, rows in src.csv(r"^paper/task_outcomes\.csv$"):
        for row in rows:
            arm = row["arm"].strip()
            resolved = row["resolved"].strip() == "True"
            mode = (row.get("failure_mode") or "").strip()
            yield {
                "file": path,
                "system": arm.split("@", 1)[0],
                "run": arm,
                "task": row["task_id"].strip(),
                "y": int(resolved),
                "model": "31B",
                "harness": "official",
                "err": (not resolved) and mode not in PLAIN_FAILURES,
                "status": " ".join(mode.split()) or ("resolved" if resolved else "failed"),
            }
