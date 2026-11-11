"""Adapter for HivemindEnt/hivemind-gemma4-dev-agent-pub ("Out of Time, Not Out of Ideas")."""
import re

REPO = "HivemindEnt/hivemind-gemma4-dev-agent-pub"

CARD = {
    "eligible": True,
    "licence": {"spdx": "MIT-0", "evidence": "LICENSE (MIT No Attribution); README.md 'Licence: MIT No Attribution (MIT-0)'"},
    "models": ["31B"],
    "harness": "official",
    "summary": "21 Kaggle runs of a YAML-only 31B agent: 14 arms, each changing one variable from baseline A0.",
    "evidence": {
        "model": "arms/A0/arm.json: \"model\": \"gemma-4-31b-it-qat-w4a16-ct\"; results/*/summary.json repeat it per run",
        "harness": "paper/paper.md s3: 'All runs use the host's starter stack: swegemma 0.2.11, vLLM 0.19.1 ... on Kaggle 4x L4'",
        "systems": "README.md: 'arms/<ARM>/ One agent configuration per experiment arm ... Each arm changes one variable from the baseline A0'",
        "replicates": "README.md: 'the baseline's three dev repeats are results/A0-r1, A0-r2 and A0-r3'; paper Table 2 lists A2 and A5 as 'two runs'",
        "task_sets": "README.md: 'r = 30-task dev subset, h = 47 held-out tasks, c = one-task-at-a-time check'",
    },
    "decisions": [
        "system = arm; runs of one arm on the 30-task subset (r) and on the held-out tasks (h) are the same system",
        "the one-task-at-a-time check (tag c) changes the execution condition (concurrency 1 instead of 4), so it is kept as a separate system '<arm>@serial'",
        "err = the harness reported a non-empty error string (session timeout or tool-call budget)",
    ],
    "reported_numbers": [
        {"what": "A0-r1, A0-r2, A0-r3 resolved", "value": "5, 4 and 7 of 30", "where": "README.md, paper abstract"},
        {"what": "A2 two runs", "value": "4/30, 4/30", "where": "paper Table 2"},
        {"what": "A5 two runs", "value": "7/30, 4/30", "where": "paper Table 2"},
    ],
}

RUN = re.compile(r"^results/([A-Z]\d+)-([rhc])(\d+)/summary\.jsonl$")


def attempts(src):
    for path, rows in src.jsonl(RUN.pattern):
        arm, tag, k = RUN.match(path).groups()
        system = arm if tag in "rh" else f"{arm}@serial"
        for row in rows:
            yield {
                "file": path,
                "system": system,
                "run": f"{arm}-{tag}{k}",
                "task": row["instance_id"],
                "y": int(bool(row["resolved"])),
                "model": "31B",
                "harness": "official",
                "err": bool(row.get("error")),
                "status": row.get("error") or ("resolved" if row["resolved"] else "failed"),
            }
