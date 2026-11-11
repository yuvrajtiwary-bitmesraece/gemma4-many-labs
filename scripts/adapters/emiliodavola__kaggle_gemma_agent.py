"""Adapter for emiliodavola/kaggle-gemma-agent. Per-task outcomes exist only in markdown run analyses."""
import re

REPO = "emiliodavola/kaggle-gemma-agent"

CARD = {
    "eligible": True,
    "licence": {"spdx": "MIT", "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Emilio Dávola'"},
    "models": ["31B"],
    "harness": "official",
    "summary": ("Host trials of the lab's own declarative agent (12 skills, own system prompts legacy/v2/v3) with the swegemma "
                "harness on a Windows/Docker host and a local Gemma 4 31B QAT backend, budget 60 min / 100 tool calls per task. "
                "The raw runs/ directory is git-ignored; complete per-task outcomes are published only as markdown tables for "
                "run_01 (9 tasks), run_02 (13 tasks), run_03 (48 rich tasks) and a 2x2-rollout adapter check."),
    "evidence": {
        "model": ("docs/runs/20261006T103737Z/02-successes-vs-failures.md: 'backend `gemma-4-31b-it-qat`, `STATUS DONE`'; "
                  "odd/tasks/configurable-system-prompt.md: '`runs/20261004T225306Z` (backend `gemma-4-31b-it-qat`, 9 tasks, 3 "
                  "resolved)'; docs/journal/20261031.md: 'every run so far used the 31B'; docs/dev-experiment-loop.md: '--backend "
                  "lmstudio --model gemma-4-31b-it-qat'; docs/runs/20261026T152738Z/01-falsifier.md: 'the local 31B backend'"),
        "harness": ("README.md: 'The agent is packaged and evaluated by the `swegemma` harness; the scored metric is **Resolution "
                    "Rate** on a per-task budget of **60 minutes** and **100 tool calls**'; docs/README.md: host-trial-runbook = "
                    "'Operator runbook to run a real swegemma trial on Windows/Docker or Linux/WSL2'; "
                    "docs/proposals/20261026-harness2-adapter.md: 'swegemma 0.2.7, adk_submission 0.2.11, adk_eval_core 0.1.0'"),
        "systems": ("docs/journal/20261006.md J-20261006-03: 'run_03 mixed three prompt variants: v2 on 21 tasks, v3a on 11, v3 on "
                    "16 ... The shared checkout materialized the prompt while commits landed, so the batch is not one "
                    "configuration'; odd/tasks/host-trial-trace-metrics.md: 'run_01 = `runs/20261004T225306Z` (9 tasks, legacy "
                    "prompt, 3 resolved)', 'run_02 = `runs/20261005T050122Z` (13 tasks, v2 prompt, 1 resolved)'"),
        "replicates": ("none claimed: docs/journal/*.md header 'without randomisation a batch comparison is \"29/48 -> X/48 on a "
                       "different batch\", never \"it improved\"'; docs/runs/20261006T103737Z/01-audit.md: 'Rich overlap between "
                       "run_02 and run_03: 4 tasks ... With n=4 there is no signal'"),
        "task_sets": ("docs/runs/20261006T103737Z/02-successes-vs-failures.md: 'Universe: 48 tasks, all `Textualize/rich`. This "
                      "document is a census of 48, not a sample; the prompt phases (`v2` idx 1-21, `v3a` idx 22-32, `v3` idx "
                      "33-48) are temporal cuts, not randomization'"),
    },
    "decisions": [
        "only runs whose complete per-task outcome list is published are yielded; runs with aggregate counts or hand-picked "
        "example rows are listed in `excluded`",
        "run_03 is split into three runs/systems by prompt phase (v2, v3a, v3) because the prompt changed while the run was in "
        "progress; task assignment to a phase follows execution order, not randomisation",
        "run_02 (v2 prompt) and the v2 phase of run_03 are different systems: the sandbox test dependencies were repaired between "
        "them (journal J-20261005-05) and the lab never presents them as repeats",
        "y is the harness verdict; in run_03 three of the 29 resolved tasks (rich_3180, rich_3471, rich_3938) 'ship scratch only' "
        "(journal J-20261006-02), i.e. the task passed without a source change; they stay y=1",
        "run_01/run_02: attempts the lab marks as environment failures are dropped: fastapi rows tagged 'collection_error "
        "(infra)' in docs/runs/20261006T103737Z/05-segment-fastapi.md (missing module inline_snapshot, infra_error:true) and "
        "all requests_* attempts (07-segment-requests-httpx.md: 'requests failures are environmental (httpbin fixture + offline "
        "network)', 'resolved=false measures the harness, not the fix')",
        "run_03 err: True for the nine failures docs/runs/20261006T103737Z/06-segment-rich.md marks as timeout, budget or "
        "no-submit, False for the other ten failures, None for resolved tasks (no error column is published for them); "
        "rich_3296 and rich_3782 (collection_error, 'import to site-packages' in the superseded 01-audit.md) are kept as failures "
        "because the run report says `environment_blocked false`, `missing_modules []`",
        "the local backend is an LM Studio/llama.cpp build named gemma-4-31b-it-qat, not the scorer's W4A16 vLLM checkpoint; "
        "harness = official because swegemma itself ran and verified every task",
    ],
    "excluded": [
        "E4B baseline on Kaggle, run 20261101T021024Z (docs/journal/20261101.md): 13/129 resolved (0.101), 55 test failures, 47 "
        "collection errors of which 24 are missing modules in the notebook image; a second E4B run 10/129; only five hand-picked "
        "per-task rows are published (docs/runs/20261101T213823Z/01-pilot.md, column 'E4B baseline'): aggregated only",
        "teacher runs with deepseek-v4.1-flash (non-Gemma): pilot 3/5 (docs/runs/20261101T213823Z/01-pilot.md), batch 32/100 "
        "(docs/runs/20261102T214038Z/01-batch.md: rich 24/37, requests 2/10, fastapi 6/52), 20-task sample 6/20",
        "requests-only host trials 20261008T025904Z (2/13), 20261009T122847Z (1/13), 20261026T122756Z (1/13): counts in "
        "docs/journal/events.jsonl, five selected rows in docs/proposals/20261026-harness2-adapter.md: aggregated only",
        "Kaggle 31B runs 20261027T140509Z (11 tasks, 5 resolved) and 20261028T031307Z (11 tasks, 4 resolved) "
        "(odd/tasks/e4b-baseline-run.md): aggregated only",
        "29-task hold-out runs (docs/journal/events.jsonl): 20261107T202830Z 8/29, 20261108T100239Z 8/29, 20261109T124008Z 7/29; "
        "LoRA adapter run 20261110T130026Z 1/29 against base E4B 8/29 (docs/journal/20261110.md); two runs with 0/29 and "
        "unknown=29 (20261107T021631Z, 20261107T130105Z): aggregated only",
        "run 20261007T165050Z: one task (httpx_3672), not resolved, 'an import/collection failure hidden as infra_error:false' "
        "(docs/journal/20261007.md J-20261007-12): environment failure",
        "docs/tasks/*.md, experiments/subsets/*.json: task digests and task lists, no outcomes",
    ],
    "dropped": {"not run / infrastructure": 9},
    "flags": ["markdown_only", "config_changed_mid_run", "aggregated_only", "long_budget_60min_100calls"],
    "reported_numbers": [
        {"what": "run_03: tasks, resolved, rate", "value": "48, 29, 0.604",
         "where": "docs/runs/20261006T103737Z/02-successes-vs-failures.md s1.1; docs/journal/20261006.md",
         "adapter": "48, 29, 0.604 (sum of the three phase runs)", "match": True},
        {"what": "run_03 resolved by prompt phase v2 / v3a / v3", "value": "15/21, 7/11, 7/16",
         "where": "docs/journal/20261006.md J-20261006-03", "adapter": "15/21, 7/11, 7/16", "match": True},
        {"what": "run_03 process failures (timeout, call budget, no submit)", "value": "9 of 19 failures",
         "where": "docs/runs/20261006T103737Z/01-audit.md s0", "adapter": "9 failures with err=True", "match": True},
        {"what": "run_01: tasks, resolved", "value": "9, 3", "where": "odd/tasks/host-trial-trace-metrics.md",
         "adapter": "9 rows read, 3 resolved; 6 yielded after dropping 3 environment-blocked attempts", "match": True},
        {"what": "run_02: tasks, resolved", "value": "13, 1", "where": "odd/tasks/host-trial-trace-metrics.md; docs/journal/20261005.md",
         "adapter": "13 rows read, 1 resolved; 7 yielded after dropping 6 environment-blocked attempts", "match": True},
        {"what": "rich overlap run_02 -> run_03", "value": "3006 T->T, 3061 F->T, 3063 F->T, 3064 F->F",
         "where": "docs/runs/20261006T103737Z/01-audit.md s1", "adapter": "identical", "match": True},
        {"what": "adapter falsifier: resolved", "value": "requests_7315 on both trees, requests_6644 on neither",
         "where": "docs/runs/20261026T152738Z/01-falsifier.md", "adapter": "identical", "match": True},
    ],
}

EARLY = "odd/tasks/host-trial-trace-metrics.md"
EARLY_FASTAPI = "docs/runs/20261006T103737Z/05-segment-fastapi.md"
RUN03 = "docs/runs/20261006T103737Z/02-successes-vs-failures.md"
RUN03_FAILURES = "docs/runs/20261006T103737Z/06-segment-rich.md"
FALSIFIER = "docs/runs/20261026T152738Z/01-falsifier.md"
FILES = (EARLY, EARLY_FASTAPI, RUN03, RUN03_FAILURES, FALSIFIER)

TASK = r"(?:fastapi|rich|requests|httpx)_\d+"
EARLY_HEADER = re.compile(r"^(run_0[12]) = `runs/(\d{8}T\d{6}Z)`")
EARLY_ROW = re.compile(rf"^({TASK})\s+([Yn])\s+\d+")
EARLY_SYSTEM = {"run_01": "prompt-legacy@run_01", "run_02": "prompt-v2@run_02"}
FASTAPI_ROW = re.compile(r"^\|\s*(\d{4,5})\s*\|\s*(\d{8})\s*\|\s*(true|false)\s*\|\s*([^|]*?)\s*\|")
RUN03_ROW = re.compile(r"^\|\s*\d+\s*\|\s*(rich_\d+)\s*\|\s*(resolved|failed)\s*\|\s*([^|]*?)\s*\|\s*(v2|v3a|v3)\s*\|\s*$")
RUN03_FAILURE_ROW = re.compile(r"^\|\s*(\d{4})\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|")
FALSIFIER_ROW = re.compile(rf"^\|\s*`({TASK})`\s*\|\s*(base|variant)\s*\|\s*(yes|no)\s*\|\s*(yes|no)\s*\|")
FALSIFIER_SYSTEM = {"base": "falsifier-base", "variant": "falsifier-inv-edit"}


def _record(path, system, run, task, y, err, status):
    return {"file": path, "system": system, "run": run, "task": task, "y": y, "model": "31B", "harness": "official",
            "err": err, "status": status}


def attempts(src):
    pattern = "^(" + "|".join(re.escape(p) for p in FILES) + ")$"
    text = {path: raw.decode("utf-8", "replace").splitlines() for path, raw in src.read_many(pattern)}

    # run_01 and run_02: fixed-width listing; environment-blocked attempts are dropped (see CARD["decisions"])
    infra = set()
    for line in text[EARLY_FASTAPI]:
        m = FASTAPI_ROW.match(line.strip())
        if m and "(infra)" in m.group(4):
            infra.add((m.group(2), f"fastapi_{m.group(1)}"))
    label = run_id = None
    for line in text[EARLY]:
        h = EARLY_HEADER.match(line.strip())
        if h:
            label, run_id = h.groups()
            continue
        m = EARLY_ROW.match(line.strip())
        if not m or label is None:
            continue
        task, res = m.groups()
        if task.startswith("requests_") or (run_id[:8], task) in infra:
            continue
        y = int(res == "Y")
        yield _record(EARLY, EARLY_SYSTEM[label], run_id, task, y, None, "resolved" if y else "failed")

    # run_03: census table of section 1; error flags from the failure table of the rich segment report
    failure = {}
    for line in text[RUN03_FAILURES]:
        m = RUN03_FAILURE_ROW.match(line.strip())
        if m:
            kind, cause = m.group(2), m.group(3)
            process = bool(re.search(r"timeout|budget", kind, re.I)) or cause.lower().startswith("no-submit")
            failure[f"rich_{m.group(1)}"] = (kind, process)
    for line in text[RUN03]:
        m = RUN03_ROW.match(line.strip())
        if not m:
            continue
        task, verdict, kind, phase = m.groups()
        if verdict == "resolved":
            y, err, status = 1, None, "resolved"
        else:
            y = 0
            status, err = failure.get(task, (kind, None))
        yield _record(RUN03, f"prompt-{phase}@run_03", f"20261006T103737Z/{phase}", task, y, err, status)

    # harness2 adapter falsifier: two tasks on the base tree and on base plus one prompt edit
    for line in text[FALSIFIER]:
        m = FALSIFIER_ROW.match(line.strip())
        if not m:
            continue
        task, tree, delivered, resolved = m.groups()
        y = int(resolved == "yes")
        yield _record(FALSIFIER, FALSIFIER_SYSTEM[tree], f"20261026T152738Z/{tree}", task, y, None,
                      "resolved" if y else ("failed" if delivered == "yes" else "no patch delivered"))
