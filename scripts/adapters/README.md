# Source adapters

One file per candidate source. An adapter reads a pinned repository snapshot
(never unpacked, never executed) and yields one record per **attempt**: one
execution of a Gemma 4 agent on one public development-set task with a binary
verified outcome. The rules below restate `ANALYSIS_PLAN.md` sections 3-5 in
operational form; the plan wins if the two ever disagree.

`HivemindEnt__hivemind_gemma4_dev_agent_pub.py` is the reference example.

## File and interface

File name: `scripts/adapters/<owner>__<repo>.py` with every character outside
`[A-Za-z0-9_]` replaced by `_`.

```python
REPO = "owner/repo"          # exactly as in data/sources/manifest.json

CARD = {
    "eligible": True,        # False: no usable records; then give "reason"
    "reason": "",            # why not eligible (only when eligible is False)
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0'"},
    "models": ["31B"],       # Gemma 4 sizes that occur in the yielded records
    "harness": "official",   # "official", "local", "mixed" or None (unknown)
    "summary": "one or two sentences: what the lab ran",
    "evidence": {            # where each decision comes from: path + short quote
        "model": "...", "harness": "...", "systems": "...", "replicates": "...", "task_sets": "...",
    },
    "decisions": ["every judgement call, one line each"],
    "excluded": ["path or family: reason"],          # what was left out and why
    "dropped": {"not run / infrastructure": 0},       # counts of attempts dropped inside the adapter
    "flags": [],             # e.g. "oracle_hints", "fine_tuned", "aggregated_only", "partial_runs"
    "reported_numbers": [    # numbers the source itself states, used to verify the adapter
        {"what": "A0-r1 resolved", "value": "5/30", "where": "README.md", "adapter": "5/30", "match": True},
    ],
}

def attempts(src):           # src is a srcio.Source
    yield {
        "file": "path/in/repo",      # provenance, required
        "system": "label",           # agent configuration, required
        "run": "label",              # one execution, unique within the lab, required
        "task": "fastapi_14372",     # task id as the source writes it, required
        "y": 1,                      # 1 resolved, 0 not resolved, required
        "model": "31B",              # E2B | E4B | 12B | 26B-A4B | 31B | None
        "harness": "official",       # official | local | None
        "err": False,                # True/False/None, see below
        "status": "short raw status",
    }
```

Records must be yielded in file order. Use only the standard library and
`srcio`. Check an adapter with

    py -3.12 -I scripts/harmonize.py --check owner/repo [--runs]

which must report `problems=0`.

## Rules

**Licence (source level).** A source is used only if it carries an explicit
licence that permits reuse (Apache-2.0, MIT, MIT-0, BSD, CC-BY, CC0 ...),
stated in a licence file or the README. Quote the evidence. If there is none,
set `eligible: False` with reason `"no licence"` and yield nothing.

**Eligible attempt.** An agent built on a Gemma 4 model (any size, with or
without LoRA / fine-tuning), run on a task of the competition's public
development set (129 tasks, ids `<repo>_<number>` with repo in fastapi, rich,
requests, httpx), with a binary verified outcome scored by the competition
harness or by a local re-implementation the source documents.

**Exclude**

- tasks outside the public development set (SWE-bench, SWE-rebench, QuixBugs,
  synthetic or mutated tasks, private tasks). Ids that do not look like
  `<repo>_<number>` are dropped by the harmonizer; do not rewrite other id
  schemes into this form;
- runs of non-Gemma-4 models (teacher models, Gemma 3, API models) and runs
  that apply the reference ("gold") patch, an empty patch, or any other
  non-agent control. Mention such control runs in `excluded`;
- test fixtures, demos, examples, schema samples;
- metrics that are not a per-task resolved/not-resolved outcome (localisation
  hits, retrieval ranks, partial test counts, patch-applies flags);
- hidden-test / leaderboard results;
- derived or merged tables that repeat runs already yielded from another file.
  Use the rawest per-run file and yield every run once.

**Outcome.** `y = 1` only if the source reports the task as resolved. Timeouts,
exhausted budgets, agent errors, empty patches and failed tests are `y = 0`,
because the competition scores them as failures. Attempts the source marks as
not run, skipped, or invalid for infrastructure reasons (sandbox build failed,
server crashed, quota ended) are **dropped** when the source makes that
distinction; count them in `CARD["dropped"]`.

**err.** `True` if the source attaches any error / timeout / budget-exhausted
flag to the attempt, `False` if it has such a field and it is empty, `None`
if the source has no such field.

**System.** One agent configuration within a lab, as the lab labels it (arm,
variant, version, config name). Different prompts, tools, budgets, sampling
settings, adapters or model sizes are different systems.

**Run.** The attempts one system produced in one execution. Shards of one
execution share one run label. Copies of the same run in two directories get
the same run label (the harmonizer keeps the first copy).

**Replicates.** Two runs get the same `system` label only if the lab itself
identifies them as repeats of one configuration: a seed or repeat index in the
run name or a column, or a statement in the documentation. Quote the evidence
in `CARD["evidence"]["replicates"]`. If it is unclear, give the runs different
system labels. Never infer replicates from similar scores.

**Model / harness.** Record what the source states. "official" means the
competition's own harness (swegemma / adk-eval, usually `summary.jsonl` or
`task_results.jsonl` with `instance_id`, `resolved`, `test_exit_code`) scored
the attempt; "local" means the lab's own grader. Unknown stays `None`.

**Runs with special conditions** (oracle hints such as the gold file being
given, fine-tuned adapters, partial or interrupted runs) are kept, the
condition goes into the system label and into `CARD["flags"]`.

**Aggregates only.** If a source gives only per-task counts over several
repeats (for example "3 of 6 solved") and no per-run rows, do not invent runs:
set the flag `aggregated_only` and describe the table in `excluded`.
