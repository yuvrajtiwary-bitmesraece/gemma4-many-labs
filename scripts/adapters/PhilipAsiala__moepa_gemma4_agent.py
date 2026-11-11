"""Adapter for PhilipAsiala/moepa-gemma4-agent ("Deterministic Cognitive Governance for Offline Software Engineering Agents: MOEPA")."""

REPO = "PhilipAsiala/moepa-gemma4-agent"

CARD = {
    "eligible": True,
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License': 'Copyright 2026 Philip Asiala · **Apache License 2.0**'; NOTICE"},
    "models": ["12B"],
    "harness": "local",
    "summary": ("Research probe harness (Ollama + local snapshots + pytest in python:3.13-slim) around a five-layer 'MOEPA' loop. "
                "Almost everything is localisation and gate measurement; the only agent attempts with a pass/fail outcome are "
                "three Gemma 4 12B tasks, all ending in NO_PATCH."),
    "evidence": {
        "model": "outputs/pytest_resolve.json: '\"model\": \"gemma4:12b\"'; paper/WRITEUP.md s3.1: '`gemma4:12b` (about 8 GB) on a local Ollama server'",
        "harness": ("outputs/pytest_resolve.json note: 'Local wheel-backed pytest resolve inside python:3.13-slim. Not the official "
                    "swegemma Docker verifier.'; paper/WRITEUP.md s3.1: 'Targeted tests are those added by each task's `test_patch`'"),
        "systems": ("paper/WRITEUP.md s3.2: 'a 3-task Praxeology attempt with the pytest gate enabled' "
                    "(scripts/pytest_resolve_probe.py probe_agent -> src/moepa/agent.py MOEPAGemmaAgent with OllamaClient)"),
        "replicates": "none: one execution of three tasks",
        "task_sets": ("scripts/pytest_resolve_probe.py: the agent probe takes the first --gemma-limit (3) tasks whose gold patch "
                      "resolved in the 20-task stratified gold probe; paper/WRITEUP.md: '0/3 patches on tasks where gold already resolves'"),
    },
    "decisions": [
        "the three agent_rows are yielded as one run of system 'moepa-12b-pytest-gate'; the agent abstained (error 'NO_PATCH', "
        "agent_patch false) on all three, which the competition scores as not resolved, so y=0 and err=True",
        "the run has 3 tasks and will be removed by the harmonizer's 10-task minimum; it is yielded because it is a real agent "
        "execution with a verified outcome",
        "harness = local: the lab's own pytest probe, not swegemma; the paper says 'we do not report a full resolution rate under "
        "the official swegemma harness'",
    ],
    "excluded": [
        "outputs/pytest_resolve.json gold_rows (control, gold patch, no model): local pytest resolves 16/20 (Rich 7/7, Requests "
        "6/6, FastAPI 3/7; four FastAPI tasks miss dependencies); all 20 tasks fail their targeted tests before the patch "
        "(fail_baseline 20/20)",
        "outputs/snapshot_probe.json (no model, with_gemma false): gold diff applies / parses / passes the Axiology policy gate on "
        "91 / 91 / 90 of 91 single-file tasks; apply and policy flags, not outcomes",
        "outputs/microbench.json localization_rows (129 tasks): lexical-seed and call-graph coverage of gold files and symbols "
        "(53/278 files, 70/364 hunk symbols); localisation metrics",
        "outputs/microbench.json gemma / gemma_26b and docs/microbench-results.md table: file-localisation hits of gemma4:12b "
        "(issue only 10/30, with neighbourhood 14/30) and gemma4:26b (1/12, 3/12); localisation hits, not resolved outcomes",
        "outputs/microbench.json axiology / metrology / embeddings: policy-gate checks on gold diffs (129/129 approved) and "
        "embedding statistics; no agent",
        "paper/WRITEUP.md 'the earlier syntax-only smoke was 1/3 Axiology-approved edits': no per-task file, no test outcome",
    ],
    "dropped": {},
    "flags": ["runs_under_10_tasks", "local_grader"],
    "reported_numbers": [
        {"what": "Praxeology + pytest gate (n=3)", "value": "0/3 patches (agent_resolve 0)", "where": "paper/WRITEUP.md s3.3 table; outputs/pytest_resolve.json",
         "adapter": "0/3 resolved, 3 NO_PATCH", "match": True},
        {"what": "local wheel pytest gold resolve (excluded control)", "value": "16/20, fail-baseline 20/20",
         "where": "paper/WRITEUP.md s3.3 table", "adapter": "not yielded (file counts 16/20 and 20/20)", "match": True},
    ],
}


def attempts(src):
    for path, obj in src.json(r"^outputs/pytest_resolve\.json$"):
        if obj.get("model") != "gemma4:12b":
            continue
        for row in obj.get("agent_rows") or []:
            y = int(bool(row.get("agent_resolve")))
            error = row.get("error")
            yield {
                "file": path,
                "system": "moepa-12b-pytest-gate",
                "run": "pytest_resolve/agent-gemma4-12b",
                "task": row["instance_id"],
                "y": y,
                "model": "12B",
                "harness": "local",
                "err": bool(error),
                "status": str(error) if error else ("resolved" if y else "failed"),
            }
