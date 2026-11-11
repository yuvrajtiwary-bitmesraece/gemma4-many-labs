"""Extract the external validation set Zenodo 23834104 into the pooled-table format.

Source (NOT part of the pooled table): Toeda, T. (2026). "Scaffolding Helps the Smallest
Gemma 4 Agent, Not the Larger Ones: run records, pre-registration and analysis code
(3 capacities x 2 scaffold levels x 5 seeds, 104 tasks)", version 1.0.5, Zenodo,
doi:10.5281/zenodo.23834104, Apache-2.0.  File gemma4-scaffold-capacity-v1.0.5.zip
(sha256 2ee9d5d874d7a3c74ddc36983335c43ddbd64640dde37518600f36130ae8ee90), member
data/runs.jsonl (sha256 2212377d510825d35cb2fdd4aa748b823040566619385da9be78e93e027f3663):
one JSON line per (model, scaffold level, seed, task).

This script reads the extracted text files only.  Nothing from the archive is imported or
executed.  Standard library only.

    python -I scripts/external_zenodo.py            writes data/derived/external_zenodo_attempts.csv

Output columns are those of data/derived/attempts.csv:

    lab      "zenodo-23834104"
    source   "zenodo:23834104"
    system   lab/<size>-<level>            e.g. zenodo-23834104/E4B-min   (9 systems)
    run      system/s<seed>                e.g. zenodo-23834104/E4B-min/s1 (39 runs)
    task     competition development-set id, e.g. fastapi_14372
    y        1 only if the record says resolved is true, else 0
    model    E4B | 12B | 26B-A4B  (the dataset has no E2B and no 31B)
    harness  "official": the competition's swegemma 0.2.11 harness (verify_task) scored every
             attempt.  It ran on the author's machine, in a repaired sandbox image / wheel set,
             against llama.cpp (QAT q4_0 GGUF) through a wrapper with three fidelity patches;
             it is not the Kaggle scoring server.
    err      1 if the harness attached an error string (error_class not null: session timeout,
             tool-call budget, turn budget, context-window overflow), else 0
    status   end_reason, plus the test exit code when not resolved
    file     path of the record file inside the archive

Coding decisions

* Seeds are sampling seeds (temperature 1.0, top_p 0.95; the runner appends "seed: <s>" to
  the generation config).  The runs of one (size, level) are therefore replicates of one
  system: same system label, one run label per seed.
* Level "mid" (9-rule instruction, six tools, seeds 1-3) is exploratory in the source's
  design.  It is a regular set of runs and is kept; use --levels min,max for the
  confirmatory 3 x 2 x 5 design only.
* Time caps, tool-call caps, turn caps, empty patches, failed or timed-out tests and
  context-window overflows are y = 0: the harness scores them as failures and the source
  analyses them as failures.
* Rows the source marks as not run or invalid for infrastructure reasons are dropped and
  counted.  Version 1.0.5 contains none: aborted cells were discarded and re-run by the
  author before release (DEVIATIONS.md D1, D2, D5, D6, D10), so the count is 0.  An
  unknown end_reason stops the script instead of being guessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SRC_DEFAULT = ROOT / "data" / "raw" / "_external" / "zenodo_23834104" / "gemma4-scaffold-capacity"
OUT_DEFAULT = ROOT / "data" / "derived" / "external_zenodo_attempts.csv"
LOG_DEFAULT = ROOT / "data" / "raw" / "_notes" / "external_zenodo" / "extract_log.json"
DEV_TASKS = ROOT / "data" / "derived" / "dev_tasks.txt"

LAB = "zenodo-23834104"
SOURCE = "zenodo:23834104"
RECORD_FILE = "data/runs.jsonl"
RUNS_SHA256 = "2212377d510825d35cb2fdd4aa748b823040566619385da9be78e93e027f3663"

COLUMNS = ["lab", "source", "system", "run", "task", "y", "model", "harness", "err", "status", "file"]
TASK = re.compile(r"^(fastapi|requests|rich|httpx)[_\-](\d{1,6})$", re.I)
MODEL = {"gemma-4-e4b-it": "E4B", "gemma-4-12b-it": "12B", "gemma-4-26b-a4b-it": "26B-A4B"}
LEVELS = ("min", "mid", "max")
HARNESS = "official"
MIN_TASKS_PER_RUN = 10

# end_reason values of version 1.0.5 and how they are coded.
KEPT_END_REASONS = {
    "resolved",               # verify_task passed
    "submitted_unresolved",   # patch scored, tests failed
    "time_cap",               # "Agent exceeded session timeout (6.0 min)"
    "tool_cap",               # "Agent exceeded tool call budget (50 calls)"
    "turn_cap",               # "Agent exceeded turns budget (80 turns)"
    "other_error",            # "Sandbox execution error: litellm.ContextWindowExceededError:" (agent context overflow)
}
INFRA_END_REASONS: set = set()            # none in version 1.0.5
INFRA_FLAGS = ("invalid", "not_run", "skipped", "aborted")   # none of these fields exists in version 1.0.5


def canonical_task(raw):
    m = TASK.match(str(raw).strip())
    return f"{m.group(1).lower()}_{int(m.group(2))}" if m else None


def infra_invalid(rec):
    """Reason string if the source marks the row as not run / invalid for infrastructure reasons."""
    if rec.get("end_reason") in INFRA_END_REASONS:
        return f"end_reason={rec['end_reason']}"
    for flag in INFRA_FLAGS:
        if rec.get(flag):
            return f"flag {flag}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT, help="extracted archive folder")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--log", type=Path, default=LOG_DEFAULT)
    ap.add_argument("--levels", default=",".join(LEVELS), help="scaffold levels to keep (default: min,mid,max)")
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    args = ap.parse_args()
    levels = [x for x in args.levels.split(",") if x]
    if any(x not in LEVELS for x in levels):
        print(f"unknown level in {levels}; allowed {LEVELS}")
        return 2

    raw = (args.src / RECORD_FILE).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RUNS_SHA256:
        print(f"sha256 of {RECORD_FILE} is {digest}, expected {RUNS_SHA256}")
        if not args.allow_hash_mismatch:
            return 2
    dev = {t.strip() for t in DEV_TASKS.read_text(encoding="utf-8").splitlines() if t.strip()}
    taskset = [t.strip() for t in (args.src / "TASKSET.txt").read_text(encoding="utf-8").splitlines() if t.strip()]

    stats = Counter()
    problems = []
    rows, seen = [], set()
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        stats["records_read"] += 1
        try:
            rec = json.loads(line)
        except ValueError as exc:
            problems.append(f"line {n}: not JSON ({exc})")
            continue
        missing = [k for k in ("level", "model", "seed", "instance_id", "resolved", "end_reason") if k not in rec]
        if missing:
            problems.append(f"line {n}: missing {missing}")
            continue
        reason = infra_invalid(rec)
        if reason:
            stats["dropped_not_run_or_infrastructure"] += 1
            stats[f"dropped_infrastructure[{reason}]"] += 1
            continue
        if rec["end_reason"] not in KEPT_END_REASONS:
            problems.append(f"line {n}: unknown end_reason {rec['end_reason']!r}")
            continue
        if not isinstance(rec["resolved"], bool):
            problems.append(f"line {n}: resolved is not a boolean: {rec['resolved']!r}")
            continue
        if (rec["end_reason"] == "resolved") != rec["resolved"]:
            problems.append(f"line {n}: end_reason {rec['end_reason']!r} contradicts resolved={rec['resolved']!r}")
            continue
        if rec["model"] not in MODEL:
            problems.append(f"line {n}: unknown model {rec['model']!r}")
            continue
        if rec["level"] not in LEVELS or not isinstance(rec["seed"], int) or isinstance(rec["seed"], bool):
            problems.append(f"line {n}: bad level/seed {rec['level']!r}/{rec['seed']!r}")
            continue
        if rec["level"] not in levels:
            stats["skipped_level_not_requested"] += 1
            continue
        task = canonical_task(rec["instance_id"])
        if task is None:
            stats["dropped_not_a_dev_style_id"] += 1
            continue
        if task not in dev:
            stats["dropped_not_in_dev_set"] += 1
            continue
        model = MODEL[rec["model"]]
        system = f"{LAB}/{model}-{rec['level']}"
        run = f"{system}/s{rec['seed']}"
        if (run, task) in seen:
            stats["dropped_repeat_within_run"] += 1
            continue
        seen.add((run, task))
        y = 1 if rec["resolved"] is True else 0
        status = "resolved" if y else f"{rec['end_reason']} (test exit {rec.get('test_exit_code')})"
        rows.append({"lab": LAB, "source": SOURCE, "system": system, "run": run, "task": task, "y": y,
                     "model": model, "harness": HARNESS, "err": int(rec.get("error_class") is not None),
                     "status": status[:60], "file": RECORD_FILE})
    if problems:
        print(f"{len(problems)} problems, nothing written; first: {problems[:5]}")
        return 1

    # A run needs at least MIN_TASKS_PER_RUN eligible tasks (rule of the pooled table).
    per_run = Counter(r["run"] for r in rows)
    small = {run for run, k in per_run.items() if k < MIN_TASKS_PER_RUN}
    if small:
        stats["dropped_run_below_min_tasks"] = sum(per_run[run] for run in small)
        rows = [r for r in rows if r["run"] not in small]
    stats["dropped_not_run_or_infrastructure"] += 0
    stats["attempts_written"] = len(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    runs = OrderedDict()
    for r in rows:
        runs.setdefault(r["run"], []).append(r)
    systems = OrderedDict()
    for run, rs in runs.items():
        systems.setdefault(rs[0]["system"], []).append(run)
    tasks = {r["task"] for r in rows}
    summary = {
        "source": SOURCE, "doi": "10.5281/zenodo.23834104", "record_file": RECORD_FILE, "record_file_sha256": digest,
        "levels_kept": levels, "counts": dict(sorted(stats.items())),
        "attempts": len(rows), "systems": len(systems), "runs": len(runs), "tasks": len(tasks),
        "resolved": sum(r["y"] for r in rows), "err": sum(r["err"] for r in rows),
        "tasks_equal_TASKSET": tasks == set(taskset),
        "models": dict(Counter(r["model"] for r in rows)),
        "status": dict(Counter(r["status"] for r in rows).most_common()),
        "per_run": [{"system": rs[0]["system"], "run": run, "model": rs[0]["model"], "n_tasks": len(rs),
                     "n_resolved": sum(r["y"] for r in rs), "n_err": sum(r["err"] for r in rs)}
                    for run, rs in runs.items()],
    }

    # Check against the numbers the source reports for its six confirmatory cells.
    checks = []
    reported_path = args.src / "analysis" / "summary.json"
    if reported_path.exists():
        reported = json.loads(reported_path.read_text(encoding="utf-8"))["cells"]
        inverse = {v: k for k, v in MODEL.items()}
        for system, sruns in systems.items():
            size, level = system.split("/", 1)[1].rsplit("-", 1)
            cell = reported.get(f"{level}|{inverse[size]}")
            if cell is None:
                continue
            ys = [r["y"] for run in sruns for r in runs[run]]
            rate = sum(ys) / len(ys)
            checks.append({"system": system, "extracted_rate": rate, "reported_rate": cell["resolved_rate"],
                           "runs": len(sruns), "reported_seeds": cell["seeds"],
                           "match": abs(rate - cell["resolved_rate"]) < 1e-12 and len(sruns) == cell["seeds"]})
    summary["checks_against_source_summary"] = checks

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    print(f"read {stats['records_read']} records from {RECORD_FILE} (sha256 ok: {digest == RUNS_SHA256})")
    print(f"dropped: not run / infrastructure {stats['dropped_not_run_or_infrastructure']}, "
          f"not a dev-style id {stats['dropped_not_a_dev_style_id']}, not in dev set {stats['dropped_not_in_dev_set']}, "
          f"repeat within run {stats['dropped_repeat_within_run']}, run below {MIN_TASKS_PER_RUN} tasks "
          f"{stats['dropped_run_below_min_tasks']}, level not requested {stats['skipped_level_not_requested']}")
    print(f"written {len(rows)} attempts: {len(systems)} systems, {len(runs)} runs, {len(tasks)} tasks, "
          f"{summary['resolved']} resolved, {summary['err']} with err=1 -> {args.out}")
    print(f"{'run':36s} {'model':8s} {'n':>4s} {'res':>4s} {'err':>4s}")
    for p in summary["per_run"]:
        print(f"{p['run']:36s} {p['model']:8s} {p['n_tasks']:4d} {p['n_resolved']:4d} {p['n_err']:4d}")
    ok = all(c["match"] for c in checks)
    print(f"check against the source's analysis/summary.json: {sum(c['match'] for c in checks)}/{len(checks)} cells match"
          f" -> {'PASS' if ok and checks else 'no check' if not checks else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
