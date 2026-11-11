"""Build the pooled attempts table from the per-source adapters.

Each adapter (scripts/adapters/*.py) reads one pinned source and yields one
record per attempt.  This script applies the inclusion rules of
ANALYSIS_PLAN.md that are common to all sources and writes

    data/derived/attempts.csv     one row per (run, task)
    data/derived/runs.csv         one row per run
    data/derived/sources.csv      one row per candidate source
    data/derived/harmonize_log.json   counts of everything that was dropped

Standard library only.

    python scripts/harmonize.py                  build everything
    python scripts/harmonize.py --check owner/repo [--runs]   inspect one adapter
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import srcio  # noqa: E402

ADAPTERS = HERE / "adapters"
DERIVED = srcio.ROOT / "data" / "derived"

TASK = re.compile(r"^(fastapi|requests|rich|httpx)[_\-](\d{1,6})$", re.I)
MODELS = {"E2B", "E4B", "12B", "26B-A4B", "31B"}
HARNESS = {"official", "local"}
DEV_COMPOSITION = {"fastapi": 67, "rich": 48, "requests": 13, "httpx": 1}
MIN_TASKS_PER_RUN = 10

ATTEMPT_COLUMNS = ["lab", "source", "system", "run", "task", "y", "model", "harness", "err", "status", "file"]


def load_adapters(wanted=None):
    mods = []
    for path in sorted(ADAPTERS.glob("*.py"), key=lambda p: p.name.lower()):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location("adapter_" + re.sub(r"\W", "_", path.stem), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if wanted and mod.REPO not in wanted:
            continue
        mods.append(mod)
    return mods


def canonical_task(raw) -> str | None:
    m = TASK.match(str(raw).strip())
    return f"{m.group(1).lower()}_{int(m.group(2))}" if m else None


def collect(mod):
    """Run one adapter. Returns (rows, problems, stats)."""
    src = srcio.Source(mod.REPO)
    lab = mod.REPO.split("/", 1)[0]
    rows, problems, stats = [], [], Counter()
    run_system, run_meta = {}, {}
    for i, a in enumerate(mod.attempts(src)):
        stats["yielded"] += 1
        missing = [k for k in ("file", "system", "run", "task", "y") if k not in a or a[k] is None or a[k] == ""]
        if missing:
            problems.append(f"record {i}: missing {missing}: {a}")
            continue
        y = a["y"]
        if isinstance(y, bool):
            y = int(y)
        if y not in (0, 1):
            problems.append(f"record {i}: y must be 0/1, got {a['y']!r}")
            continue
        model, harness, err = a.get("model"), a.get("harness"), a.get("err")
        if model is not None and model not in MODELS:
            problems.append(f"record {i}: model {model!r} not in {sorted(MODELS)}")
            continue
        if harness is not None and harness not in HARNESS:
            problems.append(f"record {i}: harness {harness!r} not in {sorted(HARNESS)}")
            continue
        if err not in (True, False, None):
            problems.append(f"record {i}: err must be True/False/None, got {err!r}")
            continue
        system, run = str(a["system"]), str(a["run"])
        if run_system.setdefault(run, system) != system:
            problems.append(f"run {run!r} is assigned to two systems: {run_system[run]!r} and {system!r}")
            continue
        if run_meta.setdefault(run, (model, harness)) != (model, harness):
            problems.append(f"run {run!r} has inconsistent model/harness: {run_meta[run]} vs {(model, harness)}")
            continue
        task = canonical_task(a["task"])
        if task is None:
            stats["dropped_not_a_dev_style_id"] += 1
            continue
        rows.append({"lab": lab, "source": mod.REPO, "system": f"{lab}/{system}", "run": f"{lab}/{run}",
                     "task": task, "y": y, "model": model or "", "harness": harness or "",
                     "err": "" if err is None else int(err), "status": str(a.get("status") or "")[:60],
                     "file": a["file"]})
    return rows, problems, stats


def file_hashes(repo: str, paths) -> dict:
    """sha256 of the named files of one source (one pass over the tarball)."""
    want = set(paths)
    out = {}
    for tf, m, rel in srcio.Source(repo)._members():
        if rel in want:
            out[rel] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    return out


def summarise_runs(rows):
    runs = defaultdict(list)
    for r in rows:
        runs[r["run"]].append(r)
    out = []
    for run, rs in runs.items():
        tasks = {}
        for r in rs:
            tasks.setdefault(r["task"], r["y"])
        sig = hashlib.sha1(",".join(sorted(tasks)).encode()).hexdigest()[:10]
        errs = [r["err"] for r in rs if r["err"] != ""]
        out.append({"lab": rs[0]["lab"], "source": rs[0]["source"], "system": rs[0]["system"], "run": run,
                    "n_tasks": len(tasks), "n_resolved": sum(tasks.values()),
                    "rate": round(sum(tasks.values()) / len(tasks), 4), "model": rs[0]["model"],
                    "harness": rs[0]["harness"], "n_err": sum(errs) if errs else "",
                    "task_set": sig, "files": len({r["file"] for r in rs})})
    return out


def check(repo: str, show_runs: bool) -> int:
    mods = load_adapters({repo})
    if not mods:
        print(f"no adapter declares REPO = {repo!r}")
        return 2
    mod = mods[0]
    card = getattr(mod, "CARD", {})
    print(f"== {mod.REPO} ==")
    for k in ("eligible", "reason", "licence", "models", "harness"):
        if k in card:
            print(f"  {k}: {card[k]}")
    rows, problems, stats = collect(mod)
    print(f"  yielded={stats['yielded']} usable={len(rows)} not_dev_style_id={stats['dropped_not_a_dev_style_id']} "
          f"problems={len(problems)}")
    for p in problems[:15]:
        print("  PROBLEM:", p)
    runs = summarise_runs(rows)
    systems = defaultdict(list)
    for r in runs:
        systems[r["system"]].append(r)
    dup = Counter((r["run"], r["task"]) for r in rows)
    n_dup = sum(1 for v in dup.values() if v > 1)
    print(f"  systems={len(systems)} runs={len(runs)} (run,task) duplicates={n_dup} "
          f"distinct tasks={len({r['task'] for r in rows})} "
          f"runs with <{MIN_TASKS_PER_RUN} tasks={sum(r['n_tasks'] < MIN_TASKS_PER_RUN for r in runs)}")
    if show_runs or len(runs) <= 60:
        print(f"  {'system':38s} {'run':44s} {'n':>4s} {'res':>4s} {'model':8s} {'harness':8s} {'err':>4s} task_set")
        for system in sorted(systems):
            for r in sorted(systems[system], key=lambda r: r["run"]):
                print(f"  {system[-38:]:38s} {r['run'][-44:]:44s} {r['n_tasks']:4d} {r['n_resolved']:4d} "
                      f"{r['model']:8s} {r['harness']:8s} {str(r['n_err']):>4s} {r['task_set']}")
    else:
        print("  (more than 60 runs; pass --runs to list them)")
        by = Counter((len(v)) for v in systems.values())
        print("  runs per system:", dict(sorted(by.items())))
    return 1 if problems else 0


def build() -> int:
    DERIVED.mkdir(parents=True, exist_ok=True)
    mods = load_adapters()
    log = {"sources": {}, "totals": Counter()}
    all_rows = []
    source_rows = []
    for mod in mods:
        card = getattr(mod, "CARD", {})
        entry = srcio.manifest()[mod.REPO]
        srow = {"source": mod.REPO, "lab": mod.REPO.split("/", 1)[0], "commit": entry["commit"],
                "licence": (card.get("licence") or {}).get("spdx", ""),
                "licence_evidence": (card.get("licence") or {}).get("evidence", ""),
                "eligible": bool(card.get("eligible")), "reason_not_used": card.get("reason", ""),
                "models": ";".join(card.get("models") or []), "harness": card.get("harness", "") or ""}
        if not card.get("eligible"):
            srow.update(n_attempts=0, n_runs=0, n_systems=0)
            source_rows.append(srow)
            continue
        rows, problems, stats = collect(mod)
        if problems:
            print(f"{mod.REPO}: {len(problems)} problems, e.g. {problems[0]}")
            return 1
        slog = Counter(stats)
        # Identical files within a source are counted once.
        hashes = file_hashes(mod.REPO, {r["file"] for r in rows})
        first_by_hash = {}
        for path in sorted(hashes):
            first_by_hash.setdefault(hashes[path], path)
        keep = []
        for r in rows:
            h = hashes.get(r["file"])
            if h is not None and first_by_hash[h] != r["file"]:
                slog["dropped_identical_file"] += 1
                continue
            keep.append(r)
        # One row per (run, task): first occurrence.
        seen, rows2 = set(), []
        for r in keep:
            key = (r["run"], r["task"])
            if key in seen:
                slog["dropped_repeat_within_run"] += 1
                continue
            seen.add(key)
            rows2.append(r)
        log["sources"][mod.REPO] = slog
        srow["_rows"] = rows2
        source_rows.append(srow)
        all_rows.extend(rows2)

    # Development set.  We do not hold the competition data, so the set is taken
    # from the sources: a run "lists the development set" if it covers exactly
    # 129 tasks with the documented composition (67 fastapi, 48 rich, 13
    # requests, 1 httpx).  The set named by the most labs is used, and the
    # log records how many labs name it and whether any lab names another.
    tasks_by_run, lab_by_run = defaultdict(set), {}
    for r in all_rows:
        tasks_by_run[r["run"]].add(r["task"])
        lab_by_run[r["run"]] = r["lab"]
    labs_by_set = defaultdict(set)
    for run, tasks in tasks_by_run.items():
        if len(tasks) == sum(DEV_COMPOSITION.values()) and \
                dict(Counter(t.split("_")[0] for t in tasks)) == DEV_COMPOSITION:
            labs_by_set[tuple(sorted(tasks))].add(lab_by_run[run])
    if not labs_by_set:
        print("no run covers a 129-task set with the documented composition")
        return 1
    dev_tasks = max(labs_by_set, key=lambda s: len(labs_by_set[s]))
    log["dev_set"] = {"n_tasks": len(dev_tasks), "labs_listing_exactly_this_set": len(labs_by_set[dev_tasks]),
                      "other_full_composition_sets": len(labs_by_set) - 1}
    dev = set(dev_tasks)
    task_universe = Counter(r["task"] for r in all_rows)
    log["dev_set"]["distinct_dev_style_ids_before_filter"] = len(task_universe)
    rows3 = []
    for r in all_rows:
        if r["task"] not in dev:
            log["sources"][r["source"]]["dropped_not_in_dev_set"] += 1
            continue
        rows3.append(r)
    # Runs need at least MIN_TASKS_PER_RUN eligible tasks.
    n_by_run = Counter(r["run"] for r in rows3)
    final = []
    for r in rows3:
        if n_by_run[r["run"]] < MIN_TASKS_PER_RUN:
            log["sources"][r["source"]]["dropped_run_too_small"] += 1
            continue
        final.append(r)

    with (DERIVED / "attempts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ATTEMPT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(final)
    runs = summarise_runs(final)
    with (DERIVED / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(runs, key=lambda r: (r["lab"].lower(), r["system"], r["run"])))
    (DERIVED / "dev_tasks.txt").write_text("\n".join(dev_tasks) + "\n", encoding="utf-8")
    used = Counter(r["source"] for r in final)
    runs_by = Counter(r["source"] for r in runs)
    systems_by = defaultdict(set)
    for r in runs:
        systems_by[r["source"]].add(r["system"])
    cols = ["source", "lab", "commit", "licence", "licence_evidence", "eligible", "reason_not_used", "models",
            "harness", "n_attempts", "n_runs", "n_systems"]
    with (DERIVED / "sources.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for s in source_rows:
            if "_rows" in s:
                s.update(n_attempts=used[s["source"]], n_runs=runs_by[s["source"]],
                         n_systems=len(systems_by[s["source"]]))
            w.writerow(s)
    log["totals"] = {"attempts": len(final), "runs": len(runs), "systems": len({r["system"] for r in runs}),
                     "labs": len({r["lab"] for r in runs}), "sources_with_adapter": len(mods),
                     "sources_used": len(used), "tasks": len({r["task"] for r in final})}
    log["sources"] = {k: dict(v) for k, v in log["sources"].items()}
    (DERIVED / "harmonize_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(log["totals"]), json.dumps(log["dev_set"]))
    return 0


def main(argv) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv and argv[0] == "--check":
        return check(argv[1], "--runs" in argv)
    return build()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
