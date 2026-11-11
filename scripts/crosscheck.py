"""Independent recomputation of the headline numbers, without mlstats or pandas.

Recomputes from data/derived/attempts.csv and controls.csv with the standard
library only, and compares with results/results.json. Also tallies the
source-reported numbers that the adapters reproduce.

    python scripts/crosscheck.py
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harmonize  # noqa: E402

ROOT = HERE.parent


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def main() -> int:
    res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((ROOT / "data" / "derived" / "attempts.csv").open(encoding="utf-8")))
    for r in rows:
        r["y"] = int(r["y"])
    checks = []

    def check(name, mine, theirs, tol=5e-4):
        ok = abs(mine - theirs) <= tol
        checks.append(ok)
        print(f"{'OK ' if ok else 'DIFF'} {name}: recomputed {mine:.5f}, results.json {theirs:.5f}")

    # Counts.
    check("attempts", len(rows), res["data"]["n_attempts"], 0)
    check("labs", len({r["lab"] for r in rows}), res["data"]["n_labs"], 0)
    check("systems", len({r["system"] for r in rows}), res["data"]["n_systems"], 0)
    check("runs", len({r["run"] for r in rows}), res["data"]["n_runs"], 0)

    # Levels: attempts -> (lab, system, task) cells -> lab-task means -> consensus.
    cell = defaultdict(list)
    for r in rows:
        cell[(r["lab"], r["system"], r["task"])].append(r["y"])
    lab_task = defaultdict(list)
    for (lab, system, task), ys in cell.items():
        lab_task[(lab, task)].append(mean(ys))
    by_task = defaultdict(list)
    for (lab, task), ps in lab_task.items():
        by_task[task].append(mean(ps))
    d5 = {t: mean(ps) for t, ps in by_task.items() if len(ps) >= 5}
    check("tasks with at least 5 labs", len(d5), res["P1"]["n_tasks_5_labs"], 0)
    check("floor tasks", sum(d <= 0.10 for d in d5.values()), res["P1"]["floor_tasks"], 0)
    check("ceiling tasks", sum(d >= 0.90 for d in d5.values()), res["P1"]["ceiling_tasks"], 0)
    check("never resolved by any lab", sum(d == 0 for d in d5.values()), res["P1"]["zero_everywhere_tasks"], 0)

    # Lab-balanced resolve rate: systems within lab, then labs.
    sys_rate = defaultdict(list)
    for r in rows:
        sys_rate[(r["lab"], r["system"])].append(r["y"])
    lab_rate = defaultdict(list)
    for (lab, system), ys in sys_rate.items():
        lab_rate[lab].append(mean(ys))
    check("lab-balanced resolve rate", mean(mean(v) for v in lab_rate.values()), res["data"]["lab_balanced_resolve_rate"])

    # Flip rate over pairs of repeated runs sharing at least 10 tasks, weighted by shared tasks.
    runs_of = defaultdict(dict)
    for r in rows:
        runs_of[(r["lab"], r["system"])].setdefault(r["run"], {})[r["task"]] = r["y"]
    flips = shared = n_pairs = 0
    for runs in runs_of.values():
        for a, b in itertools.combinations(list(runs), 2):
            common = set(runs[a]) & set(runs[b])
            if len(common) >= 10:
                n_pairs += 1
                shared += len(common)
                flips += sum(runs[a][t] != runs[b][t] for t in common)
    check("pairs of repeated runs", n_pairs, res["P3"]["flip_rate"]["n_pairs"], 0)
    check("pooled flip rate", flips / shared, res["P3"]["flip_rate"]["pooled"])

    # Share of attempts on tasks whose reference patch fails in the reference audit.
    gold, empty = {}, {}
    for c in csv.DictReader((ROOT / "data" / "derived" / "controls.csv").open(encoding="utf-8")):
        if c["lab"] == "TechNaom" and c["environment"] == "kaggle_subprocess":
            (gold if c["kind"] == "reference_patch" else empty)[c["task"]] = int(c["value"])
    bad = {t for t in gold if gold[t] == 0 and empty[t] == 0}
    check("tasks where the reference patch fails", len(bad), res["X1_validity"]["status_counts"]["reference_patch_fails"], 0)
    check("share of attempts on those tasks", mean(r["task"] in bad for r in rows),
          res["X1_validity"]["share_of_attempts_on_tasks_where_reference_patch_fails"])
    valid = {t for t in gold if gold[t] == 1 and empty[t] == 0}
    check("floor tasks that are valid", sum(1 for t, d in d5.items() if d <= 0.10 and t in valid),
          res["X1_validity"]["floor_tasks_that_are_valid_in_reference_audit"], 0)

    # Source-reported numbers that the adapters reproduce.
    n = yes = no = 0
    mismatches = []
    for mod in harmonize.load_adapters():
        for item in getattr(mod, "CARD", {}).get("reported_numbers", []):
            n += 1
            if item.get("match") is True:
                yes += 1
            elif item.get("match") is False:
                no += 1
                mismatches.append(f"{mod.REPO}: {item.get('what')} (source {item.get('value')}, adapter {item.get('adapter')})")
    print(f"source-reported numbers recorded in adapter cards: {n}; reproduced: {yes}; not reproduced: {no}; "
          f"unmarked: {n - yes - no}")
    for m in mismatches:
        print("  mismatch:", m[:260])
    print("ALL CHECKS OK" if all(checks) else "SOME CHECKS DIFFER")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
