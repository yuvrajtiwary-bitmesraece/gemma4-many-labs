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

    # Lab-balanced resolve rate: repeats within system, systems within lab, then labs (plan, section 6).
    run_rate = defaultdict(list)
    for r in rows:
        run_rate[(r["lab"], r["system"], r["run"])].append(r["y"])
    sys_rate = defaultdict(list)
    for (lab, system, run), ys in run_rate.items():
        sys_rate[(lab, system)].append(mean(ys))
    lab_rate = defaultdict(list)
    for (lab, system), rates in sys_rate.items():
        lab_rate[lab].append(mean(rates))
    check("lab-balanced resolve rate", mean(mean(v) for v in lab_rate.values()), res["data"]["lab_balanced_resolve_rate"])

    # Run-to-run noise share on cells with repeats, every lab with the same total weight on those cells.
    rep = {k: ys for k, ys in cell.items() if len(ys) >= 2}
    n_lab = defaultdict(int)
    for (lab, system, task), ys in rep.items():
        n_lab[lab] += len(ys)
    sw = swy = 0.0
    for (lab, system, task), ys in rep.items():
        sw += len(ys) / n_lab[lab]
        swy += sum(ys) / n_lab[lab]
    grand = swy / sw
    noise = total = 0.0
    for (lab, system, task), ys in rep.items():
        n, pm = len(ys), mean(ys)
        noise += (n / n_lab[lab]) * (n / (n - 1)) * pm * (1 - pm)
        total += sum((y - grand) ** 2 for y in ys) / n_lab[lab]
    check("noise share on replicated cells (lab-balanced)", noise / total,
          res["P2"]["replicate_noise_lab_balanced"]["noise_share"])

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

    # The same pairs, pair -> system -> lab, then the mean over labs.
    sys_flip = defaultdict(list)
    for (lab, system), runs in runs_of.items():
        for a, b in itertools.combinations(list(runs), 2):
            common = set(runs[a]) & set(runs[b])
            if len(common) >= 10:
                sys_flip[(lab, system)].append(sum(runs[a][t] != runs[b][t] for t in common) / len(common))
    lab_flip = defaultdict(list)
    for (lab, system), v in sys_flip.items():
        lab_flip[lab].append(mean(v))
    check("labs with repeats", len(lab_flip), res["P3"]["flip_rate"]["n_labs"], 0)
    check("mean of the lab-level flip rates", mean(mean(v) for v in lab_flip.values()), res["P3"]["flip_rate"]["lab_mean"])

    # Task status from the four full audits of the stock sandbox (Amendment 4).
    ref, empty = defaultdict(dict), {}
    for c in csv.DictReader((ROOT / "data" / "derived" / "controls.csv").open(encoding="utf-8")):
        if c["environment"] != "kaggle_subprocess":
            continue
        if c["kind"] == "reference_patch":
            ref[c["lab"]][c["task"]] = int(float(c["value"]))
        elif c["kind"] == "empty_patch" and c["lab"] == "TechNaom":
            empty[c["task"]] = int(float(c["value"]))
    full = [lab for lab, v in ref.items() if len(v) == 129]
    check("full audits of the stock sandbox", len(full), len(res["X1_validity"]["full_audits"]), 0)
    votes = {t: sum(ref[lab][t] for lab in full) for t in ref[full[0]]}
    bad = {t for t, v in votes.items() if v == 0 and empty[t] == 0}
    valid = {t for t, v in votes.items() if v == len(full) and empty[t] == 0}
    sc = res["X1_validity"]["status_counts"]
    check("tasks where the reference patch fails in every full audit", len(bad), sc["reference_patch_fails"], 0)
    check("valid tasks", len(valid), sc["valid"], 0)
    check("tasks that resolve with no patch", sum(empty.values()), sc["passes_without_patch"], 0)
    check("share of attempts on tasks where the reference patch fails", mean(r["task"] in bad for r in rows),
          res["X1_validity"]["share_of_attempts_on_tasks_where_reference_patch_fails"])
    check("floor tasks that are valid", sum(1 for t, d in d5.items() if d <= 0.10 and t in valid),
          res["X1_validity"]["floor_tasks_that_are_valid"], 0)
    solved_bad = defaultdict(int)
    tried_bad = defaultdict(int)
    for r in rows:
        if r["task"] in bad:
            tried_bad[r["lab"]] += 1
            solved_bad[r["lab"]] += r["y"]
    some = {lab for lab, k in solved_bad.items() if k > 0}
    kinds = res["X1_validity"]["attempts_on_reference_patch_fail_tasks_by_kind_of_lab"]
    check("labs that resolved some such tasks", len(some), kinds["labs_that_resolved_some"]["n_labs"], 0)
    check("their resolve rate on such tasks", sum(solved_bad[lab] for lab in some) / sum(tried_bad[lab] for lab in some),
          kinds["labs_that_resolved_some"]["resolve_rate_there"])
    none_labs = {r["lab"] for r in rows} - some
    check("share of the other labs' attempts on such tasks",
          sum(tried_bad[lab] for lab in none_labs) / sum(1 for r in rows if r["lab"] in none_labs),
          kinds["labs_that_resolved_none"]["share"])

    # Source-reported numbers that the adapters reproduce.
    n = yes = no = 0
    mismatches = []
    for mod in harmonize.load_adapters():
        if harmonize.frame_of(mod.REPO) != "registered":
            continue                      # sources of the extended frame are reported separately
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
