"""Checks of tools/compare_runs.py against the pooled table.

1. Repeats of one system must fall inside the tool's noise band about 95% of
   the time (they are the cases where the true difference is zero).
2. The tool's single-run numbers must equal direct computations.
3. Pairs of clearly different systems in one lab must fall outside the band.

    python scripts/test_compare_runs.py
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare_runs as cr  # noqa: E402


def main() -> int:
    band = json.loads((ROOT / "tools" / "noise_band.json").read_text(encoding="utf-8"))
    runs = defaultdict(dict)
    system = {}
    for r in csv.DictReader((ROOT / "data" / "derived" / "attempts.csv").open(encoding="utf-8")):
        runs[r["run"]][r["task"]] = int(r["y"])
        system[r["run"]] = r["system"]
    by_system = defaultdict(list)
    for run, s in system.items():
        by_system[s].append(run)

    inside = total = 0
    for s, rs in by_system.items():
        for a, b in itertools.combinations(rs, 2):
            if len(set(runs[a]) & set(runs[b])) < 10:
                continue
            res = cr.compare([runs[a]], [runs[b]], band)
            total += 1
            inside += res["within_noise_band"]
    cover = inside / total
    print(f"repeats of one system inside the 95% band: {inside}/{total} = {cover:.3f}")
    assert 0.90 <= cover <= 0.99, cover

    # Direct check of one pair.
    s, rs = next((s, rs) for s, rs in by_system.items() if len(rs) >= 2 and len(set(runs[rs[0]]) & set(runs[rs[1]])) >= 30)
    a, b = runs[rs[0]], runs[rs[1]]
    shared = sorted(set(a) & set(b))
    res = cr.compare([a], [b], band)
    assert res["tasks_compared"] == len(shared)
    assert abs(res["mean_a"] - sum(a[t] for t in shared) / len(shared)) < 1e-12
    only_a = sum(1 for t in shared if a[t] and not b[t])
    only_b = sum(1 for t in shared if b[t] and not a[t])
    assert (res["solved_only_by_a"], res["solved_only_by_b"]) == (only_a, only_b)
    assert abs(cr.mcnemar_exact(3, 9) - 0.14599609375) < 1e-12        # 2 * P(X <= 3 | n = 12, p = 1/2)
    assert cr.mcnemar_exact(0, 0) == 1.0
    print("single-pair numbers and the exact McNemar test: ok")

    # Several runs per agent: the permutation test on a repeated system against itself is not extreme.
    s, rs = next((s, rs) for s, rs in by_system.items() if len(rs) >= 4)
    common = set.intersection(*[set(runs[r]) for r in rs[:4]])
    if len(common) >= 10:
        res = cr.compare([runs[rs[0]], runs[rs[1]]], [runs[rs[2]], runs[rs[3]]], band)
        assert "p_value_permutation" in res and 0 < res["p_value_permutation"] <= 1
        assert "informative_tasks" in res
        print(f"two runs against two runs of one system: difference {100 * res['difference_b_minus_a']:+.1f} points, "
              f"band +/- {100 * res['noise_band_95']:.1f}, inside: {res['within_noise_band']}")

    # The exact test within tasks: with one run each it is the exact McNemar test.
    pair = [(a, b)]
    for (x, y) in pair:
        sh = sorted(set(x) & set(y))
        p_strat, informative = cr.stratified_exact([x], [y], sh)
        oa = sum(1 for t in sh if x[t] and not y[t])
        ob = sum(1 for t in sh if y[t] and not x[t])
        assert informative == oa + ob
        assert abs(p_strat - cr.mcnemar_exact(oa, ob)) < 1e-9, (p_strat, cr.mcnemar_exact(oa, ob))
    # A hand-checked case: one task, agent A solves 0 of 2, agent B 2 of 2 -> P = 2 * (1 / C(4,2)) = 1/3.
    p_hand, _ = cr.stratified_exact([{"t": 0}, {"t": 0}], [{"t": 1}, {"t": 1}], ["t"])
    assert abs(p_hand - 1 / 3) < 1e-12, p_hand
    # Under the null (a system against itself, every split of four of its runs into two and two) the test
    # must not reject much more often than its level.
    n_null = rejected = 0
    for s_, rs_ in by_system.items():
        if len(rs_) < 4:
            continue
        for four in itertools.combinations(rs_, 4):
            common = sorted(set.intersection(*[set(runs[r]) for r in four]))
            if len(common) < 20:
                continue
            for left in ((0, 1), (0, 2), (0, 3)):
                right = [i for i in range(4) if i not in left]
                p_null, _ = cr.stratified_exact([runs[four[i]] for i in left], [runs[four[i]] for i in right], common)
                n_null += 1
                rejected += p_null <= 0.05
    if n_null:
        print(f"exact test within tasks, a system against itself (two runs against two): rejected {rejected}/{n_null} "
              f"at the 5% level")
        assert rejected / n_null <= 0.10
    print("exact test within tasks: ok (equals McNemar for one run each; hand-checked case)")

    # Different systems with a large gap must be flagged.
    model = {}
    for r in csv.DictReader((ROOT / "data" / "derived" / "attempts.csv").open(encoding="utf-8")):
        model[r["run"]] = (r["lab"], r["model"])
    flagged = seen = 0
    for a, b in itertools.combinations(list(runs), 2):
        if model[a][0] != model[b][0] or {model[a][1], model[b][1]} != {"31B", "E4B"}:
            continue
        if len(set(runs[a]) & set(runs[b])) < 50:
            continue
        res = cr.compare([runs[a]], [runs[b]], band)
        seen += 1
        flagged += not res["within_noise_band"]
        if seen >= 400:
            break
    print(f"31B against E4B in one lab on at least 50 shared tasks: flagged as outside the band {flagged}/{seen}")
    assert seen == 0 or flagged / seen > 0.8
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
