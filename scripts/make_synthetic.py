"""Write a synthetic attempts table with known parameters, to test analyze.py.

The table has the columns of data/derived/attempts.csv.  Outcomes follow a
Rasch model with a lab effect, so the expected answers are known: task
difficulty is shared across labs, replicates differ only by Bernoulli noise.

    python scripts/make_synthetic.py out.csv [seed]
"""
from __future__ import annotations

import csv
import sys

import numpy as np


def main(argv) -> int:
    out = argv[0]
    rng = np.random.default_rng(int(argv[1]) if len(argv) > 1 else 7)
    repos = ["fastapi"] * 67 + ["rich"] * 48 + ["requests"] * 13 + ["httpx"]
    tasks = [f"{r}_{1000 + i}" for i, r in enumerate(repos)]
    b = rng.normal(2.0, 2.0, size=len(tasks))          # difficulties: most tasks are hard
    rows = []
    for li in range(14):
        lab = f"lab{li:02d}"
        lab_shift = rng.normal(0, 0.4)
        model = rng.choice(["31B", "31B", "31B", "E4B", ""])
        harness = rng.choice(["official", "official", "local", ""])
        subset_size = int(rng.choice([129, 129, 60, 47, 30, 20]))
        subset = sorted(rng.choice(len(tasks), size=subset_size, replace=False))
        for si in range(int(rng.integers(1, 6))):
            theta = lab_shift + rng.normal(0, 0.5) - (1.0 if model == "E4B" else 0.0)
            n_rep = int(rng.choice([1, 1, 2, 3]))
            for ri in range(n_rep):
                for ti in subset:
                    p = 1 / (1 + np.exp(-(theta - b[ti])))
                    y = int(rng.random() < p)
                    rows.append([lab, f"{lab}/repo", f"{lab}/s{si}", f"{lab}/s{si}-r{ri}", tasks[ti], y, model,
                                 harness, int(rng.random() < 0.3) if y == 0 else 0, "", "f.jsonl"])
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["lab", "source", "system", "run", "task", "y", "model", "harness", "err", "status", "file"])
        w.writerows(rows)
    print(f"{len(rows)} synthetic attempts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
