"""Is the difference between two agents real, or run-to-run noise?

Give it per-task results of agent A and agent B on the development tasks of
the Gemma 4 Developer Agent competition (one or more runs each). It reports

  - the score difference and an exact test that uses no model: the exact McNemar
    test for one run each, the exact permutation test within tasks for more runs;
  - the noise band: how far two runs of ONE agent at this score level differ
    on exactly these tasks. The band comes from the pooled results of 14 labs
    (task difficulties and a noise scale in noise_band.json). Leave-one-lab-out,
    its 95% band covered 96% of the score differences between repeated runs;
  - how many runs per agent a difference of this size needs to be detected.

Standard library only. Result files may be the harness's summary.jsonl
(instance_id, resolved), a CSV with a task column and an outcome column, or a
JSON list of such records.

    python tools/compare_runs.py --a base_run1.jsonl base_run2.jsonl --b new_run1.jsonl
    python tools/compare_runs.py --a a.csv --b b.csv --json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = re.compile(r"^(fastapi|requests|rich|httpx)[_\-](\d{1,6})$", re.I)
TASK_KEYS = ("instance_id", "task", "task_id", "id")
OUTCOME_KEYS = ("resolved", "y", "passed", "success", "outcome", "status")
TRUE = {"1", "true", "t", "yes", "y", "resolved", "pass", "passed", "success", "solved", "ok"}
Z975, Z80 = 1.959964, 0.841621


def read_run(path: str) -> dict:
    """task -> 0/1 from a JSONL, JSON or CSV file."""
    text = Path(path).read_text(encoding="utf-8-sig")
    records = []
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = [r for r in json.loads(text) if isinstance(r, dict)]
    elif stripped.startswith("{"):
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    else:
        records = list(csv.DictReader(io.StringIO(text)))
    out = {}
    for rec in records:
        low = {str(k).lower(): v for k, v in rec.items()}
        task = next((low[k] for k in TASK_KEYS if k in low), None)
        outcome = next((low[k] for k in OUTCOME_KEYS if k in low), None)
        if task is None or outcome is None:
            continue
        m = TASK.match(str(task).strip())
        if not m:
            continue
        key = f"{m.group(1).lower()}_{int(m.group(2))}"
        val = outcome if isinstance(outcome, bool) else str(outcome).strip().lower() in TRUE
        out.setdefault(key, int(bool(val)))
    if not out:
        raise SystemExit(f"{path}: no per-task outcomes found (need a task id column and a resolved column)")
    return out


def expit(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def ability_for_rate(rate: float, b: list) -> float:
    rate = min(max(rate, 0.25 / len(b)), 1 - 0.25 / len(b))
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(expit(mid - x) for x in b) / len(b) < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def stratified_exact(runs_a: list, runs_b: list, tasks: list) -> tuple:
    """Two-sided exact permutation test within tasks. Returns (p-value, number of informative tasks).

    If the two agents were the same, the run labels would be exchangeable within each task. For a task solved
    in s of the ka + kb runs, the number of those successes that fall on agent B is then hypergeometric; the
    statistic is the total number of B's successes over the tasks. One run each gives the exact McNemar test.
    """
    ka, kb = len(runs_a), len(runs_b)
    dist = [1.0]
    obs = informative = 0
    for t in tasks:
        xb = sum(r[t] for r in runs_b)
        s = xb + sum(r[t] for r in runs_a)
        if s == 0 or s == ka + kb:
            continue
        informative += 1
        obs += xb
        lo, hi = max(0, s - ka), min(kb, s)
        den = math.comb(ka + kb, s)
        pmf = [math.comb(kb, x) * math.comb(ka, s - x) / den if x >= lo else 0.0 for x in range(hi + 1)]
        new = [0.0] * (len(dist) + hi)
        for i, d in enumerate(dist):
            if d:
                for x in range(lo, hi + 1):
                    new[i + x] += d * pmf[x]
        dist = new
    if not informative:
        return 1.0, 0
    return min(1.0, 2 * min(sum(dist[: obs + 1]), sum(dist[obs:]))), informative


def normal_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def compare(runs_a: list, runs_b: list, band: dict) -> dict:
    shared = set.intersection(*[set(r) for r in runs_a + runs_b])
    known = sorted(t for t in shared if t in band["difficulty"])
    if len(known) < 5:
        raise SystemExit("fewer than 5 shared development-set tasks across the runs")
    n = len(known)
    score_a = [sum(r[t] for t in known) / n for r in runs_a]
    score_b = [sum(r[t] for t in known) / n for r in runs_b]
    mean_a, mean_b = sum(score_a) / len(score_a), sum(score_b) / len(score_b)
    diff = mean_b - mean_a
    b = [band["difficulty"][t] for t in known]
    theta = ability_for_rate((mean_a + mean_b) / 2, b)
    p = [expit(theta - x) for x in b]
    per_run_var = band["scale"] * sum(q * (1 - q) for q in p) / (n * n)        # variance of one run's score from noise
    sd_diff = math.sqrt(per_run_var * (1 / len(runs_a) + 1 / len(runs_b)))
    z = diff / sd_diff if sd_diff > 0 else 0.0
    out = {
        "tasks_compared": n, "tasks_shared_but_unknown": len(shared) - n,
        "runs_a": len(runs_a), "runs_b": len(runs_b), "scores_a": score_a, "scores_b": score_b,
        "mean_a": mean_a, "mean_b": mean_b, "difference_b_minus_a": diff,
        "expected_flip_rate_between_two_runs_of_one_agent": 2 * band["scale"] * sum(q * (1 - q) for q in p) / n,
        "noise_sd_of_the_difference": sd_diff, "noise_band_95": 1.959964 * sd_diff,
        "z": z, "p_value_noise_model": normal_two_sided(z),
        "within_noise_band": abs(diff) <= 1.959964 * sd_diff,
    }
    if len(runs_a) == 1 and len(runs_b) == 1:
        only_b = sum(1 for t in known if runs_b[0][t] and not runs_a[0][t])
        only_a = sum(1 for t in known if runs_a[0][t] and not runs_b[0][t])
        out.update(solved_only_by_a=only_a, solved_only_by_b=only_b, p_value_exact_mcnemar=mcnemar_exact(only_a, only_b))
    else:
        # Exact permutation test stratified by task: if the two agents were the same, the run labels would be
        # exchangeable within each task.  For a task solved in s of the ka + kb runs, the number of those
        # successes that fall on agent B is then hypergeometric; the test statistic is the total number of
        # B's successes.  With one run each this is the exact McNemar test.  It uses no noise model.
        p_perm, informative = stratified_exact(runs_a, runs_b, known)
        out.update(p_value_permutation=p_perm, informative_tasks=informative)
    if abs(diff) > 0:
        pair_var = 2 * per_run_var                       # variance of the difference of one run each
        out["runs_per_agent_to_detect_this_difference"] = max(1, math.ceil((Z975 + Z80) ** 2 * pair_var / diff ** 2))
    for gain in (0.03, 0.05, 0.10):
        out[f"runs_per_agent_to_detect_{int(100 * gain)}_points"] = max(
            1, math.ceil((Z975 + Z80) ** 2 * 2 * per_run_var / gain ** 2))
    return out


def render(r: dict) -> str:
    pc = lambda x: f"{100 * x:.1f}%"       # noqa: E731
    pt = lambda x: f"{100 * x:.1f} points"  # noqa: E731
    lines = [
        f"Tasks compared: {r['tasks_compared']}" + (f" ({r['tasks_shared_but_unknown']} shared tasks are not development-set tasks and were left out)" if r["tasks_shared_but_unknown"] else ""),
        f"Agent A: {r['runs_a']} run(s), scores {', '.join(pc(s) for s in r['scores_a'])}, mean {pc(r['mean_a'])}",
        f"Agent B: {r['runs_b']} run(s), scores {', '.join(pc(s) for s in r['scores_b'])}, mean {pc(r['mean_b'])}",
        f"Difference (B - A): {'+' if r['difference_b_minus_a'] >= 0 else ''}{pt(r['difference_b_minus_a'])}",
        "",
        f"Two runs of ONE agent at this score level, on these tasks, are expected to disagree on "
        f"{pc(r['expected_flip_rate_between_two_runs_of_one_agent'])} of the tasks.",
        f"With these numbers of runs, the same agent on both sides would show a difference within "
        f"+/- {pt(r['noise_band_95'])} 95% of the time.",
    ]
    verdict = ("WITHIN the noise band: this difference is what one agent produces against itself."
               if r["within_noise_band"] else "OUTSIDE the noise band: larger than run-to-run noise at the 5% level.")
    lines += [verdict, f"  noise-model p-value: {r['p_value_noise_model']:.3f}"]
    if "p_value_exact_mcnemar" in r:
        lines.append(f"  exact McNemar test (solved only by A: {r['solved_only_by_a']}, only by B: {r['solved_only_by_b']}): "
                     f"p = {r['p_value_exact_mcnemar']:.3f}")
    if "p_value_permutation" in r:
        lines.append(f"  exact permutation test within tasks, no noise model ({r['informative_tasks']} tasks on which the "
                     f"runs differ): p = {r['p_value_permutation']:.3f}")
    lines.append("")
    if "runs_per_agent_to_detect_this_difference" in r:
        lines.append(f"Runs per agent the noise-band test needs to detect a difference of this size with 80% power: "
                     f"{r['runs_per_agent_to_detect_this_difference']}")
    lines.append("Runs per agent it needs on these tasks for a gain of 3 / 5 / 10 points: "
                 f"{r['runs_per_agent_to_detect_3_points']} / {r['runs_per_agent_to_detect_5_points']} / "
                 f"{r['runs_per_agent_to_detect_10_points']}")
    lines.append("(The exact test, which does not rely on the pooled noise model, needs somewhat more; "
                 "see 'What repeats buy' in results/RESULTS.md.)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", nargs="+", required=True, help="result file(s) of agent A, one per run")
    ap.add_argument("--b", nargs="+", required=True, help="result file(s) of agent B, one per run")
    ap.add_argument("--band", default=str(HERE / "noise_band.json"), help="task difficulties and noise scale")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    band = json.loads(Path(args.band).read_text(encoding="utf-8"))
    result = compare([read_run(f) for f in args.a], [read_run(f) for f in args.b], band)
    print(json.dumps(result, indent=1) if args.json else render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
