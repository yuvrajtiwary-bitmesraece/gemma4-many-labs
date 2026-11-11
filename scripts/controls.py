"""Collect the non-agent control results that labs published per task.

Some labs checked the development set itself: they applied the reference
("gold") patch, or no patch at all, and recorded whether the task's tests then
pass in their environment. These are not agent attempts and never enter the
pooled attempts table. They are used only in the exploratory validity analysis
(X1 in scripts/analyze.py).

Output: data/derived/controls.csv with the columns

    lab, source, kind, environment, task, value, file, note

kind is one of
    reference_patch   value 1 if the task resolves with the reference patch
    empty_patch       value 1 if the task resolves with no patch
    flag_missing_dependency
                      value 1 if the lab flags the task's tests as importing a
                      package that the sandbox lacks (static check)

environment is the lab's own description, shortened to one of
    kaggle_subprocess      the competition harness with its subprocess sandbox
                           on Kaggle (or the same set-up run locally) and the
                           packages the environment ships with
    local_subprocess       the same harness and sandbox on the lab's own machine
                           (plus a "diagnostic" variant of it)
    docker_per_task        Docker with dependencies installed per task
    public_docker_kit      the competition's public Docker kit
    scorer_stack           the lab's replica of the scoring stack
    static                 no execution

    python scripts/controls.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harmonize  # noqa: E402
import srcio  # noqa: E402

DERIVED = srcio.ROOT / "data" / "derived"


def technaom(dev):
    """Sandbox audit with the official harness: one line per task and mode."""
    repo = "TechNaom/gemma4-agentic-swe"
    note = "official swegemma harness, subprocess sandbox, Kaggle CPU notebook, competition wheels (audit README)"
    for name, kind in (("gold_runs", "reference_patch"), ("empty_runs", "empty_patch")):
        path = f"experiments/audit-2026-10-15/{name}.jsonl"
        for _, rows in srcio.Source(repo).jsonl(rf"^experiments/audit-2026-10-15/{name}\.jsonl$"):
            for row in rows:
                yield repo, kind, "kaggle_subprocess", row["instance_id"], int(bool(row["resolved"])), path, note


def sebaustin(dev):
    """Reference patch on Kaggle and in Docker with per-task wheels; empty patch in Docker."""
    repo = "SebAustin/gemma4-developer-agent"
    for path, obj in srcio.Source(repo).json(r"^experiments/(oracle_kaggle|oracle_local|empty_local)\.json$"):
        note = obj.get("environment", "")
        if path.endswith("oracle_kaggle.json"):
            reasons = obj.get("failure_reasons", {})
            for task in obj["resolvable"]:
                yield repo, "reference_patch", "kaggle_subprocess", task, 1, path, note
            for task in obj["not_resolvable"]:
                yield repo, "reference_patch", "kaggle_subprocess", task, 0, path, f"{note}; {reasons.get(task, '')}"
        elif path.endswith("oracle_local.json"):
            for key, value in (("resolvable", 1), ("not_resolvable", 0)):
                for task in obj[key]:
                    yield repo, "reference_patch", "docker_per_task", task, value, path, note
        else:
            for key, value in (("passes_with_empty_patch", 1), ("fails_with_empty_patch", 0)):
                for task in obj[key]:
                    yield repo, "empty_patch", "docker_per_task", task, value, path, note


def emamira02(dev):
    """Reference-patch check of all 67 fastapi tasks; the file lists those that resolved."""
    repo = "emamira02/gemma4-swe-agent"
    path = "results/gold_check/pools.json"
    note = ("Kaggle kernel, swegemma subprocess sandbox; summary.json reports 31 of 67 fastapi tasks resolved, "
            "pools.json names the 31, the other 36 are the complement")
    for _, obj in srcio.Source(repo).json(r"^results/gold_check/pools\.json$"):
        solved = {harmonize.canonical_task(t) for t in obj["fastapi/fastapi"]}
        for task in dev:
            if task.startswith("fastapi_"):
                yield repo, "reference_patch", "kaggle_subprocess", task, int(task in solved), path, note


def listed_as_passing(repo, path, note):
    """A file that lists the tasks whose reference patch passes; the rest of the 129 fail."""
    def gen(dev):
        text = srcio.Source(repo).text(path)
        ok = {f"{a.lower()}_{int(b)}" for a, b in harmonize.INVENTORY_ID.findall(text)}
        for task in dev:
            yield repo, "reference_patch", "kaggle_subprocess", task, int(task in ok), path, note
    return gen


def hiapptitude(dev):
    """Single-run labels for 129 tasks on three routes."""
    repo = "HIAPPtitude-owner/gemma-eval-kit"
    path = "labels/dev129_labels.csv"
    env = {"as_shipped": "local_subprocess", "diagnostic": "local_subprocess_diagnostic",
           "public_docker": "public_docker_kit"}
    for _, rows in srcio.Source(repo).csv(r"^labels/dev129_labels\.csv$"):
        for row in rows:
            note = f"route {row['route']} (README: as_shipped and diagnostic are a WSL1 subprocess setup); {row['root_cause']}"
            yield repo, "reference_patch", env[row["route"]], row["task_id"], int(srcio.truthy(row["gold_pass"])), path, note
            yield repo, "empty_patch", env[row["route"]], row["task_id"], int(not srcio.truthy(row["empty_fail"])), path, note


def veljjanoski(dev):
    """Reference and empty patch on the lab's replica of the scoring stack, 65 pre-selected tasks."""
    repo = "veljjanoski/gemma4-developer-agent"
    note = "scorer's stack (4x L4, vLLM, swegemma 0.2.11); the 65 tasks had passed the lab's local reference-patch check"
    for path, rows in srcio.Source(repo).csv(r"^results/scorer_stack/eval_(gold|empty)\.csv$"):
        kind = "reference_patch" if path.endswith("eval_gold.csv") else "empty_patch"
        for row in rows:
            yield repo, kind, "scorer_stack", row["instance_id"], int(srcio.truthy(row["resolved"])), path, note


def haresh276(dev):
    """Static flag: the task's tests import a package missing from the sandbox."""
    repo = "Haresh276/where-the-fix-lives"
    path = "data/features_gemma_own129.csv"
    note = "static heuristic of the lab: test files mention inline_snapshot or dirty_equals"
    for _, rows in srcio.Source(repo).csv(r"^data/features_gemma_own129\.csv$"):
        for row in rows:
            yield repo, "flag_missing_dependency", "static", row["task_id"], int(srcio.truthy(row["needs_missing_pkg"])), path, note


def sumitmehta(dev):
    """Reference fix on the 104 tasks left after the lab's text-based exclusions."""
    repo = "sumitmehta1025/given-a-map"
    path = "analysis/output/task_features.csv"
    note = ("official swegemma 0.2.11, subprocess sandbox, Kaggle 4x L4; 104 tasks tested after the lab's "
            "text-based exclusions (paper 3.2): 54 cannot pass even with the reference fix")
    for _, rows in srcio.Source(repo).csv(r"^analysis/output/task_features\.csv$"):
        for row in rows:
            if row["excluded"] == "not solvable in the sandbox":
                yield repo, "reference_patch", "kaggle_subprocess", row["instance_id"], 0, path, note
            elif row["excluded"] == "":
                yield repo, "reference_patch", "kaggle_subprocess", row["instance_id"], 1, path, note


GENERATORS = [
    technaom, sebaustin, emamira02, hiapptitude, veljjanoski, haresh276, sumitmehta,
    listed_as_passing("matisdsp/gemma-4-developer-agent", "reports/gold_ok.txt",
                      "reports/gold_ok.txt lists the 75 tasks whose reference patch passes in the subprocess sandbox on Kaggle"),
    listed_as_passing("sujith-kamme/gemma-swe-agent", "kaggle/solvable_tasks.json",
                      "kaggle/solvable_tasks.json lists the 75 tasks whose reference patch passes in the Kaggle subprocess sandbox"),
]


def main() -> int:
    dev = (DERIVED / "dev_tasks.txt").read_text(encoding="utf-8").split()
    dev_set = set(dev)
    rows = []
    for gen in GENERATORS:
        for repo, kind, env, task, value, path, note in gen(dev):
            t = harmonize.canonical_task(task)
            if t in dev_set:
                rows.append({"lab": repo.split("/", 1)[0], "source": repo, "kind": kind, "environment": env,
                             "task": t, "value": value, "file": path, "note": note})
    with (DERIVED / "controls.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n = Counter((r["lab"], r["environment"], r["kind"]) for r in rows)
    ok = Counter((r["lab"], r["environment"], r["kind"]) for r in rows if r["value"])
    for key in sorted(n):
        print(f"{key[0]:18s} {key[1]:28s} {key[2]:24s} tasks={n[key]:3d} value=1: {ok[key]:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
