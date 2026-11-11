"""X1: task validity. Exploratory, added after the data were seen (DEVIATIONS.md).

Compares what agents did with the control runs that labs published per task
(data/derived/controls.csv, built by scripts/controls.py): does a task resolve
with its reference patch, and does it resolve with no patch at all?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_AUDIT = "TechNaom|kaggle_subprocess"   # all 129 tasks, both controls, official harness
DOCKER_AUDIT = "SebAustin|docker_per_task"


def e1_leaderboard(path: Path):
    """E1: development-set rate against public-leaderboard score, as the labs state them.

    The input table is compiled by hand from the labs' own documents (file and
    wording are given per row), so this is a description, not a measurement.
    """
    if not path.exists():
        return None
    from scipy import stats
    t = pd.read_csv(path)
    pairs = t[t["kind"] == "pair"].copy()
    pairs["ratio"] = pairs["leaderboard"] / pairs["dev_rate"]
    out = {"n_pairs": int(len(pairs)), "n_labs": int(pairs["lab"].nunique()),
           "pairs_with_leaderboard_below_dev_rate": int((pairs["leaderboard"] < pairs["dev_rate"]).sum()),
           "ratio_leaderboard_to_dev": {"median": float(pairs["ratio"].median()), "q25": float(pairs["ratio"].quantile(0.25)),
                                        "q75": float(pairs["ratio"].quantile(0.75)), "min": float(pairs["ratio"].min()),
                                        "max": float(pairs["ratio"].max())},
           "dev_rate_median": float(pairs["dev_rate"].median()), "leaderboard_median": float(pairs["leaderboard"].median())}
    within = {}
    for (lab, dev_set), g in pairs.groupby(["lab", "dev_set"]):
        if len(g) >= 4 and g["dev_rate"].nunique() > 1 and g["leaderboard"].nunique() > 1:
            within[f"{lab} | {dev_set}"] = {"n_systems": int(len(g)),
                                            "spearman": float(stats.spearmanr(g["dev_rate"], g["leaderboard"]).statistic)}
    out["within_lab_rank_agreement"] = within
    rs = t[t["kind"] == "resubmission"].copy()
    if len(rs):
        rs["abs_diff"] = (rs["leaderboard"] - rs["leaderboard_2"]).abs()
        out["identical_resubmissions"] = {
            "n": int(len(rs)), "n_labs": int(rs["lab"].nunique()),
            "abs_difference": {"median": float(rs["abs_diff"].median()), "mean": float(rs["abs_diff"].mean()),
                               "max": float(rs["abs_diff"].max()), "n_identical": int((rs["abs_diff"] == 0).sum())},
            "scores": [[float(a), float(b)] for a, b in zip(rs["leaderboard"], rs["leaderboard_2"])]}
    return out


def x1_validity(att: pd.DataFrame, cons: pd.DataFrame, codes: dict, controls_path: Path, core, tables: Path):
    """Returns (results dict, per-task audit table, per-lab table) or (None, None, None)."""
    if not controls_path.exists():
        return None, None, None
    c = pd.read_csv(controls_path)
    c["audit"] = c["lab"] + "|" + c["environment"]
    ref = c[c["kind"] == "reference_patch"].pivot_table(index="task", columns="audit", values="value", aggfunc="first")
    emp = c[c["kind"] == "empty_patch"].pivot_table(index="task", columns="audit", values="value", aggfunc="first")
    gold, empty = ref[REFERENCE_AUDIT], emp[REFERENCE_AUDIT]
    status = pd.Series(np.where(empty == 1, "passes_without_patch",
                                np.where(gold == 1, "valid", "reference_patch_fails")), index=ref.index)
    out = {"reference_audit": REFERENCE_AUDIT, "n_tasks": int(len(status)),
           "status_counts": status.value_counts().to_dict(),
           "status_by_repo": {r: g.value_counts().to_dict()
                              for r, g in status.groupby(status.index.str.split("_").str[0])}}

    # Every published audit: how many tasks it covers and how many pass.
    audits = c.groupby(["lab", "environment", "kind"])["value"].agg(tasks="size", positive="sum").reset_index()
    out["audits"] = audits.to_dict(orient="records")

    # Agreement between reference-patch audits run in the same kind of environment.
    same_env = [a for a in ref.columns if a.endswith("|kaggle_subprocess")]
    full = [a for a in same_env if ref[a].notna().sum() == len(ref)]
    pair = {}
    for a in same_env:
        if a == REFERENCE_AUDIT:
            continue
        both = ref[[REFERENCE_AUDIT, a]].dropna()
        diff = both.index[both[REFERENCE_AUDIT] != both[a]]
        pair[a] = {"n_tasks": int(len(both)), "resolves": int(both[a].sum()),
                   "same_verdict_as_reference_audit": int(len(both) - len(diff)), "different_verdict": sorted(diff)}
    out["kaggle_subprocess_audits_against_reference"] = pair
    votes = ref[full].sum(axis=1)
    mixed = (votes > 0) & (votes < len(full))
    out["kaggle_subprocess_full_audits"] = {
        "audits": full, "pass_in_all": int((votes == len(full)).sum()), "fail_in_all": int((votes == 0).sum()),
        "mixed": int(mixed.sum()), "mixed_tasks": sorted(votes.index[mixed])}

    docker = ref[DOCKER_AUDIT] if DOCKER_AUDIT in ref.columns else None
    if docker is not None:
        out["docker_audit_against_reference"] = {
            "audit": DOCKER_AUDIT, "resolves": int(docker.sum()),
            "fails_on_kaggle_but_resolves_in_docker": int(((gold == 0) & (docker == 1)).sum()),
            "fails_in_both": int(((gold == 0) & (docker == 0)).sum()),
            "resolves_on_kaggle_but_fails_in_docker": int(((gold == 1) & (docker == 0)).sum())}
    flags = c[c["kind"] == "flag_missing_dependency"]
    for lab, g in flags.groupby("lab"):
        flagged = g.loc[g["value"] == 1, "task"]
        hit = int((gold.reindex(flagged) == 0).sum())
        out.setdefault("static_flags", {})[lab] = {
            "flagged": int(len(flagged)), "flagged_and_reference_patch_fails": hit,
            "reference_patch_fails_but_not_flagged": int((gold == 0).sum()) - hit}
    out["tasks_passing_without_patch"] = {a: sorted(emp.index[emp[a] == 1]) for a in emp.columns}

    # What agents did on each class of task.
    fine = status.copy()
    if docker is not None:
        fails = fine == "reference_patch_fails"
        fine[fails & (docker == 1)] = "reference_patch_fails_on_kaggle_only"
        fine[fails & (docker == 0)] = "reference_patch_fails_in_both"
    a = att.assign(status=att["task"].map(status), fine=att["task"].map(fine))
    for key, col, labels in (("agent_attempts_by_status", "status", status),
                             ("agent_attempts_by_status_with_docker_audit", "fine", fine)):
        by = a.groupby(col)["y"].agg(attempts="size", resolved="sum")
        by["rate"] = by["resolved"] / by["attempts"]
        by["tasks_attempted"] = a.groupby(col)["task"].nunique()
        by["labs"] = a.groupby(col)["lab"].nunique()
        by["tasks"] = labels.value_counts()
        out[key] = by.to_dict(orient="index")
    out["share_of_attempts_on_tasks_where_reference_patch_fails"] = float((a["status"] == "reference_patch_fails").mean())
    lab_share = a.assign(bad=a["status"] == "reference_patch_fails").groupby("lab")["bad"].mean()
    out["lab_share_of_attempts_on_reference_patch_fail_tasks"] = {
        "min": float(lab_share.min()), "median": float(lab_share.median()), "max": float(lab_share.max()),
        "n_labs": int(len(lab_share)), "n_labs_zero": int((lab_share == 0).sum()),
        "n_labs_above_20_percent": int((lab_share > 0.20).sum())}
    bad = a[a["status"] == "reference_patch_fails"].groupby("lab")["y"].agg(attempts="size", resolved="sum")
    out["labs_attempting_reference_patch_fail_tasks"] = int(len(bad))
    out["labs_resolving_none_of_them"] = int((bad["resolved"] == 0).sum())
    out["labs_resolving_some_of_them"] = {
        codes[lab]: {"attempts": int(r["attempts"]), "resolved": int(r["resolved"]),
                     "rate": float(r["resolved"] / r["attempts"])}
        for lab, r in bad[bad["resolved"] > 0].iterrows()}
    per_task = a[a["status"] == "reference_patch_fails"].groupby("task")["y"].agg(attempts="size", resolved="sum")
    out["reference_patch_fail_tasks_attempted"] = int(len(per_task))
    out["reference_patch_fail_tasks_never_resolved_by_any_agent"] = int((per_task["resolved"] == 0).sum())

    # Consensus class against audit status (tasks with at least 5 labs).
    c5 = cons[cons["n_labs"] >= 5].copy()
    c5["class"] = np.where(c5["d"] <= 0.10, "floor", np.where(c5["d"] >= 0.90, "ceiling", "discriminating"))
    c5["status"] = c5["task"].map(fine)
    out["consensus_class_by_status"] = {k: v.to_dict() for k, v in pd.crosstab(c5["class"], c5["status"]).iterrows()}
    out["mean_consensus_by_status"] = c5.groupby("status")["d"].mean().to_dict()
    floor = c5[c5["class"] == "floor"]
    out["floor_tasks_that_are_valid_in_reference_audit"] = int((floor["status"] == "valid").sum())
    out["floor_tasks"] = int(len(floor))

    # The headline quantities on the tasks the reference audit found valid.
    valid_tasks = set(status[status == "valid"].index)
    out["valid_tasks_only"] = core(att[att["task"].isin(valid_tasks)])

    audit = pd.DataFrame({"task": status.index, "audit_status": fine.to_numpy(),
                          "reference_patch_resolves": gold.to_numpy(), "empty_patch_resolves": empty.to_numpy(),
                          "kaggle_audits_passing": votes.to_numpy(), "kaggle_audits": len(full)})
    if docker is not None:
        audit["reference_patch_resolves_docker"] = docker.to_numpy()
    lab_tab = a.groupby(["lab", "fine"])["y"].agg(attempts="size", resolved="sum").reset_index()
    lab_tab["rate"] = lab_tab["resolved"] / lab_tab["attempts"]
    lab_tab["lab"] = lab_tab["lab"].map(codes)
    lab_tab = lab_tab.rename(columns={"fine": "audit_status"}).sort_values(["lab", "audit_status"])
    audits.to_csv(tables / "audits.csv", index=False, lineterminator="\n")
    return out, audit, lab_tab
