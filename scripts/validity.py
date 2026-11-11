"""X1: task validity. Exploratory, added after the data were seen (DEVIATIONS.md).

Compares what agents did with the control runs that labs published per task
(data/derived/controls.csv, built by scripts/controls.py): does a task resolve
with its reference patch, and does it resolve with no patch at all?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMPTY_AUDIT = "TechNaom|kaggle_subprocess"       # the one audit in the stock sandbox that also ran the no-patch control
DOCKER_AUDIT = "SebAustin|docker_per_task"
# Test environment as documented by the lab (source: the card of its adapter in scripts/adapters/).
ENVIRONMENT_NOTES = {
    "lspecian": "per-task Python environments from its 36th experiment on (the lab reports that the reference patch "
                "passes on 115 of 129 tasks there, against 52 in the stock environment)",
    "veljjanoski": "pinned packages and environment patches on its own rigs (the lab reports 107 of 129 reference "
                   "patches passing locally after its fixes)",
    "lmassaron": "its own scorer in its own Python environment, not the competition harness",
    "yosh95": "the competition harness run off Kaggle in a sandbox the lab provisioned itself",
}


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
           "n_systems": int(pairs.groupby(["lab", "system"]).ngroups),
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
    """Returns (results dict, per-task audit table, per-lab table) or (None, None, None).

    Status of a task in the stock sandbox (Amendment 4): from the agreement of the reference-patch
    audits that cover all 129 tasks in the competition harness's subprocess sandbox on Kaggle.
        passes_without_patch   resolves with no patch in the audit that ran that control there
        valid                  the reference patch resolves in every full audit (and no patch does not)
        reference_patch_fails  the reference patch fails in every full audit
        audits_disagree        anything else
    Before Amendment 4 the status came from one audit (EMPTY_AUDIT below); an independent review
    showed that this audit is the one that disagrees most with the other three.
    """
    if not controls_path.exists():
        return None, None, None
    c = pd.read_csv(controls_path)
    c["audit"] = c["lab"] + "|" + c["environment"]
    ref = c[c["kind"] == "reference_patch"].pivot_table(index="task", columns="audit", values="value", aggfunc="first")
    emp = c[c["kind"] == "empty_patch"].pivot_table(index="task", columns="audit", values="value", aggfunc="first")
    same_env = [a for a in ref.columns if a.endswith("|kaggle_subprocess")]
    full = [a for a in same_env if ref[a].notna().sum() == len(ref)]
    votes = ref[full].sum(axis=1)
    empty = emp[EMPTY_AUDIT]
    unanimous = pd.Series(np.where(votes == len(full), 1.0, np.where(votes == 0, 0.0, np.nan)), index=ref.index)
    status = pd.Series(np.where(empty == 1, "passes_without_patch",
                                np.where(votes == len(full), "valid",
                                         np.where(votes == 0, "reference_patch_fails", "audits_disagree"))),
                       index=ref.index)
    out = {"definition": "agreement of the full reference-patch audits in the stock Kaggle sandbox; "
                         "empty-patch control from " + EMPTY_AUDIT,
           "full_audits": full, "empty_patch_audit": EMPTY_AUDIT, "n_tasks": int(len(status)),
           "status_counts": status.value_counts().to_dict(),
           "status_by_repo": {r: g.value_counts().to_dict()
                              for r, g in status.groupby(status.index.str.split("_").str[0])}}

    # Every published audit: how many tasks it covers and how many pass.
    audits = c.groupby(["lab", "environment", "kind"])["value"].agg(tasks="size", positive="sum").reset_index()
    out["audits"] = audits.to_dict(orient="records")

    # Agreement between the full audits, pair by pair, and who stands alone on the disputed tasks.
    pairwise = {}
    for i, a in enumerate(full):
        for b in full[i + 1:]:
            pairwise[f"{a.split('|')[0]} / {b.split('|')[0]}"] = int((ref[a] == ref[b]).sum())
    mixed = (votes > 0) & (votes < len(full))
    alone = {}
    for a in full:
        others = [x for x in full if x != a]
        o = ref[others].sum(axis=1)
        alone[a.split("|")[0]] = {
            "fails_alone": sorted(ref.index[(ref[a] == 0) & (o == len(others))]),
            "passes_alone": sorted(ref.index[(ref[a] == 1) & (o == 0)])}
    out["kaggle_subprocess_full_audits"] = {
        "audits": full, "pass_in_all": int((votes == len(full)).sum()), "fail_in_all": int((votes == 0).sum()),
        "mixed": int(mixed.sum()), "mixed_tasks": sorted(votes.index[mixed]),
        "pairwise_same_verdict": pairwise,
        "majority": {"pass": int((votes > len(full) / 2).sum()), "fail": int((votes < len(full) / 2).sum()),
                     "tie": int((votes == len(full) / 2).sum())},
        "stands_alone": alone,
        "resolves_per_audit": {a.split("|")[0]: int(ref[a].sum()) for a in full}}
    # Audits of part of the task set against the unanimous verdict.
    partial = {}
    for a in same_env:
        if a in full:
            continue
        both = pd.DataFrame({"a": ref[a], "u": unanimous}).dropna()
        partial[a.split("|")[0]] = {"n_tasks": int(ref[a].notna().sum()), "resolves": int(ref[a].sum()),
                                    "tasks_with_unanimous_verdict": int(len(both)),
                                    "same_verdict": int((both["a"] == both["u"]).sum())}
    out["partial_audits_against_unanimous_verdict"] = partial

    # The status used before Amendment 4: one audit with both controls.
    gold1 = ref[EMPTY_AUDIT]
    single = pd.Series(np.where(empty == 1, "passes_without_patch", np.where(gold1 == 1, "valid", "reference_patch_fails")),
                       index=ref.index)
    out["single_audit_definition"] = {"audit": EMPTY_AUDIT, "status_counts": single.value_counts().to_dict(),
                                      "tasks_with_a_different_status": int((single != status).sum())}

    docker = ref[DOCKER_AUDIT] if DOCKER_AUDIT in ref.columns else None
    if docker is not None:
        out["docker_audit"] = {
            "audit": DOCKER_AUDIT, "resolves": int(docker.sum()),
            "fail_in_all_kaggle_audits_but_resolve_in_docker": int(((votes == 0) & (docker == 1)).sum()),
            "fail_in_all_kaggle_audits_and_in_docker": int(((votes == 0) & (docker == 0)).sum()),
            "pass_in_all_kaggle_audits_but_fail_in_docker": int(((votes == len(full)) & (docker == 0)).sum()),
            "disputed_on_kaggle_and_resolve_in_docker": int((mixed & (docker == 1)).sum())}
    flags = c[c["kind"] == "flag_missing_dependency"]
    for lab, g in flags.groupby("lab"):
        flagged = g.loc[g["value"] == 1, "task"]
        hit = int((votes.reindex(flagged) == 0).sum())
        out.setdefault("static_flags", {})[lab] = {
            "flagged": int(len(flagged)), "flagged_and_reference_patch_fails": hit,
            "reference_patch_fails_but_not_flagged": int((votes == 0).sum()) - hit}
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
    bad_mask = a["status"] == "reference_patch_fails"
    out["share_of_attempts_on_tasks_where_reference_patch_fails"] = float(bad_mask.mean())
    lab_share = a.assign(bad=bad_mask).groupby("lab")["bad"].mean()
    out["lab_share_of_attempts_on_reference_patch_fail_tasks"] = {
        "min": float(lab_share.min()), "median": float(lab_share.median()), "max": float(lab_share.max()),
        "n_labs": int(len(lab_share)), "n_labs_zero": int((lab_share == 0).sum()),
        "n_labs_above_20_percent": int((lab_share > 0.20).sum())}
    bad = a[bad_mask].groupby("lab")["y"].agg(attempts="size", resolved="sum")
    out["labs_attempting_reference_patch_fail_tasks"] = int(len(bad))
    out["labs_resolving_none_of_them"] = int((bad["resolved"] == 0).sum())
    out["labs_resolving_some_of_them"] = {
        codes[lab]: {"attempts": int(r["attempts"]), "resolved": int(r["resolved"]),
                     "rate": float(r["resolved"] / r["attempts"])}
        for lab, r in bad[bad["resolved"] > 0].iterrows()}
    # Two kinds of lab.  Where a lab resolves such tasks, its environment (or its scorer) is not the stock
    # sandbox, and for it the tasks were not unpassable.  The pooled share is the sum of both kinds.
    tot = a.groupby("lab")["y"].agg(attempts="size", resolved="sum")
    groups = {}
    for name, labs in (("labs_that_resolved_none", [lab for lab in tot.index if bad["resolved"].get(lab, 0) == 0]),
                       ("labs_that_resolved_some", [lab for lab in tot.index if bad["resolved"].get(lab, 0) > 0])):
        n_all = int(tot.loc[labs, "attempts"].sum())
        n_bad = int(bad["attempts"].reindex(labs).fillna(0).sum())
        k_bad = int(bad["resolved"].reindex(labs).fillna(0).sum())
        groups[name] = {"labs": sorted(codes[x] for x in labs), "n_labs": len(labs), "attempts": n_all,
                        "attempts_on_reference_patch_fail_tasks": n_bad,
                        "share": n_bad / n_all if n_all else None, "resolved_there": k_bad,
                        "resolve_rate_there": k_bad / n_bad if n_bad else None,
                        "resolve_rate_overall": float(tot.loc[labs, "resolved"].sum() / n_all) if n_all else None}
    out["attempts_on_reference_patch_fail_tasks_by_kind_of_lab"] = groups
    # What the cards of the adapters record about the test environment of the labs that resolve such tasks.
    out["environment_of_labs_that_resolved_some"] = {
        codes[lab]: ENVIRONMENT_NOTES.get(lab, "not documented in the adapter card")
        for lab in sorted(tot.index, key=lambda x: codes[x]) if bad["resolved"].get(lab, 0) > 0}
    top = bad.sort_values("attempts", ascending=False)
    out["two_largest_labs_share_of_those_attempts"] = float(top["attempts"].head(2).sum() / max(int(bad["attempts"].sum()), 1))
    per_task = a[bad_mask].groupby("task")["y"].agg(attempts="size", resolved="sum")
    out["reference_patch_fail_tasks_attempted"] = int(len(per_task))
    out["reference_patch_fail_tasks_never_resolved_by_any_agent"] = int((per_task["resolved"] == 0).sum())
    # Tasks that fail in every Kaggle audit AND in Docker with per-task dependencies, yet are resolved by agents.
    if docker is not None:
        both = a[a["fine"] == "reference_patch_fails_in_both"]
        pt = both.groupby("task")["y"].agg(attempts="size", resolved="sum").sort_values("resolved", ascending=False)
        by_lab = both.groupby("lab")["y"].sum()
        out["fails_in_every_audit_and_in_docker"] = {
            "tasks": int((fine == "reference_patch_fails_in_both").sum()), "attempts": int(len(both)),
            "resolved": int(both["y"].sum()), "rate": float(both["y"].mean()) if len(both) else None,
            "labs_resolving": sorted(codes[x] for x in by_lab.index[by_lab > 0]),
            "tasks_resolved_at_least_once": int((pt["resolved"] > 0).sum()),
            "most_resolved": [[t, int(r["resolved"]), int(r["attempts"])] for t, r in pt.head(4).iterrows()]}
    # The disputed tasks: what agents do on them.
    disp = a[a["task"].isin(votes.index[mixed])].groupby("task")["y"].agg(attempts="size", resolved="sum")
    out["disputed_tasks"] = {t: {"audits_passing": int(votes[t]), "attempts": int(r["attempts"]), "resolved": int(r["resolved"])}
                             for t, r in disp.iterrows()}

    # Consensus class against audit status (tasks with at least 5 labs).
    c5 = cons[cons["n_labs"] >= 5].copy()
    c5["class"] = np.where(c5["d"] <= 0.10, "floor", np.where(c5["d"] >= 0.90, "ceiling", "discriminating"))
    c5["status"] = c5["task"].map(fine)
    c5["coarse"] = c5["task"].map(status)
    out["consensus_class_by_status"] = {k: v.to_dict() for k, v in pd.crosstab(c5["class"], c5["status"]).iterrows()}
    out["mean_consensus_by_status"] = c5.groupby("status")["d"].mean().to_dict()
    floor = c5[c5["class"] == "floor"]
    out["floor_tasks"] = int(len(floor))
    out["floor_tasks_by_status"] = floor["coarse"].value_counts().to_dict()
    out["floor_tasks_that_are_valid"] = int((floor["coarse"] == "valid").sum())
    out["floor_tasks_valid_in_docker_audit"] = int(floor["task"].map(docker).eq(1).sum()) if docker is not None else None

    # The headline quantities on the valid tasks.
    valid_tasks = set(status[status == "valid"].index)
    out["valid_tasks_only"] = core(att[att["task"].isin(valid_tasks)])

    audit = pd.DataFrame({"task": status.index, "audit_status": fine.to_numpy(),
                          "kaggle_audits_passing": votes.to_numpy(), "kaggle_audits": len(full),
                          "stock_sandbox_verdict": unanimous.to_numpy(),
                          "empty_patch_resolves": empty.to_numpy(),
                          "single_audit_status": single.to_numpy()})
    if docker is not None:
        audit["reference_patch_resolves_docker"] = docker.to_numpy()
    lab_tab = a.groupby(["lab", "fine"])["y"].agg(attempts="size", resolved="sum").reset_index()
    lab_tab["rate"] = lab_tab["resolved"] / lab_tab["attempts"]
    lab_tab["lab"] = lab_tab["lab"].map(codes)
    lab_tab = lab_tab.rename(columns={"fine": "audit_status"}).sort_values(["lab", "audit_status"])
    audits.to_csv(tables / "audits.csv", index=False, lineterminator="\n")
    return out, audit, lab_tab
