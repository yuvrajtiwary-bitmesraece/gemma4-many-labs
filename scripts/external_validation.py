"""External validation on a dataset outside the source frame. Not pooled.

Zenodo record 23834104 (doi:10.5281/zenodo.23834104, Apache-2.0): run records
of a pre-registered factorial by another participant, three Gemma 4 sizes x
scaffold levels x repeated seeds on 104 development-set tasks. The record was
not returned by the searches of ANALYSIS_PLAN.md section 3 and its author owns
none of the pooled sources, so it was not used for any estimate. Here it is
used to test the pooled estimates: does task difficulty carry over, and how
often do repeated runs disagree in an independent lab?

Input: data/derived/external_zenodo_attempts.csv (scripts/external_zenodo.py)
and the record's own reference-patch / empty-patch check of all 129 tasks.

    python scripts/external_validation.py   -> results/external_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import mlstats as ml  # noqa: E402

ROOT = HERE.parent
EXT = ROOT / "data" / "derived" / "external_zenodo_attempts.csv"
GOLD = ROOT / "data" / "derived" / "external_zenodo_controls.csv"    # written by external_zenodo.py


def main() -> int:
    ext = analyze.load(EXT)
    res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    tasks = pd.read_csv(ROOT / "results" / "tables" / "tasks.csv")
    out = {"source": "zenodo:23834104", "doi": "10.5281/zenodo.23834104", "licence": "Apache-2.0",
           "n_attempts": int(len(ext)), "n_systems": int(ext["system"].nunique()), "n_runs": int(ext["run"].nunique()),
           "n_tasks": int(ext["task"].nunique()), "resolve_rate": float(ext["y"].mean()),
           "sampling_temperature": 1.0}

    # Repeats: flip rate between seeds of one system.
    pairs = ml.replicate_pairs(ext, min_shared=10)
    model = ext.groupby("system")["model"].first()
    pairs["model"] = pairs["system"].map(model)
    out["flip_rate"] = {"n_pairs": int(len(pairs)), "pooled": float(np.average(pairs["flip_rate"], weights=pairs["n_shared"])),
                        "mean_abs_score_difference": float(pairs["abs_score_diff"].mean()),
                        "max_abs_score_difference": float(pairs["abs_score_diff"].max()),
                        "by_model": {m: float(np.average(g["flip_rate"], weights=g["n_shared"])) for m, g in pairs.groupby("model")},
                        "by_system": {s.split("/", 1)[1]: float(g["flip_rate"].mean()) for s, g in pairs.groupby("system")}}
    noise = ml.replicate_noise(ext)
    out["replicate_noise"] = noise
    out["pooled_estimates_for_comparison"] = {
        "flip_rate": res["P3"]["flip_rate"]["pooled"], "noise_share": res["P2"]["replicate_noise_unweighted"]["noise_share"]}

    # Balanced block: the six confirmatory systems (min and max), five seeds each.
    conf = ext[~ext["system"].str.endswith("-mid")]
    blk = ml.balanced_block(conf)
    if blk is not None:
        g = ml.g_study(blk[0])
        out["g_study"] = {"n_systems": g["n_systems"], "n_tasks": g["n_tasks"], "n_replicates": g["n_replicates"],
                          "shares": g["shares"], "g_129_tasks_1_run": ml.g_coefficient(g["components"], 129, 1),
                          "g_129_tasks_3_runs": ml.g_coefficient(g["components"], 129, 3),
                          "se_difference_129_tasks_1_run": ml.se_difference(g["components"], 129, 1)}
        out["pooled_estimates_for_comparison"]["g_study_median_shares"] = {
            k.replace("share_", ""): v["median"] for k, v in res["P2b"].items() if k.startswith("share_")}

    # Does pooled task difficulty carry over?  Per-task solve rate here against the pooled consensus.
    cell = ext.groupby(["system", "task"])["y"].mean().reset_index()
    cell["model"] = cell["system"].map(model)
    rate_all = cell.groupby("task")["y"].mean().rename("ext_rate")
    m = tasks.merge(rate_all, left_on="task", right_index=True)
    m5 = m[m["n_labs"] >= 5]
    out["difficulty_carries_over"] = {
        "n_tasks_compared": int(len(m5)),
        "spearman_all_systems": float(stats.spearmanr(m5["d"], m5["ext_rate"]).statistic),
        "pearson_all_systems": float(np.corrcoef(m5["d"], m5["ext_rate"])[0, 1]),
        "by_model": {}}
    for mod, gm in cell.groupby("model"):
        r = gm.groupby("task")["y"].mean().rename("r")
        mm = m5.merge(r, left_on="task", right_index=True)
        out["difficulty_carries_over"]["by_model"][mod] = float(stats.spearmanr(mm["d"], mm["r"]).statistic)
    keep = [c for c in ("task", "d", "n_labs", "class", "audit_status", "ext_rate") if c in m5.columns]
    m5[keep].sort_values("d").to_csv(ROOT / "results" / "tables" / "external_task_rates.csv", index=False, lineterminator="\n")
    cls = m5.groupby("class")["ext_rate"].agg(["size", "mean"])
    out["external_rate_by_pooled_class"] = {k: {"tasks": int(v["size"]), "mean_external_solve_rate": float(v["mean"])}
                                            for k, v in cls.iterrows()}
    floor = m5[m5["class"] == "floor"]
    out["pooled_floor_tasks_in_external_set"] = {
        "tasks": int(len(floor)), "never_resolved_externally": int((floor["ext_rate"] == 0).sum()),
        "resolved_under_10_percent_externally": int((floor["ext_rate"] <= 0.10).sum())}

    # Tasks by their status in the pooled study's reference audit (standard Kaggle sandbox).
    if "audit_status" in m.columns:
        by = m.groupby("audit_status")["ext_rate"].agg(["size", "mean"])
        out["external_rate_by_audit_status"] = {k: {"tasks": int(v["size"]), "mean_external_solve_rate": float(v["mean"])}
                                                for k, v in by.iterrows()}

    # The record's own reference-patch / empty-patch check of all 129 tasks (repaired Docker environment).
    if GOLD.exists():
        gv = pd.read_csv(GOLD).rename(columns={"reference_patch_resolves": "ext_gold", "empty_patch_resolves": "ext_empty"})
        a = tasks.merge(gv, on="task")
        audit = {"tasks": int(len(gv)), "reference_patch_resolves": int(gv["ext_gold"].sum()),
                 "empty_patch_resolves": int(gv["ext_empty"].sum()),
                 "scorable": int(((gv["ext_gold"] == 1) & (gv["ext_empty"] == 0)).sum())}
        if "reference_patch_resolves_docker" in a.columns:
            audit["same_verdict_as_pooled_docker_audit"] = int((a["ext_gold"] == a["reference_patch_resolves_docker"]).sum())
        if "stock_sandbox_verdict" in a.columns:
            # 1 / 0 where the four full audits of the stock sandbox agree, empty where they do not.
            agreed = a[a["stock_sandbox_verdict"].notna()]
            audit["tasks_with_unanimous_stock_sandbox_verdict"] = int(len(agreed))
            audit["same_verdict_as_stock_sandbox_audits"] = int((agreed["ext_gold"] == agreed["stock_sandbox_verdict"]).sum())
            audit["fails_in_stock_sandbox_audits"] = int((agreed["stock_sandbox_verdict"] == 0).sum())
            audit["fails_in_kaggle_audit_but_resolves_here"] = int(((agreed["stock_sandbox_verdict"] == 0) & (agreed["ext_gold"] == 1)).sum())
        out["reference_patch_check"] = audit

    # Table for the report.
    runs = ext.groupby(["system", "run"])["y"].sum().reset_index()
    rows_out = []
    for system, g in runs.groupby("system"):
        label = system.split("/", 1)[1]
        rows_out.append([label, model[system], len(g), ", ".join(str(int(v)) for v in g["y"]),
                         f"{100 * out['flip_rate']['by_system'][label]:.1f}%"])
    out["header"] = ["System (size and scaffold level)", "Model", "Seeds", "Resolved of 104, per seed", "Flip rate between seeds"]
    out["rows"] = rows_out
    d, f = out["difficulty_carries_over"], out["flip_rate"]
    pe = out["pooled_estimates_for_comparison"]
    cls_txt = "; ".join(f"{k} {100 * v['mean_external_solve_rate']:.1f}% ({v['tasks']} tasks)"
                        for k, v in sorted(out["external_rate_by_pooled_class"].items()))
    fl = out["pooled_floor_tasks_in_external_set"]
    summary = (
        f"Zenodo record 23834104 (doi:10.5281/zenodo.23834104, Apache-2.0) holds the run records of a pre-registered study "
        f"by a participant whose repository was not among the candidates: {out['n_attempts']:,} attempts, "
        f"{out['n_systems']} systems (Gemma 4 E4B, 12B and 26B-A4B at two or three scaffold levels), {out['n_runs']} runs with "
        f"3 or 5 seeds per system, on {out['n_tasks']} development-set tasks, scored by the competition harness in a repaired "
        f"Docker environment, sampling temperature 1.0. It was not used for any pooled estimate.\n\n"
        f"- **Task difficulty carries over.** Across the {d['n_tasks_compared']} tasks that are in both sets and were "
        f"attempted by at least 5 pooled labs, the per-task solve rate in this dataset correlates with the pooled consensus "
        f"solve rate at Spearman {d['spearman_all_systems']:.2f} (by model: " +
        ", ".join(f"{k} {v:.2f}" for k, v in sorted(d["by_model"].items())) + "). "
        f"Mean solve rate here by the pooled class of the task: {cls_txt}. Of the {fl['tasks']} pooled floor tasks in this "
        f"set, {fl['never_resolved_externally']} were never resolved here in any run and {fl['resolved_under_10_percent_externally']} "
        f"were resolved at most 10% of the time.\n"
        f"- **Repeats disagree here too.** {f['n_pairs']} pairs of runs that differ only in the seed: pooled flip rate "
        f"{100 * f['pooled']:.1f}% (by model: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(f["by_model"].items())) +
        f"), against {100 * pe['flip_rate']:.1f}% in the pooled table, where most systems sample at temperature 0.2. Mean "
        f"absolute difference in resolved tasks between two seeds: {100 * f['mean_abs_score_difference']:.1f} points "
        f"(largest {100 * f['max_abs_score_difference']:.1f}). Run-to-run noise is {100 * noise['noise_share']:.1f}% of outcome "
        f"variance here, {100 * pe['noise_share']:.1f}% in the pooled table.")
    if "g_study" in out:
        s = out["g_study"]["shares"]
        summary += (f"\n- **Variance components** on the balanced block of {out['g_study']['n_systems']} systems x "
                    f"{out['g_study']['n_tasks']} tasks x {out['g_study']['n_replicates']} seeds: task {100 * s['task']:.1f}%, "
                    f"system {100 * s['system']:.1f}%, system x task {100 * s['system_x_task']:.1f}%, run-to-run "
                    f"{100 * s['replicate']:.1f}%.")
    if "reference_patch_check" in out:
        a = out["reference_patch_check"]
        summary += (f"\n- **Reference-patch check.** In this author's repaired environment the reference patch resolves "
                    f"{a['reference_patch_resolves']} of {a['tasks']} tasks and the empty patch {a['empty_patch_resolves']}"
                    + (f"; the verdict equals that of the pooled Docker audit on {a['same_verdict_as_pooled_docker_audit']} tasks"
                       if "same_verdict_as_pooled_docker_audit" in a else "")
                    + (f", and {a['fails_in_kaggle_audit_but_resolves_here']} of the {a['fails_in_stock_sandbox_audits']} "
                       f"tasks that fail in every audit of the stock Kaggle sandbox resolve here"
                       if "fails_in_kaggle_audit_but_resolves_here" in a else "") + ".")
    if "external_rate_by_audit_status" in out:
        summary += (" Agents' mean solve rate here by the task's status in the stock sandbox (X1): " +
                    "; ".join(f"{k.replace('_', ' ')} {100 * v['mean_external_solve_rate']:.1f}% ({v['tasks']} tasks)"
                              for k, v in sorted(out["external_rate_by_audit_status"].items())) + ".")
    out["summary"] = summary

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 5)
        if isinstance(x, np.integer):
            return int(x)
        return x

    (ROOT / "results" / "external_validation.json").write_text(json.dumps(clean(out), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
