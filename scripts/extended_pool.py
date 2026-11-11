"""Do the conclusions hold on more sources? Added after the results; not in the plan.

The registered frame is the six repository searches fixed in the plan. After
the results were known, a broader search (scripts/search_extended.py:
further repository searches and code search for task identifiers) found more
repositories with per-task results. They are processed by the same rules and
adapters, kept out of every registered number, and used here only to see
whether the headline quantities move when they are added.

    python scripts/extended_pool.py  -> results/extended_pool.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import mlstats as ml  # noqa: E402

ROOT = HERE.parent


def describe(att: pd.DataFrame) -> dict:
    return {"attempts": int(len(att)), "labs": int(att["lab"].nunique()), "systems": int(att["system"].nunique()),
            "runs": int(att["run"].nunique()), "tasks": int(att["task"].nunique()),
            "systems_with_repeats": int((att.groupby("system")["run"].nunique() >= 2).sum()),
            "full_129_runs": int((att.groupby("run")["task"].nunique() == 129).sum())}


def flip_detail(att: pd.DataFrame) -> dict:
    """Lab-level flip rates with a t interval (every lab counts once), and the pairs that cover most of the set."""
    from scipy import stats
    pairs = ml.replicate_pairs(att, min_shared=10)
    if pairs.empty:
        return {}
    per_lab = pairs.groupby(["lab", "system"])["flip_rate"].mean().groupby("lab").mean()
    g = len(per_lab)
    out = {"flip_lab_mean": float(per_lab.mean()), "flip_lab_min": float(per_lab.min()), "flip_lab_max": float(per_lab.max())}
    if g >= 3:
        half = float(stats.t.ppf(0.975, g - 1) * per_lab.std(ddof=1) / np.sqrt(g))
        out["flip_lab_mean_t_ci95"] = [float(per_lab.mean() - half), float(per_lab.mean() + half)]
    w = pairs["n_shared"].groupby(pairs["lab"]).sum()
    w = w / w.sum()
    out["effective_number_of_labs"] = float(1 / np.sum(w ** 2))
    big = pairs[pairs["n_shared"] >= 100]
    out["pairs_with_100_or_more_shared_tasks"] = {
        "n_pairs": int(len(big)), "n_labs": int(big["lab"].nunique()),
        "pooled": float(np.average(big["flip_rate"], weights=big["n_shared"])) if len(big) else None,
        "mean_abs_score_difference": float(big["abs_score_diff"].mean()) if len(big) else None}
    return out


def main() -> int:
    path = ROOT / "data" / "derived" / "attempts_extended.csv"
    if not path.exists():
        print("no extended pool (data/derived/attempts_extended.csv is missing)")
        return 0
    ext = analyze.load(path)
    reg = ext[ext["frame"] == "registered"]
    new = ext[ext["frame"] == "extended"]
    out = {"registered": {**describe(reg), **analyze.core(reg), **flip_detail(reg)},
           "registered_plus_extended": {**describe(ext), **analyze.core(ext), **flip_detail(ext)},
           "extended_sources_only": describe(new) if len(new) else {}}
    if new["lab"].nunique() >= 4:
        out["extended_sources_only"].update({**analyze.core(new), **flip_detail(new)})
    # Agreement of the two frames on task difficulty: consensus from registered labs against extended-only labs.
    only_new_labs = set(new["lab"]) - set(reg["lab"])
    if len(only_new_labs) >= 2:
        d_reg = ml.consensus(ml.lab_task_means(ml.cell_means(reg))).set_index("task")
        d_new = ml.consensus(ml.lab_task_means(ml.cell_means(new[new["lab"].isin(only_new_labs)]))).set_index("task")
        both = d_reg.add_suffix("_reg").join(d_new.add_suffix("_new"), how="inner")
        both = both[(both["n_labs_reg"] >= 5) & (both["n_labs_new"] >= 2)]
        if len(both) >= 10:
            from scipy import stats
            out["difficulty_registered_vs_new_labs"] = {
                "tasks": int(len(both)), "new_labs": len(only_new_labs),
                "spearman": float(stats.spearmanr(both["d_reg"], both["d_new"]).statistic)}
    pairs_new = ml.replicate_pairs(new, min_shared=10)
    if len(pairs_new):
        out["flip_rate_extended_sources_only"] = {
            "n_pairs": int(len(pairs_new)), "n_labs": int(pairs_new["lab"].nunique()),
            "pooled": float(np.average(pairs_new["flip_rate"], weights=pairs_new["n_shared"]))}
    out["sources_added"] = sorted(new["source"].unique())
    log_path = ROOT / "data" / "derived" / "harmonize_log.json"
    if log_path.exists():
        twins = json.loads(log_path.read_text(encoding="utf-8")).get("extended_pool", {}).get("identical_runs_across_labs", [])
        out["runs_of_different_labs_with_identical_outcomes"] = twins

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

    out = clean(out)
    (ROOT / "results" / "extended_pool.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(k, json.dumps(v)[:600])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
