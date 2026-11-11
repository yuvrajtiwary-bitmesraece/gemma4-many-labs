"""Sensitivity and uncertainty analyses. Added after the first results; not in the plan.

Each block changes one decision or one definition and recomputes the affected
quantity, so a reader can see how much the headline numbers depend on it.

    python scripts/sensitivity.py        -> results/sensitivity.json
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import mlstats as ml  # noqa: E402

ROOT = HERE.parent
SEED = 20261111
N_BOOT = 5000
Z = 1.959964 + 0.841621          # two-sided 5% test, 80% power

# Decisions under test (labels as they appear in data/derived/attempts.csv).
L01_WEAK_REPEATS = {"E037", "E038", "E040", "E049", "E053"}     # merged on pooled reporting, not on a "repeat" statement
L04_EARLY_RUNS = {"lmassaron/flexonafft_v2@core_60", "lmassaron/v10_lean_thinking@core_60",
                  "lmassaron/v11_targeted_playbooks+hints@core_60", "lmassaron/v12_native_discipline+hints@core_60",
                  "lmassaron/v13_claude_clean@core_60", "lmassaron/v13_non_empty_patch+hints@core_60",
                  "lmassaron/v14_heredoc_synthesis@core_60", "lmassaron/v15_surgical_triage@core_60"}


def flip_summary(att: pd.DataFrame) -> dict:
    pairs = ml.replicate_pairs(att, min_shared=10)
    s = ml.summarise_pairs(pairs, "flip_rate")
    return {"n_pairs": s.get("n_pairs", 0), "n_systems": s.get("n_systems"), "n_labs": s.get("n_labs"),
            "pooled": s.get("pooled"), "lab_median": s.get("lab_median"), "lab_mean": s.get("lab_mean")}


def pooled(pairs: pd.DataFrame) -> float:
    return float(np.average(pairs["flip_rate"], weights=pairs["n_shared"]))


def bootstrap_flip(pairs: pd.DataFrame) -> dict:
    """Cluster bootstrap of the pooled flip rate, resampling labs and, separately, systems."""
    rng = np.random.default_rng(SEED)
    out = {}
    for unit in ("lab", "system"):
        groups = {k: g for k, g in pairs.groupby(unit)}
        keys = list(groups)
        num = np.array([float((groups[k]["flip_rate"] * groups[k]["n_shared"]).sum()) for k in keys])
        den = np.array([float(groups[k]["n_shared"].sum()) for k in keys])
        idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
        est = num[idx].sum(1) / den[idx].sum(1)
        out[f"{unit}_clustered"] = {"n_units": len(keys), "ci95": [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))],
                                    "se": float(est.std(ddof=1))}
    per_lab = pairs.groupby(["lab", "system"])["flip_rate"].mean().groupby("lab").mean().to_numpy()
    idx = rng.integers(0, len(per_lab), size=(N_BOOT, len(per_lab)))
    means = per_lab[idx].mean(1)
    out["lab_mean"] = {"estimate": float(per_lab.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
    return out


def runs_needed(flip: float, n_tasks: int, gain: float) -> int:
    """Runs per arm so that a gain is detected with 80% power (normal approximation).

    The mean score difference of two systems run k times each on the same n
    tasks has variance about (flip + gain) / (k n): the flip rate is the
    run-to-run variance of one paired difference, and, as in the planned
    McNemar calculation, an improvement adds its own size to the discordance.
    """
    return int(math.ceil(Z * Z * (flip + gain) / (n_tasks * gain * gain)))


def detectable(flip: float, n_tasks: int, k: int) -> float:
    """Smallest gain detected with 80% power with k runs per arm (same approximation)."""
    a, b, c = n_tasks * k, -Z * Z, -Z * Z * flip
    return float((-b + math.sqrt(b * b - 4 * a * c)) / (2 * a))


def main() -> int:
    att = analyze.load()
    res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    out = {}

    # S1. Uncertainty of the flip rate.
    pairs = ml.replicate_pairs(att, min_shared=10)
    base = flip_summary(att)
    out["S1_flip_rate_uncertainty"] = {"estimate": base, "bootstrap": bootstrap_flip(pairs)}
    lo, hi = out["S1_flip_rate_uncertainty"]["bootstrap"]["lab_clustered"]["ci95"]
    out["S1_flip_rate_uncertainty"]["detectable_on_129_tasks_at_ci_bounds"] = [
        ml.minimum_detectable_difference(129, lo), ml.minimum_detectable_difference(129, hi)]

    # S2. Repeats only where the lab says "repeat": split the two weak groups in L01.
    a2 = att.copy()
    exp = a2["run"].str.extract(r"(E\d+)")[0]
    weak = (a2["lab"] == "lspecian") & exp.isin(L01_WEAK_REPEATS)
    a2.loc[weak, "system"] = a2.loc[weak, "system"] + "|" + exp[weak]
    out["S2_strict_repeats_L01"] = {"runs_reassigned": int(a2.loc[weak, "run"].nunique()), **flip_summary(a2)}

    # S3. Without the eight early runs of L04 that resolve 0 or 1 of 60 tasks.
    a3 = att[~att["run"].isin(L04_EARLY_RUNS)]
    out["S3_without_L04_early_runs"] = {"runs_removed": int(att["run"].isin(L04_EARLY_RUNS).groupby(att["run"]).any().sum()),
                                        "attempts_removed": int(len(att) - len(a3)), **analyze.core(a3)}

    # S4. Without the four runs of L11 rebuilt from published flip lists.
    rebuilt = att["status"].str.contains("rebuilt", na=False)
    a4 = att[~att["run"].isin(set(att.loc[rebuilt, "run"]))]
    out["S4_without_rebuilt_runs_L11"] = {"runs_removed": int(att.loc[rebuilt, "run"].nunique()), **flip_summary(a4)}

    # S5. Only pairs of runs on exactly the same task set (clean repeats).
    sets = att.groupby("run")["task"].apply(frozenset)
    same = pairs[[sets[a] == sets[b] for a, b in zip(pairs["run_a"], pairs["run_b"])]]
    out["S5_pairs_on_identical_task_sets"] = {
        "n_pairs": int(len(same)), "n_labs": int(same["lab"].nunique()), "pooled": pooled(same) if len(same) else None,
        "lab_median": float(same.groupby(["lab", "system"])["flip_rate"].mean().groupby("lab").mean().median()) if len(same) else None,
        "mean_abs_score_difference": float(same["abs_score_diff"].mean()) if len(same) else None}

    # S6. Flip rate by model size (systems with repeats).
    model = att.groupby("system")["model"].first()
    pm = pairs.assign(model=pairs["system"].map(model))
    out["S6_flip_rate_by_model"] = {m: {"n_pairs": int(len(g)), "n_labs": int(g["lab"].nunique()), "pooled": pooled(g)}
                                    for m, g in pm.groupby("model")}

    # S7. Thresholds behind "floor" and "at least 5 labs".
    cons = ml.consensus(ml.lab_task_means(ml.cell_means(att)))
    grid = {}
    for min_labs in (3, 5, 8):
        c = cons[cons["n_labs"] >= min_labs]
        for cut in (0.05, 0.10, 0.20):
            grid[f"min_labs_{min_labs}_floor_{int(100 * cut)}"] = {
                "tasks": int(len(c)), "floor": int((c["d"] <= cut).sum()), "share_floor": float((c["d"] <= cut).mean()),
                "ceiling": int((c["d"] >= 1 - cut).sum())}
    out["S7_floor_thresholds"] = grid

    # S8. Validity by agreement of the four full audits instead of one reference audit.
    ctl = pd.read_csv(ROOT / "data" / "derived" / "controls.csv")
    ref = ctl[(ctl["kind"] == "reference_patch") & (ctl["environment"] == "kaggle_subprocess")]
    wide = ref.pivot_table(index="task", columns="lab", values="value", aggfunc="first")
    full = [lab for lab in wide.columns if wide[lab].notna().sum() == len(wide)]
    votes = wide[full].sum(axis=1)
    fail_all, pass_all = set(votes.index[votes == 0]), set(votes.index[votes == len(full)])
    a8 = att.assign(fail_all=att["task"].isin(fail_all))
    by_lab = a8.groupby("lab")["fail_all"].mean()
    solved = a8[a8["fail_all"]].groupby("lab")["y"].sum()
    never = a8[a8["fail_all"]].groupby("task")["y"].sum()
    c5 = cons[cons["n_labs"] >= 5]
    floor = c5[c5["d"] <= 0.10]
    out["S8_validity_by_four_audits"] = {
        "audits": full, "tasks_failing_in_all": len(fail_all), "tasks_passing_in_all": len(pass_all),
        "share_of_attempts_on_tasks_failing_in_all": float(a8["fail_all"].mean()),
        "lab_share": {"min": float(by_lab.min()), "median": float(by_lab.median()), "max": float(by_lab.max())},
        "labs_attempting": int((by_lab > 0).sum()), "labs_resolving_none": int((solved == 0).sum()),
        "tasks_never_resolved_by_any_agent": int((never == 0).sum()), "tasks_attempted": int(len(never)),
        "floor_tasks": int(len(floor)), "floor_tasks_passing_in_all_audits": int(floor["task"].isin(pass_all).sum()),
        "floor_tasks_failing_in_all_audits": int(floor["task"].isin(fail_all).sum())}

    # S9. What repeats buy: detectable gain with k runs per arm, and runs needed for a target gain.
    flip_all = base["pooled"]
    flip_valid = res["X1_validity"]["valid_tasks_only"]["flip_pooled"]
    n_valid = res["X1_validity"]["status_counts"]["valid"]
    designs = {"all_129_tasks": (129, flip_all), f"valid_{n_valid}_tasks": (n_valid, flip_valid),
               "60_tasks": (60, flip_all), "30_tasks": (30, flip_all)}
    out["S9_repeats"] = {
        "note": "normal approximation; tasks fixed; discordance = flip rate + gain, as in the planned McNemar calculation",
        "flip_rate_used": {k: v[1] for k, v in designs.items()},
        "detectable_gain": {name: {f"k{k}": detectable(flip, n, k) for k in (1, 3, 5, 10)} for name, (n, flip) in designs.items()},
        "runs_per_arm_needed": {name: {f"gain_{int(100 * g)}pt": runs_needed(flip, n, g) for g in (0.03, 0.05, 0.10)}
                                for name, (n, flip) in designs.items()},
        "exact_mcnemar_one_run": {name: ml.minimum_detectable_difference(n, flip) for name, (n, flip) in designs.items()}}

    # S10. Sampling temperature. Most labs sample at 0.2; one lab (L02) documents two settings:
    # its own configurations use 0.7 (submissions/*/configs/sampling.yaml), the third-party
    # configurations it ran use 0.2 (scorer_stack/build_ab.py).
    t07 = ("veljjanoski/v10@", "veljjanoski/v11@", "veljjanoski/v15@", "veljjanoski/e4b_v10@")
    t02 = ("veljjanoski/v19@", "veljjanoski/e4b_bf@", "veljjanoski/e4b_bf_seq@")
    p2 = pm[pm["lab"] == "veljjanoski"].copy()
    p2["temp"] = np.where(p2["system"].str.startswith(t07), "0.7", np.where(p2["system"].str.startswith(t02), "0.2", "unknown"))
    groups = {}
    for (temp, mod), g in p2[p2["temp"] != "unknown"].groupby(["temp", "model"]):
        groups[f"{temp} ({mod})"] = {"n_pairs": int(len(g)), "n_systems": int(g["system"].nunique()), "pooled": pooled(g)}
    out["S10_temperature"] = {"lab": "L02", "groups": groups,
                              "pairs_with_unknown_temperature": int((p2["temp"] == "unknown").sum())}

    # S11 / S12. What the noise means for a leaderboard of about 58 tasks.
    # Two noise levels: the development-set flip rate, and the flip rate implied by the identical
    # resubmissions that labs report (mean absolute difference m of two scores -> sd = m / sqrt(2/pi)).
    from scipy import integrate, stats as st
    n_lb = 58
    e1 = res.get("E1_leaderboard", {})
    m = (e1.get("identical_resubmissions") or {}).get("abs_difference", {}).get("mean")
    levels = {"development_set_flip_rate": flip_all}
    if m:
        sd_diff = m / math.sqrt(2 / math.pi)
        levels["implied_by_identical_resubmissions"] = sd_diff ** 2 * n_lb

    def p_wrong(n, flip, gain):
        return float(st.norm.cdf(-gain * math.sqrt(n) / math.sqrt(flip + gain)))

    def expected_max(k):
        f = lambda x: x * k * st.norm.pdf(x) * st.norm.cdf(x) ** (k - 1)      # noqa: E731
        return float(integrate.quad(f, -10, 10)[0])

    out["S11_wrong_order"] = {
        "note": "probability that the better of two agents scores lower when each is run once on the same tasks",
        "leaderboard_58_tasks": {name: {"flip_rate": flip, **{f"gain_{int(100 * g)}pt": p_wrong(n_lb, flip, g) for g in (0.02, 0.03, 0.05, 0.10)}}
                                 for name, flip in levels.items()},
        "development_set_129_tasks": {f"gain_{int(100 * g)}pt": p_wrong(129, flip_all, g) for g in (0.02, 0.03, 0.05, 0.10)}}
    out["S12_best_of_n"] = {
        "note": "expected gain in the best score from submitting the same agent N times to a 58-task leaderboard",
        **{name: {"sd_of_one_score": math.sqrt(flip / n_lb / 2),
                  **{f"n{k}": expected_max(k) * math.sqrt(flip / n_lb / 2) for k in (2, 3, 5, 10, 20)}}
           for name, flip in levels.items()}}

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
    (ROOT / "results" / "sensitivity.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(k, json.dumps(v)[:900], flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
