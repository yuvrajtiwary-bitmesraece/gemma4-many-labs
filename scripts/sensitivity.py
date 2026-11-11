"""Sensitivity and uncertainty analyses. Added after the first results; not in the plan.

Each block changes one decision or one definition and recomputes the affected
quantity, so a reader can see how much the headline numbers depend on it.
Blocks S1, S8, S9 and S11 were corrected after an independent statistical
review (DEVIATIONS.md, Amendment 4).

    python scripts/sensitivity.py        -> results/sensitivity.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import mlstats as ml  # noqa: E402

ROOT = HERE.parent
SEED = 20261111
N_BOOT = 5000
Z = ml.Z_80                      # two-sided 5% test, 80% power

# Decisions under test (labels as they appear in data/derived/attempts.csv).
L01_WEAK_REPEATS = {"E037", "E038", "E040"}     # merged on a pooled percentage, not on a "repeat" statement
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


def jackknife_flip(pairs: pd.DataFrame) -> dict:
    """Delete-one-lab jackknife of the pooled flip rate, and a t interval on the lab-level rates.

    With nine labs of very unequal size (two carry most of the pooled weight) both intervals for
    the POOLED rate are too narrow: see coverage_simulation.  The t interval on the lab-level
    rates, where every lab counts once, is the one reported first.
    """
    num = (pairs["flip_rate"] * pairs["n_shared"]).groupby(pairs["lab"]).sum()
    den = pairs["n_shared"].groupby(pairs["lab"]).sum()
    g = len(num)
    full = float(num.sum() / den.sum())
    loo = np.array([(num.sum() - num[lab]) / (den.sum() - den[lab]) for lab in num.index])
    se = float(np.sqrt((g - 1) / g * np.sum((loo - loo.mean()) ** 2)))
    tq = float(st.t.ppf(0.975, g - 1))
    per_lab = pairs.groupby(["lab", "system"])["flip_rate"].mean().groupby("lab").mean().to_numpy()
    se_lab = float(per_lab.std(ddof=1) / np.sqrt(len(per_lab)))
    w = (den / den.sum()).to_numpy()
    return {"n_labs": g, "jackknife_se": se, "t_quantile": tq,
            "jackknife_ci95": [full - tq * se, full + tq * se],
            "lab_level_mean": float(per_lab.mean()),
            "lab_level_t_ci95": [float(per_lab.mean() - tq * se_lab), float(per_lab.mean() + tq * se_lab)],
            "effective_number_of_labs": float(1 / np.sum(w ** 2)),
            "share_of_weight_two_largest_labs": float(np.sort(w)[-2:].sum()),
            "leave_one_lab_out_range": [float(loo.min()), float(loo.max())]}


def coverage_simulation(pairs: pd.DataFrame, n_sim: int = 2000) -> dict:
    """How often do the two intervals cover the truth when labs differ?  Labs keep their observed
    weights; a lab's flips are binomial around a lab-specific rate drawn around the pooled one.
    The target is the rate of the population the labs are drawn from."""
    rng = np.random.default_rng(SEED + 1)
    den = pairs["n_shared"].groupby(pairs["lab"]).sum().to_numpy(float)
    # Repeated runs of one system overlap, so the tasks are not independent trials; count each
    # system's distinct shared tasks once as a rough effective number of trials per lab.
    eff = pairs.groupby(["lab", "system"])["n_shared"].max().groupby("lab").sum().to_numpy(float)
    g = len(den)
    truth = pooled(pairs)
    tq = st.t.ppf(0.975, g - 1)
    out = {}
    for sd in (0.02, 0.04):
        hit_b = hit_j = hit_l = 0
        for _ in range(n_sim):
            rate = np.clip(rng.normal(truth, sd, size=g), 0.005, 0.6)
            lab_rate = rng.binomial(eff.astype(int), rate) / eff
            flips = lab_rate * den                                          # scaled to the pooled weights
            target = truth
            est = flips.sum() / den.sum()
            idx = rng.integers(0, g, size=(400, g))
            b = flips[idx].sum(1) / den[idx].sum(1)
            hit_b += np.percentile(b, 2.5) <= target <= np.percentile(b, 97.5)
            loo = (flips.sum() - flips) / (den.sum() - den)
            se = np.sqrt((g - 1) / g * np.sum((loo - loo.mean()) ** 2))
            hit_j += est - tq * se <= target <= est + tq * se
            half = tq * lab_rate.std(ddof=1) / np.sqrt(g)                   # every lab with equal weight
            hit_l += lab_rate.mean() - half <= target <= lab_rate.mean() + half
        out[f"between_lab_sd_{int(100 * sd)}pt"] = {"pooled_percentile_bootstrap": hit_b / n_sim,
                                                    "pooled_jackknife_t": hit_j / n_sim,
                                                    "lab_level_mean_t": hit_l / n_sim}
    return out


def main() -> int:
    att = analyze.load()
    res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    out = {}

    # S1. Uncertainty of the flip rate.
    pairs = ml.replicate_pairs(att, min_shared=10)
    base = flip_summary(att)
    jk = jackknife_flip(pairs)
    out["S1_flip_rate_uncertainty"] = {"estimate": base, "jackknife": jk, "bootstrap": bootstrap_flip(pairs),
                                       "coverage_simulation": coverage_simulation(pairs)}
    # The interval reported first is the t interval on the lab-level rates (every lab with equal weight):
    # it is the only one of the three whose coverage is close to nominal in the simulation above.
    lo, hi = jk["lab_level_t_ci95"]
    out["S1_flip_rate_uncertainty"]["detectable_on_129_tasks_at_ci_bounds"] = [
        ml.minimum_detectable_difference(129, lo), ml.minimum_detectable_difference(129, hi)]

    # S2. Repeats only on the strongest evidence. Every group that an independent audit graded as resting on
    # weaker evidence is split: in L01 one group pooled in the lab's reporting but never called repeats; in L02
    # one pair that shares only a label (v19) and one group pooled across two machines (e4b_bf).
    a2 = att.copy()
    exp = a2["run"].str.extract(r"(E\d+)")[0]
    weak = (a2["lab"] == "lspecian") & exp.isin(L01_WEAK_REPEATS)
    a2.loc[weak, "system"] = a2.loc[weak, "system"] + "|" + exp[weak]
    v19 = a2["system"] == "veljjanoski/v19@kaggle"
    a2.loc[v19, "system"] = a2.loc[v19, "run"]
    bf = a2["system"] == "veljjanoski/e4b_bf@vllm"
    a2.loc[bf, "system"] = np.where(a2.loc[bf, "run"].str.contains("scorer_stack"), "veljjanoski/e4b_bf@kaggle",
                                    "veljjanoski/e4b_bf@local-vllm")
    n_re = int(a2.loc[weak | v19 | bf, "run"].nunique())
    out["S2_strict_repeats_L01"] = {"runs_reassigned": n_re, **flip_summary(a2)}

    # S2b. Without repeats on task subsets that the lab chose from earlier outcomes. Such subsets are picked
    # for unstable, for failed or for previously solved tasks, so the disagreement between repeats on them is
    # not typical.  The list is analyze.outcome_selected_runs.
    sel_runs = analyze.outcome_selected_runs(att)
    hit = pairs["run_a"].isin(sel_runs) | pairs["run_b"].isin(sel_runs)
    keep_pairs = pairs[~hit]
    per_lab = keep_pairs.groupby(["lab", "system"])["flip_rate"].mean().groupby("lab").mean()
    out["S2b_without_outcome_selected_subsets"] = {
        "runs_flagged": len(sel_runs), "pairs_removed": int(hit.sum()), "n_pairs": int(len(keep_pairs)),
        "n_labs": int(keep_pairs["lab"].nunique()), "pooled": pooled(keep_pairs), "lab_median": float(per_lab.median()),
        "lab_q25": float(per_lab.quantile(0.25)), "lab_q75": float(per_lab.quantile(0.75)), "lab_max": float(per_lab.max()),
        "pooled_on_the_removed_pairs": pooled(pairs[hit]) if hit.any() else None}

    # S3. Without the eight early runs of L04 that resolve none of 60 tasks (scored in a test environment
    # that could not import the packages of 37 of the 60 tasks; see the card of the adapter).
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

    # S8. Task validity from ONE audit (the definition used before Amendment 4) instead of the agreement of
    # the four full audits: the audit that ran both controls on all 129 tasks.
    tasks = pd.read_csv(ROOT / "results" / "tables" / "tasks.csv")
    single = tasks.set_index("task")["single_audit_status"]
    a8 = att.assign(st=att["task"].map(single))
    bad = a8["st"] == "reference_patch_fails"
    by_lab = bad.groupby(a8["lab"]).mean()
    solved = a8[bad].groupby("lab")["y"].sum()
    never = a8[bad].groupby("task")["y"].sum()
    c5 = cons[cons["n_labs"] >= 5]
    floor = c5[c5["d"] <= 0.10]
    out["S8_validity_from_one_audit"] = {
        "status_counts": single.value_counts().to_dict(),
        "tasks_with_a_different_status": int(res["X1_validity"]["single_audit_definition"]["tasks_with_a_different_status"]),
        "share_of_attempts_on_tasks_where_reference_patch_fails": float(bad.mean()),
        "lab_share": {"min": float(by_lab.min()), "median": float(by_lab.median()), "max": float(by_lab.max())},
        "labs_attempting": int((by_lab > 0).sum()), "labs_resolving_none": int((solved == 0).sum()),
        "tasks_never_resolved_by_any_agent": int((never == 0).sum()), "tasks_attempted": int(len(never)),
        "floor_tasks": int(len(floor)), "floor_tasks_valid": int((floor["task"].map(single) == "valid").sum()),
        "valid_tasks_only": analyze.core(att[att["task"].map(single) == "valid"])}

    # S9. What repeats buy.  Two questions that must be kept apart:
    #  (a) tasks fixed: is the new agent better ON THESE TASKS?  Run-to-run noise is the only error, and it
    #      shrinks with the number of runs.  Exact task-stratified permutation test (exact McNemar for one
    #      run each), planned alternative (the gain is tasks that go from never to always solved).
    #  (b) tasks random: will the gain hold ON NEW TASKS of the same kind (a hidden test set)?  The system x
    #      task term does not shrink with more runs.  From the within-lab variance components of P2b.
    flip_all = base["pooled"]
    x1 = res["X1_validity"]
    flip_valid = x1["valid_tasks_only"]["flip_pooled"]
    n_valid = x1["status_counts"]["valid"]
    designs = {"all_129_tasks": (129, flip_all), f"valid_{n_valid}_tasks": (n_valid, flip_valid),
               "60_tasks": (60, flip_all), "30_tasks": (30, flip_all)}
    fixed, approx, need = {}, {}, {}
    for name, (n, flip) in designs.items():
        fixed[name] = {"k1": ml.minimum_detectable_difference(n, flip)}
        for k in (2, 3, 5, 10):
            fixed[name][f"k{k}"] = ml.detectable_exact_fixed_tasks(flip, n, k, n_sim=3000, seed=SEED)
        approx[name] = {f"k{k}": ml.detectable_fixed_tasks(flip, n, k) for k in (1, 2, 3, 5, 10)}
        need[name] = {}
        for g in (0.03, 0.05, 0.10):
            if fixed[name]["k1"] <= g:
                need[name][f"gain_{int(100 * g)}pt"] = 1
            else:
                # the normal approximation is slightly optimistic for two runs, so check k = 2 exactly
                k = ml.runs_needed_fixed_tasks(flip, n, g)
                if k == 2 and fixed[name]["k2"] > g:
                    k = 3
                need[name][f"gain_{int(100 * g)}pt"] = k
    gs = pd.read_csv(ROOT / "results" / "tables" / "g_study.csv")
    new_tasks, need_new = {}, {}
    for n in (58, 129):
        new_tasks[f"{n}_tasks"] = {}
        for kname in ("1", "3", "5", "10", "inf"):
            v = gs[f"detect_n{n}_k{kname}"]
            new_tasks[f"{n}_tasks"][f"k{kname}"] = {"median": float(v.median()), "min": float(v.min()), "max": float(v.max()),
                                                    "by_lab": {c: float(x) for c, x in zip(gs["code"], v)}}
        need_new[f"{n}_tasks"] = {}
        for g in (0.03, 0.05, 0.10):
            runs = {}
            for _, row in gs.iterrows():
                room = n * g * g / (2 * Z * Z) - row["var_system_x_task"]
                runs[row["code"]] = None if room <= 0 else int(math.ceil(row["var_replicate"] / room - 1e-9)) or 1
            need_new[f"{n}_tasks"][f"gain_{int(100 * g)}pt"] = {
                "runs_by_lab": runs, "labs_where_no_number_of_runs_is_enough": int(sum(v is None for v in runs.values())),
                "n_labs": int(len(runs))}
    out["S9_repeats"] = {
        "note": "fixed tasks: exact stratified permutation test, alternative = tasks that go from never to always "
                "solved; k = 1 is the exact McNemar value of P3, k >= 2 by simulation (3,000 draws). new tasks: "
                "2.80 x sqrt(2 (var_system_x_task + var_run / k) / n) from the variance components of P2b",
        "flip_rate_used": {k: v[1] for k, v in designs.items()},
        "detectable_gain_fixed_tasks": fixed,
        "detectable_gain_fixed_tasks_normal_approximation": approx,
        "runs_per_arm_needed_fixed_tasks": need,
        "detectable_gain_new_tasks": new_tasks,
        "runs_per_arm_needed_new_tasks": need_new,
        "labs_with_variance_components": int(len(gs))}

    # S10. Sampling temperature. Most labs sample at 0.2; one lab (L02) documents two settings:
    # its own configurations use 0.7 (submissions/*/configs/sampling.yaml), the third-party
    # configurations it ran use 0.2 (scorer_stack/build_ab.py).
    t07 = ("veljjanoski/v10@", "veljjanoski/v11@", "veljjanoski/v15@", "veljjanoski/e4b_v10@")
    t02 = ("veljjanoski/v19@", "veljjanoski/e4b_bf@", "veljjanoski/e4b_bf_seq@", "veljjanoski/e4b_loc@")
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
    n_lb = 58
    e1 = res.get("E1_leaderboard", {})
    m = (e1.get("identical_resubmissions") or {}).get("abs_difference", {}).get("mean")
    levels = {"development_set_flip_rate": flip_all}
    if m:
        sd_diff = m / math.sqrt(2 / math.pi)
        levels["implied_by_identical_resubmissions"] = sd_diff ** 2 * n_lb

    def p_wrong(n, flip, gain):
        # One run each on the same n tasks: the score difference has mean `gain` and variance flip / n
        # (sum over tasks of p(1-p) + q(1-q); the gain itself adds no sampling variance).
        return float(st.norm.cdf(-gain * math.sqrt(n / flip)))

    def expected_max(k):
        f = lambda x: x * k * st.norm.pdf(x) * st.norm.cdf(x) ** (k - 1)      # noqa: E731
        return float(integrate.quad(f, -10, 10)[0])

    out["S11_wrong_order"] = {
        "note": "probability that the better of two agents scores lower when each is run once on the same tasks; "
                "normal approximation, variance of the difference = flip rate / n",
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
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    out = clean(out)
    (ROOT / "results" / "sensitivity.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(k, json.dumps(v)[:1200], flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
