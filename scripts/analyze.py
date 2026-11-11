"""Run every analysis of ANALYSIS_PLAN.md / DEVIATIONS.md on the pooled table.

Reads data/derived/attempts.csv and writes

    results/results.json          every number quoted in the paper
    results/tables/*.csv          the tables behind the figures

    python scripts/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlstats as ml  # noqa: E402
import validity  # noqa: E402

ROOT = HERE.parent
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
SEED = 20261111
N_SPLIT = 2000
N_BOOT = 2000


def load(path: Path | None = None) -> pd.DataFrame:
    att = pd.read_csv(path or DERIVED / "attempts.csv", keep_default_na=False,
                      dtype={"model": str, "harness": str, "err": str, "status": str})
    att["y"] = att["y"].astype(int)
    att["repo"] = att["task"].str.split("_").str[0]
    return att


def outcome_selected_runs(att: pd.DataFrame) -> set:
    """Runs on task subsets that the lab chose from the outcomes of its own earlier runs.

    Such a subset is picked for failed, for unstable or for previously solved tasks, so a score on it,
    and the disagreement between repeats on it, are not typical of the benchmark.  Each entry is
    documented in the card of its adapter (scripts/adapters/):
      veljjanoski  swing bench: 'chosen after the fact from the runs it now judges' (17 swing + 6 always solved)
      lmassaron    calibration_48: 17 tasks resolved by v21 plus 31 near misses; hill-climb suites climb_*:
                   'Regression Guard tasks (solved by v16) + High-Signal Target Failure tasks'
      lspecian     E092-E094: 23 tasks 'solved by some DeepSeek run and by none of four Gemma runs'
      emamira02    S13: 14 tasks selected from S12 outcomes (11 failures + 3 resolved controls)
    """
    run, lab = att["run"], att["lab"]
    sel = ((lab == "veljjanoski") & run.str.contains("swing")) | \
          ((lab == "lmassaron") & run.str.contains(r"@(?:calibration_48|climb_)")) | \
          ((lab == "lspecian") & run.str.contains(r"E09[234]_")) | \
          ((lab == "emamira02") & run.str.endswith("/S13"))
    return set(run[sel])


def rnd(x, n=4):
    """Round floats inside nested containers for readable JSON."""
    if isinstance(x, dict):
        return {str(k): rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# --------------------------------------------------------------------------
# Descriptives
# --------------------------------------------------------------------------
def lab_codes(att: pd.DataFrame) -> dict:
    """L01, L02, ... by decreasing number of attempts (ties by name)."""
    n = att.groupby("lab").size().reset_index(name="n")
    n = n.sort_values(["n", "lab"], ascending=[False, True])
    return {lab: f"L{i:02d}" for i, lab in enumerate(n["lab"], 1)}


def describe(att: pd.DataFrame, codes: dict):
    runs = att.groupby("run").agg(lab=("lab", "first"), system=("system", "first"), n=("y", "size"),
                                  rate=("y", "mean"), model=("model", "first"), harness=("harness", "first"))
    # Plan, section 6: repeats are averaged within a system, systems within a lab, then labs.
    sys_rate = runs.groupby(["lab", "system"])["rate"].mean()
    rows = []
    for lab, g in att.groupby("lab"):
        r = runs[runs["lab"] == lab]
        per_system_runs = r.groupby("system").size()
        rows.append({
            "code": codes[lab], "lab": lab, "sources": ";".join(sorted(g["source"].unique())),
            "n_systems": g["system"].nunique(), "n_runs": g["run"].nunique(), "n_attempts": len(g),
            "n_tasks": g["task"].nunique(),
            "n_systems_with_replicates": int((per_system_runs >= 2).sum()),
            "models": ";".join(sorted(m for m in g["model"].unique() if m)) or "unknown",
            "harness": ";".join(sorted(h for h in g["harness"].unique() if h)) or "unknown",
            "mean_system_rate": float(sys_rate.loc[lab].mean()),
            "min_system_rate": float(sys_rate.loc[lab].min()),
            "max_system_rate": float(sys_rate.loc[lab].max()),
        })
    labs = pd.DataFrame(rows).sort_values("code")
    task_sets = att.groupby("run")["task"].apply(lambda s: frozenset(s))
    out = {
        "n_attempts": int(len(att)), "n_labs": int(att["lab"].nunique()),
        "n_sources": int(att["source"].nunique()), "n_systems": int(att["system"].nunique()),
        "n_runs": int(att["run"].nunique()), "n_tasks": int(att["task"].nunique()),
        "n_resolved": int(att["y"].sum()), "raw_resolve_rate": float(att["y"].mean()),
        "lab_balanced_resolve_rate": float(labs["mean_system_rate"].mean()),
        "runs_by_model": runs["model"].replace("", "unknown").value_counts().to_dict(),
        "runs_by_harness": runs["harness"].replace("", "unknown").value_counts().to_dict(),
        "attempts_by_model": att["model"].replace("", "unknown").value_counts().to_dict(),
        "attempts_by_harness": att["harness"].replace("", "unknown").value_counts().to_dict(),
        "attempts_by_repo": att["repo"].value_counts().to_dict(),
        "tasks_by_repo": att.drop_duplicates("task")["repo"].value_counts().to_dict(),
        "run_size": {"min": int(runs["n"].min()), "median": float(runs["n"].median()), "max": int(runs["n"].max()),
                     "n_full_129": int((runs["n"] == 129).sum())},
        "n_distinct_task_sets": int(task_sets.nunique()),
        "n_systems_with_replicates": int(labs["n_systems_with_replicates"].sum()),
        "n_labs_with_replicates": int((labs["n_systems_with_replicates"] > 0).sum()),
        "labs_per_task": {"min": int(att.groupby("task")["lab"].nunique().min()),
                          "median": float(att.groupby("task")["lab"].nunique().median()),
                          "max": int(att.groupby("task")["lab"].nunique().max())},
    }
    return out, labs, runs


# --------------------------------------------------------------------------
# P1  Consensus on task difficulty
# --------------------------------------------------------------------------
def p1(att: pd.DataFrame, n_iter: int = N_SPLIT, with_pairs: bool = True):
    cells = ml.cell_means(att)
    lt = ml.lab_task_means(cells)
    cons = ml.consensus(lt)
    mat = ml.lab_task_matrix(lt)
    rel = ml.split_half_reliability(mat, n_iter=n_iter, seed=SEED)
    c5 = cons[cons["n_labs"] >= 5]
    out = {
        "n_labs": int(mat.shape[0]), "n_tasks_any": int(mat.shape[1]), "n_tasks_5_labs": int(len(c5)),
        "split_half": rel,
        "floor_tasks": int((c5["d"] <= 0.10).sum()), "ceiling_tasks": int((c5["d"] >= 0.90).sum()),
        "discriminating_tasks": int(((c5["d"] > 0.10) & (c5["d"] < 0.90)).sum()),
        "zero_everywhere_tasks": int((c5["d"] == 0).sum()),
        "tasks_within_one_point_of_the_floor_line": int(((c5["d"] > 0.09) & (c5["d"] <= 0.11)).sum()),
        "d_quantiles": {q: float(c5["d"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(c5) else {},
        "d_mean": float(c5["d"].mean()) if len(c5) else float("nan"),
    }
    pw = None
    if with_pairs:
        pw = ml.pairwise_lab_spearman(mat, min_shared=30)
        ok = pw["rho"].dropna()
        out["pairwise"] = {"n_pairs": int(len(pw)), "n_pairs_defined": int(len(ok)),
                           "median": float(ok.median()) if len(ok) else float("nan"),
                           "q25": float(ok.quantile(0.25)) if len(ok) else float("nan"),
                           "q75": float(ok.quantile(0.75)) if len(ok) else float("nan"),
                           "min": float(ok.min()) if len(ok) else float("nan"),
                           "max": float(ok.max()) if len(ok) else float("nan"),
                           "share_positive": float((ok > 0).mean()) if len(ok) else float("nan")}
    return out, cons, mat, pw


# --------------------------------------------------------------------------
# P2  Variance decomposition
# --------------------------------------------------------------------------
def saturated_r2(att: pd.DataFrame, lab_balanced: bool) -> dict:
    """In-sample R^2 of the system x task cell-mean model on replicated cells.

    With lab_balanced the weights give every lab the same total weight on the replicated cells.
    """
    df = att[["lab", "system", "task", "y"]].copy()
    n = df.groupby(["system", "task"])["y"].transform("size")
    sub = df[n >= 2].copy()
    if sub.empty:
        return {"n_attempts": 0}
    sub["w"] = 1.0 / sub.groupby("lab")["y"].transform("size").to_numpy(float) if lab_balanced else 1.0
    y, ww = sub["y"].to_numpy(float), sub["w"].to_numpy(float)
    mean = np.sum(ww * y) / np.sum(ww)
    sst = np.sum(ww * (y - mean) ** 2)
    key = pd.factorize(pd.MultiIndex.from_frame(sub[["system", "task"]]))[0]
    fit = (np.bincount(key, weights=ww * y) / np.bincount(key, weights=ww))[key]
    sse = np.sum(ww * (y - fit) ** 2)
    out = {"n_attempts": int(len(sub)), "r2_saturated_in_sample": float(1 - sse / sst) if sst > 0 else float("nan")}
    # The same models on the replicated cells, for a like-for-like comparison.
    sub_att = att.loc[sub.index]
    out["models_on_replicated_cells"] = ml.r2_models(sub_att, ww)
    return out


def p2(att: pd.DataFrame):
    w = ml.lab_balanced_weights(att)
    out = {
        "lab_balanced": ml.r2_models(att, w),
        "unweighted": ml.r2_models(att),
        "replicate_noise_lab_balanced": ml.replicate_noise(att, lab_balanced=True),
        "replicate_noise_unweighted": ml.replicate_noise(att),
        # Before Amendment 4 the weights were taken from the whole table, which does not balance the labs
        # on the replicated cells; kept so that the change can be seen.
        "replicate_noise_whole_table_weights": ml.replicate_noise(att, w),
        "saturated_lab_balanced": saturated_r2(att, True),
        "saturated_unweighted": saturated_r2(att, False),
    }
    rep = att[att.groupby(["system", "task"])["y"].transform("size") >= 2]
    share = rep.groupby("lab").size() / len(rep)
    per_lab = {}
    for lab, g in rep.groupby("lab"):
        nz = ml.replicate_noise(g)
        per_lab[lab] = {"attempts": int(len(g)), "noise_share": nz.get("noise_share"),
                        "noise_variance": nz.get("noise_variance"), "resolve_rate": float(g["y"].mean())}
    out["replicated_cells"] = {
        "n_labs": int(rep["lab"].nunique()), "n_systems": int(rep["system"].nunique()),
        "share_of_all_attempts": float(len(rep) / len(att)),
        "resolve_rate_on_replicated_cells": float(rep["y"].mean()),
        "resolve_rate_elsewhere": float(att.drop(rep.index)["y"].mean()),
        "largest_lab_share_of_replicated_attempts": float(share.max()), "by_lab": per_lab}
    return out


def p2b(att: pd.DataFrame, codes: dict):
    rows = []
    for lab, g in att.groupby("lab"):
        blk = ml.balanced_block(g)
        if blk is None:
            continue
        block, systems, tasks = blk
        if block.shape[2] < 2 or block.shape[0] < 2:
            continue
        gs = ml.g_study(block)
        comp = gs["components"]
        row = {"code": codes[lab], "lab": lab, "n_systems": gs["n_systems"], "n_tasks": gs["n_tasks"],
               "n_replicates": gs["n_replicates"], "mean": gs["mean"]}
        for k, v in comp.items():
            row[f"var_{k}"] = v
            row[f"share_{k}"] = gs["shares"][k]
            row[f"raw_{k}"] = gs["raw"][k]
        for n in (30, 60, 129):
            for k in (1, 3, 5):
                row[f"g_n{n}_k{k}"] = ml.g_coefficient(comp, n, k)
            row[f"se_diff_n{n}"] = ml.se_difference(comp, n, 1)
        # Added after an independent review (Amendment 4): how well the system component is estimated, and
        # the smallest difference between two systems that can be detected when the claim is meant to hold
        # for new tasks (tasks random).  More runs shrink only the run-to-run term.
        row["df_system"] = gs["n_systems"] - 1
        for n in (58, 129):
            for k, name in ((1, "1"), (3, "3"), (5, "5"), (10, "10"), (np.inf, "inf")):
                row[f"detect_n{n}_k{name}"] = ml.detectable_random_tasks(comp, n, k)
        rows.append(row)
    tab = pd.DataFrame(rows)
    out = {"n_labs": int(len(tab))}
    if len(tab):
        for col in [c for c in tab.columns if c.startswith(("share_", "g_", "se_diff_", "detect_"))]:
            v = tab[col].dropna()
            out[col] = {"median": float(v.median()) if len(v) else float("nan"),
                        "min": float(v.min()) if len(v) else float("nan"),
                        "max": float(v.max()) if len(v) else float("nan"), "n": int(len(v))}
    return out, tab


# --------------------------------------------------------------------------
# P3  Replicate noise and detectable differences
# --------------------------------------------------------------------------
def p3(att: pd.DataFrame, with_mdd: bool = True):
    pairs = ml.replicate_pairs(att, min_shared=10)
    out = {"flip_rate": ml.summarise_pairs(pairs, "flip_rate"),
           "abs_score_diff": ml.summarise_pairs(pairs, "abs_score_diff")}
    if len(pairs):
        out["pair_level"] = {
            "flip_median": float(pairs["flip_rate"].median()),
            "flip_q25": float(pairs["flip_rate"].quantile(0.25)), "flip_q75": float(pairs["flip_rate"].quantile(0.75)),
            "abs_diff_median": float(pairs["abs_score_diff"].median()),
            "abs_diff_q90": float(pairs["abs_score_diff"].quantile(0.9)),
            "abs_diff_max": float(pairs["abs_score_diff"].max()),
            "share_identical_runs": float((pairs["flip_rate"] == 0).mean()),
            "median_shared_tasks": float(pairs["n_shared"].median()),
        }
        # Net change is small relative to the churn underneath it.
        out["pair_level"]["mean_flip_over_abs_diff"] = float(
            pairs["flip_rate"].mean() / max(pairs["abs_score_diff"].mean(), 1e-12))
        if with_mdd:
            pi = out["flip_rate"]["pooled"]
            out["mdd_pooled_flip"] = {str(n): ml.minimum_detectable_difference(n, pi) for n in (30, 60, 129)}
            pi_med = out["flip_rate"]["lab_median"]
            out["mdd_lab_median_flip"] = {str(n): ml.minimum_detectable_difference(n, pi_med) for n in (30, 60, 129)}
            out["mdd_pooled_flip_fine_grid"] = {str(n): ml.minimum_detectable_difference(n, pi, step=0.0005)
                                                for n in (30, 60, 129)}
    return out, pairs


def p3_extras(att: pd.DataFrame, pairs: pd.DataFrame, flip: float):
    """Additions to P3 made after an independent review (Amendment 4); none replaces a planned number."""
    out = {}
    # (a) Other ways of averaging the same pairs.
    sysw = pairs.assign(flips=pairs["flip_rate"] * pairs["n_shared"]).groupby(["lab", "system"]).agg(
        flips=("flips", "sum"), shared=("n_shared", "sum"), n_pairs=("flip_rate", "size"),
        mean_shared=("n_shared", "mean"))
    sys_rate = sysw["flips"] / sysw["shared"]
    first = pairs.groupby(["lab", "system"], sort=False).head(1)
    w_lab = pairs.groupby("lab")["n_shared"].sum() / pairs["n_shared"].sum()
    w_sys = sysw["shared"] / sysw["shared"].sum()
    att1 = ml.first_attempt_per_run(att)
    cell = att1.groupby(["system", "task"])["y"].agg(s="sum", n="size").reset_index()
    cell = cell[cell["n"] >= 2]
    cell["pairs"] = cell["n"] * (cell["n"] - 1) / 2
    cell["discordant"] = cell["s"] * (cell["n"] - cell["s"])
    per_task = cell.groupby("task")[["discordant", "pairs"]].sum()
    out["aggregation"] = {
        "pooled_pairs_weighted_by_shared_tasks": flip,
        "each_system_weighted_by_its_tasks_once": float(np.average(sys_rate, weights=sysw["mean_shared"])),
        "first_pair_of_each_system": float(np.average(first["flip_rate"], weights=first["n_shared"])),
        "mean_over_systems": float(sys_rate.mean()),
        "each_task_once": float((per_task["discordant"] / per_task["pairs"]).mean()),
        "tasks_with_repeats": int(len(per_task)),
        "largest_lab_share_of_pooled_weight": float(w_lab.max()),
        "two_largest_labs_share_of_pooled_weight": float(w_lab.sort_values().tail(2).sum()),
        "largest_system_share_of_pooled_weight": float(w_sys.max()),
        "effective_number_of_labs": float(1 / np.sum(w_lab ** 2)),
        "effective_number_of_systems": float(1 / np.sum(w_sys ** 2)),
        "pairs_with_100_or_more_shared_tasks": int((pairs["n_shared"] >= 100).sum()),
        "labs_with_such_pairs": int(pairs.loc[pairs["n_shared"] >= 100, "lab"].nunique()),
        "max_shared_tasks_in_a_pair": int(pairs["n_shared"].max()),
        "runs_of_the_largest_system": int(att1.loc[att1["system"] == w_sys.idxmax()[1], "run"].nunique()),
    }
    # (b) Error flags: how many flips have an attempt that ended in a timeout or another flagged error.
    n_flips = n_flag = n_clean_pairs = n_clean_flips = 0
    att1 = att1.assign(e=(att1["err"] == "1").astype(int))
    for _, g in att1.groupby("system", sort=False):
        if g["run"].nunique() < 2:
            continue
        wy = g.pivot(index="task", columns="run", values="y")
        we = g.pivot(index="task", columns="run", values="e")
        cols = list(wy.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                ok = wy[cols[i]].notna() & wy[cols[j]].notna()
                if int(ok.sum()) < 10:
                    continue
                ya, yb = wy.loc[ok, cols[i]].to_numpy(), wy.loc[ok, cols[j]].to_numpy()
                ea, eb = we.loc[ok, cols[i]].to_numpy(), we.loc[ok, cols[j]].to_numpy()
                flipm = ya != yb
                fail_flag = np.where(ya < yb, ea, eb)          # flag of the failing attempt
                n_flips += int(flipm.sum())
                n_flag += int((flipm & (fail_flag == 1)).sum())
                clean = (ea == 0) & (eb == 0)
                n_clean_pairs += int(clean.sum())
                n_clean_flips += int((flipm & clean).sum())
    out["error_flags"] = {"flips": n_flips, "flips_with_error_flag_on_the_failing_attempt": n_flag,
                          "share": n_flag / n_flips if n_flips else None,
                          "task_pairs_with_no_flag": n_clean_pairs, "flips_among_them": n_clean_flips,
                          "flip_rate_with_no_flag": n_clean_flips / n_clean_pairs if n_clean_pairs else None}
    # (c) How much do two DIFFERENT systems disagree, against the size of their difference?
    cross = ml.cross_system_discordance(att, min_shared=30)
    fit = ml.excess_discordance_fit(cross)
    if len(cross):
        fit["max_abs_delta"] = float(cross["delta"].abs().max())
    out["cross_system_discordance"] = fit
    # (d) The planned detectable difference under three assumptions about that disagreement.
    alt = {}
    for n in (30, 60, 129):
        row = {"planned_all_or_nothing": ml.mdd_with_discordance(n, flip, 0.0, 1.0),
               "no_added_disagreement": ml.mdd_with_discordance(n, flip, 0.0, 0.0)}
        if fit.get("mean_excess") is not None:
            # The systems compared here differ by a few points only, so the extra disagreement is taken as the
            # constant that was observed, not as a function of the gain (the fitted slope is kept in the output).
            row["as_observed_between_systems"] = ml.mdd_with_discordance(n, flip, max(fit["mean_excess"], 0.0), 0.0)
        alt[str(n)] = row
    out["mdd_alternatives"] = alt
    return out, cross


# --------------------------------------------------------------------------
# P4  Rasch model and subset comparability
# --------------------------------------------------------------------------
def p4(att: pd.DataFrame, runs: pd.DataFrame):
    fit = ml.rasch_fit(att)
    theta, b = fit["theta"], fit["b"]
    sys_meta = att.groupby("system").agg(lab=("lab", "first"), model=("model", "first"),
                                         harness=("harness", "first"), n_attempts=("y", "size"),
                                         n_runs=("run", "nunique"), n_tasks=("task", "nunique"), rate=("y", "mean"))
    sys_meta["theta"] = theta
    ref_pool = sys_meta[sys_meta["model"] == "31B"]["theta"]
    theta_ref = float(ref_pool.median()) if len(ref_pool) else float(sys_meta["theta"].median())
    sets = att.groupby("run")["task"].apply(frozenset)
    selected_runs = outcome_selected_runs(att)
    distinct = {}
    for run, s in sets.items():
        if len(s) >= 10:
            distinct.setdefault(s, []).append(run)
    rows = []
    for s, rs in distinct.items():
        labs = sorted({runs.loc[r, "lab"] for r in rs})
        rows.append({"n_tasks": len(s), "n_runs": len(rs), "n_labs": len(labs),
                     "predicted_rate": ml.predict_subset(theta_ref, b, s),
                     "share_fastapi": float(np.mean([t.startswith("fastapi_") for t in s])),
                     "chosen_from_earlier_outcomes": bool(all(r in selected_runs for r in rs)),
                     "labs": ";".join(labs), "_tasks": s})
    tab = pd.DataFrame(rows).sort_values("predicted_rate").reset_index(drop=True)
    subset_list = list(tab.pop("_tasks"))
    pred = tab["predicted_rate"]
    full = ml.predict_subset(theta_ref, b, b.index)

    def span(v):
        return {"n": int(len(v)), "min": float(v.min()), "max": float(v.max()),
                "q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75))} if len(v) else {"n": 0}

    out = {
        "converged": fit["converged"], "n_systems": int(len(theta)), "n_tasks": int(len(b)),
        "theta_ref_31B_median": theta_ref, "n_31B_systems": int(len(ref_pool)),
        "n_distinct_subsets": int(len(tab)), "predicted_full_task_set": full,
        "predicted_rate": {"min": float(pred.min()), "max": float(pred.max()), "range": float(pred.max() - pred.min()),
                           "q25": float(pred.quantile(0.25)), "median": float(pred.median()),
                           "q75": float(pred.quantile(0.75)), "iqr": float(pred.quantile(0.75) - pred.quantile(0.25))},
        "ratio_max_min": float(pred.max() / pred.min()) if pred.min() > 0 else float("nan"),
        "theta_range_31B": {"min": float(ref_pool.min()), "max": float(ref_pool.max()),
                            "q25": float(ref_pool.quantile(0.25)), "q75": float(ref_pool.quantile(0.75))} if len(ref_pool) else {},
        # Qualifiers added after an independent review (Amendment 4).
        "subsets_used_by_more_than_one_lab": int((tab["n_labs"] > 1).sum()),
        "subsets_chosen_from_earlier_outcomes": int(tab["chosen_from_earlier_outcomes"].sum()),
        "lowest_subset_chosen_from_earlier_outcomes": bool(tab["chosen_from_earlier_outcomes"].iloc[0]),
        "highest_subset_chosen_from_earlier_outcomes": bool(tab["chosen_from_earlier_outcomes"].iloc[-1]),
        "predicted_rate_not_chosen_from_outcomes": span(tab.loc[~tab["chosen_from_earlier_outcomes"], "predicted_rate"]),
        "predicted_rate_not_chosen_30plus": span(tab.loc[~tab["chosen_from_earlier_outcomes"] & (tab["n_tasks"] >= 30),
                                                         "predicted_rate"]),
        "predicted_rate_subsets_50plus": span(tab.loc[tab["n_tasks"] >= 50, "predicted_rate"]),
    }
    by_size = tab[tab["n_tasks"] >= 30]["predicted_rate"]
    if len(by_size):
        out["predicted_rate_subsets_30plus"] = {"n": int(len(by_size)), "min": float(by_size.min()),
                                                "max": float(by_size.max()),
                                                "iqr": float(by_size.quantile(0.75) - by_size.quantile(0.25))}
    # Out-of-sample checks.  (The in-sample check reported before Amendment 4 was empty: the score
    # equations of the model force predicted and observed totals per system to agree.)
    out["cross_validation"] = ml.rasch_cv(att, n_folds=5, seed=SEED)
    out["subset_effect_out_of_sample"], subset_pairs = ml.rasch_subset_check(att, subset_list)
    return out, tab, sys_meta.reset_index(), b, subset_pairs


# --------------------------------------------------------------------------
# Exploratory
# --------------------------------------------------------------------------
def e2_model_size(sys_meta: pd.DataFrame, att: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    sm = sys_meta[sys_meta["model"] != ""]
    out = {"note": "lab-clustered bootstrap of the fitted abilities; the Rasch model is not refitted"}
    labs = sm["lab"].unique()
    by = {}
    for model, g in sm.groupby("model"):
        lab_means = g.groupby("lab")["theta"].mean()
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(labs, size=len(labs), replace=True)
            vals = [lab_means[l] for l in pick if l in lab_means.index]
            if vals:
                boots.append(np.mean(vals))
        nl = len(lab_means)
        half = float(stats.t.ppf(0.975, nl - 1) * lab_means.std(ddof=1) / np.sqrt(nl)) if nl >= 3 else None
        by[model] = {"n_systems": int(len(g)), "n_labs": int(g["lab"].nunique()),
                     "mean_theta_over_labs": float(lab_means.mean()),
                     "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                     # t interval on the lab means: wider and more honest with so few labs (Amendment 4)
                     "t_interval95": None if half is None else [float(lab_means.mean() - half), float(lab_means.mean() + half)],
                     "lab_mean_min": float(lab_means.min()), "lab_mean_max": float(lab_means.max()),
                     "mean_raw_rate_over_labs": float(g.groupby("lab")["rate"].mean().mean())}
    out["by_model"] = by
    # Within-lab contrasts on shared tasks.
    contrasts = []
    cells = att[att["model"] != ""].groupby(["lab", "model", "task"])["y"].mean().reset_index()
    for lab, g in cells.groupby("lab"):
        wide = g.pivot(index="task", columns="model", values="y")
        models = list(wide.columns)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pair = wide[[models[i], models[j]]].dropna()
                if len(pair) >= 10:
                    contrasts.append({"lab": lab, "model_a": models[i], "model_b": models[j],
                                      "n_shared_tasks": int(len(pair)),
                                      "rate_a": float(pair[models[i]].mean()), "rate_b": float(pair[models[j]].mean())})
    out["within_lab_contrasts"] = contrasts
    return out


def e3_repo(cons: pd.DataFrame):
    c = cons[cons["n_labs"] >= 5].copy()
    c["repo"] = c["task"].str.split("_").str[0]
    g = c.groupby("repo")["d"]
    return {r: {"n_tasks": int(n), "mean_d": float(m), "median_d": float(md),
                "share_floor": float(fl)}
            for r, n, m, md, fl in zip(g.size().index, g.size(), g.mean(), g.median(),
                                       c.assign(f=c["d"] <= 0.10).groupby("repo")["f"].mean())}


def e4_instability(att: pd.DataFrame, cons: pd.DataFrame):
    """Per-task replicate disagreement (2 x unbiased within-cell variance) against d[t]."""
    g = att.groupby(["lab", "system", "task"])["y"].agg(p="mean", n="size").reset_index()
    g = g[g["n"] >= 2].copy()
    if g.empty:
        return {"n_tasks": 0}, None
    g["flip"] = 2 * g["n"] / (g["n"] - 1) * g["p"] * (1 - g["p"])
    per_lab = g.groupby(["lab", "task"])["flip"].mean().reset_index()
    per_task = per_lab.groupby("task")["flip"].agg(flip="mean", n_labs_rep="size").reset_index()
    m = per_task.merge(cons[["task", "d", "n_labs"]], on="task")
    m = m[(m["n_labs"] >= 5) & (m["n_labs_rep"] >= 2)]
    if len(m) < 5:
        return {"n_tasks": int(len(m))}, m
    m["binomial_ceiling"] = 2 * m["d"] * (1 - m["d"])
    disc = (m["d"] > 0.10) & (m["d"] < 0.90)
    out = {"n_tasks": int(len(m)),
           "spearman_flip_vs_binomial": float(stats.spearmanr(m["flip"], m["binomial_ceiling"]).statistic),
           "mean_flip_discriminating": float(m.loc[disc, "flip"].mean()) if disc.any() else float("nan"),
           "mean_flip_floor": float(m.loc[m["d"] <= 0.10, "flip"].mean()) if (m["d"] <= 0.10).any() else float("nan"),
           "share_of_total_flip_in_discriminating": float(m.loc[disc, "flip"].sum() / m["flip"].sum()) if m["flip"].sum() > 0 else float("nan"),
           "n_discriminating": int(disc.sum()),
           "ratio_observed_to_binomial": float(m["flip"].sum() / m["binomial_ceiling"].sum()) if m["binomial_ceiling"].sum() > 0 else float("nan")}
    return out, m


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------
def core(att: pd.DataFrame, n_iter: int = N_SPLIT) -> dict:
    """The headline quantities on a subset of the data."""
    if att["lab"].nunique() < 4:
        return {"n_labs": int(att["lab"].nunique()), "n_attempts": int(len(att)), "note": "too few labs"}
    a, _, _, _ = p1(att, n_iter=n_iter, with_pairs=False)
    w = ml.lab_balanced_weights(att)
    r2 = ml.r2_models(att, w)
    noise = ml.replicate_noise(att, lab_balanced=True)
    c, pairs = p3(att, with_mdd=False)
    out = {"n_attempts": int(len(att)), "n_labs": int(att["lab"].nunique()),
           "n_systems": int(att["system"].nunique()), "n_runs": int(att["run"].nunique()),
           "split_half_median": a["split_half"].get("median"), "split_half_sb": a["split_half"].get("spearman_brown"),
           "n_tasks_5_labs": a["n_tasks_5_labs"], "floor_tasks": a["floor_tasks"], "ceiling_tasks": a["ceiling_tasks"],
           "discriminating_tasks": a["discriminating_tasks"],
           "r2_task": r2.get("r2_task"), "r2_lab": r2.get("r2_lab"), "r2_system": r2.get("r2_system"),
           "r2_task_plus_system": r2.get("r2_task_plus_system"),
           "noise_share": noise.get("noise_share"), "noise_variance": noise.get("noise_variance"),
           "flip_pooled": c["flip_rate"].get("pooled"), "flip_lab_median": c["flip_rate"].get("lab_median"),
           "flip_n_labs": c["flip_rate"].get("n_labs"), "flip_n_pairs": c["flip_rate"].get("n_pairs")}
    if out["flip_pooled"] is not None:
        out["mdd_129"] = ml.minimum_detectable_difference(129, out["flip_pooled"])
    return out


def robustness(att: pd.DataFrame, codes: dict) -> dict:
    out = {"all": core(att)}
    out["model_31B_only"] = core(att[att["model"] == "31B"])
    out["official_harness_only"] = core(att[att["harness"] == "official"])
    no_err = att[att["err"] != "1"]          # attempts whose flag is unknown ('') are kept
    out["drop_error_flagged"] = core(no_err)
    # The same after re-applying the 10-task minimum of plan section 5 to what is left of each run.
    left = no_err.groupby("run")["task"].transform("size")
    out["drop_error_flagged_min10"] = core(no_err[left >= 10])
    flagged = att["err"] == "1"
    fails = att["y"] == 0
    out["error_flags"] = {
        "share_of_attempts_flagged": float(flagged.mean()),
        "attempts_with_unknown_flag": int((att["err"] == "").sum()),
        "share_of_failures_flagged": float((flagged & fails).sum() / max(int(fails.sum()), 1)),
        "runs_left_with_fewer_than_10_tasks": int((no_err.groupby("run").size() < 10).sum()
                                                  + (att["run"].nunique() - no_err["run"].nunique())),
        "resolve_rate_after_dropping": float(no_err["y"].mean())}
    # Unweighted variants of the weighted quantities.
    out["unweighted"] = {"r2": ml.r2_models(att), "noise": ml.replicate_noise(att)}
    # Leave one lab out.
    loo = []
    for lab in sorted(att["lab"].unique(), key=lambda l: codes[l]):
        sub = att[att["lab"] != lab]
        a, _, _, _ = p1(sub, n_iter=N_SPLIT, with_pairs=False)
        pairs = ml.replicate_pairs(sub, min_shared=10)
        s = ml.summarise_pairs(pairs, "flip_rate")
        loo.append({"left_out": codes[lab], "split_half_median": a["split_half"].get("median"),
                    "floor_tasks": a["floor_tasks"], "n_tasks_5_labs": a["n_tasks_5_labs"],
                    "flip_pooled": s.get("pooled"), "flip_lab_median": s.get("lab_median")})
    lo = pd.DataFrame(loo)
    out["leave_one_lab_out"] = {
        "split_half_median": {"min": float(lo["split_half_median"].min()), "max": float(lo["split_half_median"].max())},
        "flip_pooled": {"min": float(lo["flip_pooled"].min()), "max": float(lo["flip_pooled"].max())},
        "flip_lab_median": {"min": float(lo["flip_lab_median"].min()), "max": float(lo["flip_lab_median"].max())},
        "floor_tasks": {"min": int(lo["floor_tasks"].min()), "max": int(lo["floor_tasks"].max())},
    }
    return out, lo


# --------------------------------------------------------------------------
def main(argv=()) -> int:
    """Optional arguments (used by the test run): attempts.csv and an output directory."""
    global RESULTS, TABLES
    if len(argv) >= 2:
        RESULTS = Path(argv[1])
        TABLES = RESULTS / "tables"
    TABLES.mkdir(parents=True, exist_ok=True)
    att = load(Path(argv[0]) if argv else None)
    codes = lab_codes(att)
    res = {}

    res["data"], labs, runs = describe(att, codes)
    labs.to_csv(TABLES / "labs.csv", index=False, lineterminator="\n")
    print("data:", json.dumps(rnd({k: v for k, v in res["data"].items() if not isinstance(v, dict)})), flush=True)

    res["P1"], cons, mat, pw = p1(att)
    pw2 = pw.assign(lab_a=pw["lab_a"].map(codes), lab_b=pw["lab_b"].map(codes))
    pw2.to_csv(TABLES / "pairwise_lab_spearman.csv", index=False, lineterminator="\n")
    print("P1:", json.dumps(rnd(res["P1"])), flush=True)

    res["P2"] = p2(att)
    res["P2"]["replicated_cells"]["by_lab"] = {codes[k]: v for k, v in sorted(
        res["P2"]["replicated_cells"]["by_lab"].items(), key=lambda kv: codes[kv[0]])}
    print("P2:", json.dumps(rnd(res["P2"])), flush=True)

    res["P2b"], gtab = p2b(att, codes)
    gtab.drop(columns=["lab"], errors="ignore").to_csv(TABLES / "g_study.csv", index=False, lineterminator="\n")
    print("P2b:", json.dumps(rnd(res["P2b"])), flush=True)

    res["P3"], pairs = p3(att)
    pairs.assign(lab=pairs["lab"].map(codes)).to_csv(TABLES / "replicate_pairs.csv", index=False, lineterminator="\n")
    noise_u = res["P2"]["replicate_noise_unweighted"].get("noise_variance")
    noise_w = res["P2"]["replicate_noise_lab_balanced"].get("noise_variance")
    res["P3"]["consistency_check"] = {
        "pooled_flip_rate": res["P3"]["flip_rate"].get("pooled"),
        "twice_noise_variance_unweighted": None if noise_u is None else 2 * noise_u,
        "lab_mean_flip_rate": res["P3"]["flip_rate"].get("lab_mean"),
        "twice_noise_variance_lab_balanced": None if noise_w is None else 2 * noise_w}
    if len(pairs):
        res["P3"]["extras"], cross = p3_extras(att, pairs, res["P3"]["flip_rate"]["pooled"])
        cross.assign(lab=cross["lab"].map(codes)).to_csv(TABLES / "cross_system_discordance.csv", index=False,
                                                         lineterminator="\n")
    print("P3:", json.dumps(rnd(res["P3"])), flush=True)

    res["P4"], subsets, sys_meta, b, subset_pairs = p4(att, runs)
    subsets.assign(labs=subsets["labs"].map(lambda s: ";".join(codes[l] for l in s.split(";"))))\
        .to_csv(TABLES / "subset_predictions.csv", index=False, lineterminator="\n")
    if len(subset_pairs):
        subset_pairs.assign(lab=subset_pairs["lab"].map(codes)).to_csv(
            TABLES / "subset_effect_out_of_sample.csv", index=False, lineterminator="\n")
    sys_meta.assign(lab=sys_meta["lab"].map(codes)).to_csv(TABLES / "systems.csv", index=False, lineterminator="\n")
    print("P4:", json.dumps(rnd(res["P4"])), flush=True)

    res["E2_model_size"] = e2_model_size(sys_meta, att)
    for c in res["E2_model_size"]["within_lab_contrasts"]:
        c["lab"] = codes[c["lab"]]
    res["E3_repo"] = e3_repo(cons)
    res["E4_instability"], inst = e4_instability(att, cons)
    print("E2:", json.dumps(rnd(res["E2_model_size"])), flush=True)
    print("E3:", json.dumps(rnd(res["E3_repo"])), flush=True)
    print("E4:", json.dumps(rnd(res["E4_instability"])), flush=True)

    # Task table.
    t = cons.copy()
    t["repo"] = t["task"].str.split("_").str[0]
    t["n_systems"] = t["task"].map(att.groupby("task")["system"].nunique())
    t["n_attempts"] = t["task"].map(att.groupby("task").size())
    t["raw_rate"] = t["task"].map(att.groupby("task")["y"].mean())
    t["rasch_b"] = t["task"].map(b)
    t["class"] = np.where(t["n_labs"] < 5, "fewer_than_5_labs",
                          np.where(t["d"] <= 0.10, "floor", np.where(t["d"] >= 0.90, "ceiling", "discriminating")))
    if inst is not None and len(inst):
        t = t.merge(inst[["task", "flip", "n_labs_rep"]].rename(columns={"flip": "replicate_disagreement"}),
                    on="task", how="left")
    controls = (Path(argv[0]).parent if argv else DERIVED) / "controls.csv"
    x1, audit, lab_tab = validity.x1_validity(att, cons, codes, controls, core, TABLES)
    if x1 is not None:
        res["X1_validity"] = x1
        t = t.merge(audit, on="task", how="left")
        lab_tab.to_csv(TABLES / "validity_by_lab.csv", index=False, lineterminator="\n")
        print("X1:", json.dumps(rnd(x1)), flush=True)
    e1 = validity.e1_leaderboard(ROOT / "data" / "sources" / "leaderboard_pairs.csv") if not argv else None
    if e1 is not None:
        res["E1_leaderboard"] = e1
        flip = res["P3"]["flip_rate"].get("pooled")
        if flip:
            # Spread expected between two runs of one system on n tasks if the hidden tasks were as noisy
            # as the development tasks: sd of the difference = sqrt(flip rate / n).
            e1["expected_sd_of_difference_58_tasks_at_dev_flip_rate"] = float(np.sqrt(flip / 58))
            e1["expected_mean_abs_difference_58_tasks_at_dev_flip_rate"] = float(np.sqrt(flip / 58) * np.sqrt(2 / np.pi))
        print("E1:", json.dumps(rnd(e1)), flush=True)
    t.sort_values("d").to_csv(TABLES / "tasks.csv", index=False, lineterminator="\n")
    if "audit_status" in t.columns:
        # Task lists a team can use directly.  "Valid" = the reference patch resolves in every full audit of
        # the stock sandbox and the task does not resolve with no patch (Amendment 4).
        st = t["audit_status"].astype(str)
        lists = {
            "valid_in_the_stock_sandbox": sorted(t.loc[st == "valid", "task"]),
            "reference_patch_fails_in_every_full_audit": sorted(t.loc[st.str.startswith("reference_patch_fails"), "task"]),
            "reference_patch_fails_in_every_full_audit_and_in_docker": sorted(
                t.loc[st == "reference_patch_fails_in_both", "task"]),
            "audits_disagree": sorted(t.loc[st == "audits_disagree", "task"]),
            "passes_without_patch": sorted(t.loc[st == "passes_without_patch", "task"]),
            "informative_core_valid_and_discriminating": sorted(
                t.loc[(st == "valid") & (t["class"] == "discriminating"), "task"]),
            "valid_but_at_the_floor": sorted(t.loc[(st == "valid") & (t["class"] == "floor"), "task"]),
            "valid_in_the_single_audit_used_before_amendment_4": sorted(t.loc[t["single_audit_status"] == "valid", "task"]),
        }
        (TABLES / "task_lists.json").write_text(json.dumps(lists, indent=1) + "\n", encoding="utf-8")
        res["task_lists"] = {k: len(v) for k, v in lists.items()}
    mat.rename(index=codes).sort_index().to_csv(TABLES / "lab_task_matrix.csv", lineterminator="\n")

    res["robustness"], lo = robustness(att, codes)
    lo.to_csv(TABLES / "leave_one_lab_out.csv", index=False, lineterminator="\n")
    print("robustness:", json.dumps(rnd(res["robustness"])), flush=True)

    res["lab_codes_note"] = "Lab codes L01.. are assigned by decreasing number of attempts; see results/tables/labs.csv"
    (RESULTS / "results.json").write_text(json.dumps(rnd(res, 5), indent=1) + "\n", encoding="utf-8")
    print("written", RESULTS / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
