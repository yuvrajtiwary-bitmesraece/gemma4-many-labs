"""Estimators used by the cross-lab synthesis (see ANALYSIS_PLAN.md).

Everything takes a long table of attempts with the columns

    lab, system, run, task, y

where `system` is unique across labs, `run` is unique across systems and `y`
is 0 or 1.  Each function is checked against simulated data in
scripts/test_stats.py.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats
from scipy.sparse.linalg import lsqr
from scipy.special import expit


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def first_attempt_per_run(att: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, task): the first attempt in file order."""
    return att.drop_duplicates(subset=["run", "task"], keep="first")


# --------------------------------------------------------------------------
# Levels: replicate -> system -> lab -> consensus   (plan section 6)
# --------------------------------------------------------------------------
def cell_means(att: pd.DataFrame) -> pd.DataFrame:
    """Solve rate and attempt count for every (lab, system, task) cell."""
    g = att.groupby(["lab", "system", "task"], sort=False)["y"]
    out = g.agg(p="mean", n="size").reset_index()
    return out


def lab_task_means(cells: pd.DataFrame) -> pd.DataFrame:
    """Lab-level solve rate per task: mean over the lab's systems."""
    g = cells.groupby(["lab", "task"], sort=False)["p"]
    return g.agg(p="mean", n_systems="size").reset_index()


def consensus(lab_task: pd.DataFrame) -> pd.DataFrame:
    """Consensus difficulty d[t]: mean over labs, with the lab count."""
    g = lab_task.groupby("task", sort=True)["p"]
    return g.agg(d="mean", n_labs="size", sd_labs="std").reset_index()


def lab_task_matrix(lab_task: pd.DataFrame) -> pd.DataFrame:
    """Labs x tasks matrix of lab-level solve rates (NaN where not attempted)."""
    return lab_task.pivot(index="lab", columns="task", values="p")


# --------------------------------------------------------------------------
# P1  Consensus reliability
# --------------------------------------------------------------------------
def split_half_reliability(mat: pd.DataFrame, n_iter: int = 2000, seed: int = 20261111,
                           min_labs_per_half: int = 2, min_tasks: int = 20) -> dict:
    """Spearman correlation of consensus difficulty between random lab halves."""
    rng = np.random.default_rng(seed)
    x = mat.to_numpy(dtype=float)
    n_labs = x.shape[0]
    present = ~np.isnan(x)
    filled = np.where(present, x, 0.0)
    rs, n_tasks_used = [], []
    for _ in range(n_iter):
        perm = rng.permutation(n_labs)
        a, b = perm[: (n_labs + 1) // 2], perm[(n_labs + 1) // 2:]
        ca, cb = present[a].sum(0), present[b].sum(0)
        ok = (ca >= min_labs_per_half) & (cb >= min_labs_per_half)
        if ok.sum() < min_tasks:
            continue
        da = filled[a][:, ok].sum(0) / ca[ok]
        db = filled[b][:, ok].sum(0) / cb[ok]
        if np.ptp(da) == 0 or np.ptp(db) == 0:
            continue
        rs.append(stats.spearmanr(da, db).statistic)
        n_tasks_used.append(int(ok.sum()))
    rs = np.asarray(rs)
    if rs.size == 0:
        return {"n_valid": 0}
    med = float(np.median(rs))
    return {
        "n_valid": int(rs.size),
        "median": med,
        "p2.5": float(np.percentile(rs, 2.5)),
        "p97.5": float(np.percentile(rs, 97.5)),
        "spearman_brown": 2 * med / (1 + med),
        "median_tasks_used": float(np.median(n_tasks_used)),
    }


def pairwise_lab_spearman(mat: pd.DataFrame, min_shared: int = 30) -> pd.DataFrame:
    """Spearman correlation of lab-level solve rates for every lab pair."""
    rows = []
    labs = list(mat.index)
    x = mat.to_numpy(dtype=float)
    for i, j in itertools.combinations(range(len(labs)), 2):
        ok = ~np.isnan(x[i]) & ~np.isnan(x[j])
        n = int(ok.sum())
        if n < min_shared:
            continue
        a, b = x[i, ok], x[j, ok]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            rows.append({"lab_a": labs[i], "lab_b": labs[j], "n_shared": n, "rho": np.nan})
            continue
        rows.append({"lab_a": labs[i], "lab_b": labs[j], "n_shared": n,
                     "rho": float(stats.spearmanr(a, b).statistic)})
    return pd.DataFrame(rows, columns=["lab_a", "lab_b", "n_shared", "rho"])


# --------------------------------------------------------------------------
# P2  Variance decomposition (linear probability models)
# --------------------------------------------------------------------------
def lab_balanced_weights(att: pd.DataFrame) -> np.ndarray:
    """Weights that give every lab the same total weight (sum = number of labs)."""
    n = att.groupby("lab")["y"].transform("size").to_numpy(dtype=float)
    return 1.0 / n


def _wss(y, w, fitted):
    return float(np.sum(w * (y - fitted) ** 2))


def _group_fit(y, w, codes):
    """Weighted group means as fitted values."""
    num = np.bincount(codes, weights=w * y)
    den = np.bincount(codes, weights=w)
    return (num / den)[codes]


def r2_models(att: pd.DataFrame, weights: np.ndarray | None = None) -> dict:
    """In-sample R^2 of linear probability models with fixed effects."""
    y = att["y"].to_numpy(dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    mean = np.sum(w * y) / np.sum(w)
    sst = _wss(y, w, mean)
    out = {"n_attempts": int(y.size), "mean": float(mean), "total_variance": sst / float(np.sum(w))}
    if sst == 0:
        return out
    codes = {c: pd.factorize(att[c])[0] for c in ("task", "lab", "system")}
    for c in ("task", "lab", "system"):
        out[f"r2_{c}"] = 1 - _wss(y, w, _group_fit(y, w, codes[c])) / sst
    # Additive task + system model by sparse weighted least squares.
    n = y.size
    n_t = codes["task"].max() + 1
    n_s = codes["system"].max() + 1
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([codes["task"], n_t + codes["system"]])
    sw = np.sqrt(w)
    design = sparse.csr_matrix((np.concatenate([sw, sw]), (rows, cols)), shape=(n, n_t + n_s))
    sol = lsqr(design, sw * y, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    fitted = sol[codes["task"]] + sol[n_t + codes["system"]]
    out["r2_task_plus_system"] = 1 - _wss(y, w, fitted) / sst
    out["r2_system_given_task"] = out["r2_task_plus_system"] - out["r2_task"]
    out["r2_task_given_system"] = out["r2_task_plus_system"] - out["r2_system"]
    return out


def replicate_noise(att: pd.DataFrame, weights: np.ndarray | None = None) -> dict:
    """Unbiased within-cell variance on cells with at least two attempts.

    Returns the weighted mean within-cell variance, the total outcome variance
    on the same attempts, and their ratio (the replicate-noise share).
    """
    w = np.ones(len(att)) if weights is None else np.asarray(weights, dtype=float)
    df = att[["system", "task", "y"]].copy()
    df["w"] = w
    g = df.groupby(["system", "task"], sort=False)
    n = g["y"].transform("size").to_numpy()
    keep = n >= 2
    if not keep.any():
        return {"n_cells": 0}
    sub = df.loc[keep]
    cell = sub.groupby(["system", "task"], sort=False).agg(p=("y", "mean"), n=("y", "size"), w=("w", "sum"))
    s2 = cell["n"] / (cell["n"] - 1) * cell["p"] * (1 - cell["p"])
    noise = float(np.sum(cell["w"] * s2) / np.sum(cell["w"]))
    y = sub["y"].to_numpy(dtype=float)
    ww = sub["w"].to_numpy()
    mean = np.sum(ww * y) / np.sum(ww)
    total = float(np.sum(ww * (y - mean) ** 2) / np.sum(ww))
    return {"n_cells": int(len(cell)), "n_attempts": int(keep.sum()), "noise_variance": noise,
            "total_variance": total, "noise_share": noise / total if total > 0 else float("nan")}


# --------------------------------------------------------------------------
# P2b  Within-lab generalizability study (Amendment 1b)
# --------------------------------------------------------------------------
def balanced_block(att_lab: pd.DataFrame, min_tasks: int = 20):
    """Largest balanced systems x tasks x replicates block of one lab.

    Follows the rule in DEVIATIONS.md, Amendment 1b.  Returns (array, systems,
    tasks) with array shape (S, T, r), or None if the lab has no usable block.
    """
    att_lab = first_attempt_per_run(att_lab)
    runs_per_system = att_lab.groupby("system")["run"].nunique()
    systems = list(runs_per_system[runs_per_system >= 2].index)
    cnt = (att_lab[att_lab["system"].isin(systems)]
           .groupby(["system", "task"])["run"].nunique().unstack(fill_value=0))
    while len(systems) >= 2:
        sub = cnt.loc[systems]
        tasks = list(sub.columns[(sub >= 2).all(axis=0)])
        if len(tasks) >= min_tasks:
            break
        coverage = (sub >= 2).sum(axis=1)
        systems.remove(coverage.idxmin())
    else:
        return None
    r = int(cnt.loc[systems, tasks].to_numpy().min())
    block = np.zeros((len(systems), len(tasks), r), dtype=float)
    sub = att_lab[att_lab["system"].isin(systems) & att_lab["task"].isin(tasks)]
    s_idx = {s: i for i, s in enumerate(systems)}
    t_idx = {t: i for i, t in enumerate(tasks)}
    fill = np.zeros((len(systems), len(tasks)), dtype=int)
    for s, t, y in zip(sub["system"], sub["task"], sub["y"]):
        i, j = s_idx[s], t_idx[t]
        k = fill[i, j]
        if k < r:
            block[i, j, k] = y
            fill[i, j] = k + 1
    return block, systems, tasks


def g_study(block: np.ndarray) -> dict:
    """Variance components of a balanced systems x tasks x replicates block."""
    s, t, r = block.shape
    grand = block.mean()
    m_s = block.mean(axis=(1, 2))
    m_t = block.mean(axis=(0, 2))
    m_st = block.mean(axis=2)
    ss_s = t * r * np.sum((m_s - grand) ** 2)
    ss_t = s * r * np.sum((m_t - grand) ** 2)
    ss_st = r * np.sum((m_st - m_s[:, None] - m_t[None, :] + grand) ** 2)
    ss_e = np.sum((block - m_st[:, :, None]) ** 2)
    ms_s = ss_s / (s - 1)
    ms_t = ss_t / (t - 1)
    ms_st = ss_st / ((s - 1) * (t - 1))
    ms_e = ss_e / (s * t * (r - 1))
    raw = {
        "task": (ms_t - ms_st) / (s * r),
        "system": (ms_s - ms_st) / (t * r),
        "system_x_task": (ms_st - ms_e) / r,
        "replicate": ms_e,
    }
    comp = {k: max(0.0, float(v)) for k, v in raw.items()}
    total = sum(comp.values())
    out = {"n_systems": s, "n_tasks": t, "n_replicates": r, "mean": float(grand),
           "raw": {k: float(v) for k, v in raw.items()}, "components": comp,
           "shares": {k: (v / total if total > 0 else float("nan")) for k, v in comp.items()}}
    return out


def g_coefficient(comp: dict, n_tasks: int, n_runs: int) -> float:
    """Generalizability coefficient for comparing systems on shared tasks."""
    err = comp["system_x_task"] / n_tasks + comp["replicate"] / (n_tasks * n_runs)
    den = comp["system"] + err
    return comp["system"] / den if den > 0 else float("nan")


def se_difference(comp: dict, n_tasks: int, n_runs: int = 1) -> float:
    """Standard error of the score difference of two systems on shared tasks."""
    return math.sqrt(2 * (comp["system_x_task"] + comp["replicate"] / n_runs) / n_tasks)


# --------------------------------------------------------------------------
# P3  Replicate noise and detectable differences
# --------------------------------------------------------------------------
def replicate_pairs(att: pd.DataFrame, min_shared: int = 10) -> pd.DataFrame:
    """Flip rate and score difference for every pair of replicate runs."""
    att = first_attempt_per_run(att)
    rows = []
    for (lab, system), g in att.groupby(["lab", "system"], sort=False):
        runs = list(dict.fromkeys(g["run"]))
        if len(runs) < 2:
            continue
        wide = g.pivot(index="task", columns="run", values="y")
        for a, b in itertools.combinations(runs, 2):
            pair = wide[[a, b]].dropna()
            if len(pair) < min_shared:
                continue
            ya, yb = pair[a].to_numpy(), pair[b].to_numpy()
            rows.append({"lab": lab, "system": system, "run_a": a, "run_b": b, "n_shared": len(pair),
                         "flip_rate": float(np.mean(ya != yb)),
                         "abs_score_diff": float(abs(ya.mean() - yb.mean())),
                         "score_a": float(ya.mean()), "score_b": float(yb.mean())})
    return pd.DataFrame(rows, columns=["lab", "system", "run_a", "run_b", "n_shared", "flip_rate",
                                       "abs_score_diff", "score_a", "score_b"])


def summarise_pairs(pairs: pd.DataFrame, col: str = "flip_rate") -> dict:
    """Pairs -> systems -> labs, then median and IQR across labs, plus pooled."""
    if pairs.empty:
        return {"n_pairs": 0}
    per_system = pairs.groupby(["lab", "system"])[col].mean()
    per_lab = per_system.groupby("lab").mean()
    pooled = float(np.average(pairs[col], weights=pairs["n_shared"]))
    return {"n_pairs": int(len(pairs)), "n_systems": int(len(per_system)), "n_labs": int(len(per_lab)),
            "lab_median": float(per_lab.median()), "lab_q25": float(per_lab.quantile(0.25)),
            "lab_q75": float(per_lab.quantile(0.75)), "lab_min": float(per_lab.min()),
            "lab_max": float(per_lab.max()), "lab_mean": float(per_lab.mean()), "pooled": pooled}


def mcnemar_power(n: int, p10: float, p01: float, alpha: float = 0.05) -> float:
    """Exact power of the two-sided exact McNemar test with n pairs."""
    pd_ = p10 + p01
    if pd_ <= 0:
        return 0.0
    share = p10 / pd_
    power = 0.0
    d_pmf = stats.binom.pmf(np.arange(n + 1), n, pd_)
    for d in range(1, n + 1):
        if d_pmf[d] < 1e-14:
            continue
        b = np.arange(d + 1)
        tail = np.minimum(1.0, 2 * np.minimum(stats.binom.cdf(b, d, 0.5), stats.binom.sf(b - 1, d, 0.5)))
        reject = tail <= alpha
        if reject.any():
            power += d_pmf[d] * float(np.sum(stats.binom.pmf(b, d, share)[reject]))
    return float(power)


def minimum_detectable_difference(n: int, flip_rate: float, alpha: float = 0.05,
                                  power: float = 0.80, step: float = 0.0025) -> float:
    """Smallest delta with the required power when discordance is flip_rate + delta."""
    delta = step
    while delta <= 1 - flip_rate:
        if mcnemar_power(n, flip_rate / 2 + delta, flip_rate / 2, alpha) >= power:
            return float(delta)
        delta += step
    return float("nan")


# --------------------------------------------------------------------------
# P4  Rasch model and subset comparability
# --------------------------------------------------------------------------
def rasch_fit(att: pd.DataFrame, lam: float = 0.05) -> dict:
    """Penalised joint maximum likelihood for logit P = theta[system] - b[task]."""
    s_codes, systems = pd.factorize(att["system"])
    t_codes, tasks = pd.factorize(att["task"])
    y = att["y"].to_numpy(dtype=float)
    n_s, n_t = len(systems), len(tasks)

    def nll(par):
        theta, b = par[:n_s], par[n_s:]
        eta = theta[s_codes] - b[t_codes]
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        pen = 0.5 * lam * (np.sum(theta ** 2) + np.sum(b ** 2))
        resid = y - expit(eta)
        g_theta = np.bincount(s_codes, weights=resid, minlength=n_s) - lam * theta
        g_b = -np.bincount(t_codes, weights=resid, minlength=n_t) - lam * b
        return -(ll - pen), -np.concatenate([g_theta, g_b])

    res = optimize.minimize(nll, np.zeros(n_s + n_t), jac=True, method="L-BFGS-B",
                            options={"maxiter": 5000, "maxfun": 20000, "ftol": 1e-12, "gtol": 1e-8})
    theta, b = res.x[:n_s], res.x[n_s:]
    return {"theta": pd.Series(theta, index=systems), "b": pd.Series(b, index=tasks),
            "converged": bool(res.success), "nll": float(res.fun)}


def predict_subset(theta: float, b: pd.Series, tasks) -> float:
    """Expected resolve rate of a system with ability theta on a task subset."""
    bb = b.reindex(list(tasks)).dropna().to_numpy()
    return float(np.mean(expit(theta - bb))) if bb.size else float("nan")
