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


def replicate_noise(att: pd.DataFrame, weights: np.ndarray | None = None, lab_balanced: bool = False) -> dict:
    """Unbiased within-cell variance on cells with at least two attempts.

    Returns the weighted mean within-cell variance, the total outcome variance
    on the same attempts, and their ratio (the replicate-noise share).

    lab_balanced=True gives every lab the same total weight ON THE REPLICATED
    CELLS (weights computed on the whole table do not: a lab with few repeats
    would count for little).  `weights` is then ignored.
    """
    w = np.ones(len(att)) if weights is None else np.asarray(weights, dtype=float)
    cols = ["system", "task", "y"] + (["lab"] if lab_balanced else [])
    df = att[cols].copy()
    df["w"] = w
    g = df.groupby(["system", "task"], sort=False)
    n = g["y"].transform("size").to_numpy()
    keep = n >= 2
    if not keep.any():
        return {"n_cells": 0}
    sub = df.loc[keep].copy()
    if lab_balanced:
        sub["w"] = 1.0 / sub.groupby("lab")["y"].transform("size").to_numpy(dtype=float)
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


def se_difference(comp: dict, n_tasks: int, n_runs: int | float = 1) -> float:
    """Standard error of the score difference of two systems on shared tasks.

    Tasks are treated as a random sample (the system x task term stays however
    many runs are made); n_runs may be math.inf.
    """
    return math.sqrt(2 * (comp["system_x_task"] + comp["replicate"] / n_runs) / n_tasks)


Z_80 = 1.959964 + 0.841621      # two-sided 5% test, 80% power (normal approximation)


def detectable_random_tasks(comp: dict, n_tasks: int, n_runs: int | float = 1) -> float:
    """Smallest true difference between two systems detected with 80% power when the
    conclusion is meant to hold for new tasks of the same kind (tasks random)."""
    return Z_80 * se_difference(comp, n_tasks, n_runs)


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


def mdd_with_discordance(n: int, flip_rate: float, base: float = 0.0, slope: float = 1.0, alpha: float = 0.05,
                         power: float = 0.80, step: float = 0.0025) -> float:
    """Smallest improvement delta detected by the exact McNemar test when two different systems
    disagree on a share  flip_rate + base + slope * delta  of the tasks.

    slope = 1, base = 0 is the planned alternative (every gained task is an all-or-nothing change,
    so the gain adds its whole size to the disagreement); slope = 0 is the other extreme (the gain
    is spread thinly over many tasks and adds nothing).  The disagreement is never below delta.
    """
    delta = step
    while delta < 1:
        disc = max(flip_rate + base + slope * delta, delta)
        if disc > 1:
            break
        if mcnemar_power(n, (disc + delta) / 2, (disc - delta) / 2, alpha) >= power:
            return float(delta)
        delta += step
    return float("nan")


# --------------------------------------------------------------------------
# Repeated runs of two systems on the same tasks
# --------------------------------------------------------------------------
def detectable_fixed_tasks(flip_rate: float, n_tasks: int, k: int, base: float = 0.0, slope: float = 1.0,
                           alpha_z: float = 1.959964, power_z: float = 0.841621) -> float:
    """Smallest gain detected with 80% power by a test stratified by task, k runs per system.

    Tasks are fixed (the question is about these tasks).  With per-task success probabilities
    p (old system) and q (new system), v = p(1-p) + q(1-q) averaged over tasks is the flip rate
    and e = (p-q)^2 averaged over tasks is the extra disagreement between the two systems,
    taken as  base + slope * delta  (slope = 1, base = 0: the planned all-or-nothing alternative).

    The mean score difference has sampling variance  v / (k n).  The test that permutes run
    labels within a task (Cochran-Mantel-Haenszel; exact McNemar when k = 1) has null variance
    (v / k + e / (2k - 1)) / n.  The detectable gain solves
        delta = z_alpha * sd_null(delta) + z_power * sd_sampling.
    """
    sd1 = math.sqrt(flip_rate / (k * n_tasks))
    delta = 0.0
    for _ in range(200):
        e = max(base + slope * delta, delta * delta)
        new = alpha_z * math.sqrt((flip_rate / k + e / (2 * k - 1)) / n_tasks) + power_z * sd1
        if abs(new - delta) < 1e-10:
            break
        delta = new
    return float(delta)


def runs_needed_fixed_tasks(flip_rate: float, n_tasks: int, gain: float, base: float = 0.0, slope: float = 1.0,
                            k_max: int = 500) -> int | None:
    """Runs per system so that `gain` is detected with 80% power (tasks fixed); None if more than k_max."""
    for k in range(1, k_max + 1):
        if detectable_fixed_tasks(flip_rate, n_tasks, k, base, slope) <= gain:
            return k
    return None


def cmh_power_simulation(p: np.ndarray, q: np.ndarray, k: int, n_sim: int = 4000, seed: int = 20261111,
                         alpha_z: float = 1.959964) -> float:
    """Simulated power of the task-stratified (Cochran-Mantel-Haenszel) test, two-sided, no continuity
    correction, for per-task success probabilities p and q and k runs per system."""
    rng = np.random.default_rng(seed)
    p, q = np.asarray(p, float), np.asarray(q, float)
    sa = rng.binomial(k, p, size=(n_sim, p.size))
    sb = rng.binomial(k, q, size=(n_sim, q.size))
    s = sa + sb
    num = (sb - s / 2.0).sum(1)                                   # observed minus expected successes of the new system
    var = (s * (2 * k - s) / (4.0 * (2 * k - 1))).sum(1)          # hypergeometric variance per task
    z = np.divide(num, np.sqrt(var), out=np.zeros_like(num, dtype=float), where=var > 0)
    return float(np.mean(np.abs(z) > alpha_z))


def exact_stratified_power(p: np.ndarray, q: np.ndarray, k: int, n_sim: int = 4000, seed: int = 20261111,
                           alpha: float = 0.05) -> float:
    """Simulated power of the EXACT permutation test stratified by task (run labels are permuted within
    each task; with k = 1 this is the exact McNemar test), two-sided as in mcnemar_power."""
    rng = np.random.default_rng(seed)
    p, q = np.asarray(p, float), np.asarray(q, float)
    sa = rng.binomial(k, p, size=(n_sim, p.size))
    sb = rng.binomial(k, q, size=(n_sim, q.size))
    s = sa + sb
    pmfs = {t: stats.hypergeom.pmf(np.arange(0, k + 1), 2 * k, t, k) for t in range(1, 2 * k)}
    cache = {}
    rejected = 0
    for i in range(n_sim):
        counts = tuple(np.bincount(s[i], minlength=2 * k + 1)[1:2 * k])
        dist = cache.get(counts)
        if dist is None:
            dist = np.array([1.0])
            for t, c in enumerate(counts, start=1):
                for _ in range(int(c)):
                    dist = np.convolve(dist, pmfs[t])
            if len(cache) < 20000:
                cache[counts] = dist
        informative = (s[i] > 0) & (s[i] < 2 * k)
        obs = int(sb[i][informative].sum())
        pval = min(1.0, 2 * min(dist[: obs + 1].sum(), dist[obs:].sum()))
        rejected += pval <= alpha
    return rejected / n_sim


def planning_tasks(flip_rate: float, n_tasks: int, share_always_solved: float = 0.2) -> np.ndarray:
    """Per-task success probabilities with a given mean flip rate, for planning calculations:
    a few tasks are 'flaky' (all with the same probability), some are always solved, the rest never."""
    m = int(math.ceil(2 * flip_rate * n_tasks - 1e-9))
    m = min(max(m, 1), n_tasks)
    f = min(flip_rate * n_tasks / m, 0.5)                 # flip rate of one flaky task
    p = np.zeros(n_tasks)
    p[:m] = (1 - math.sqrt(1 - 2 * f)) / 2
    solved = min(int(round(share_always_solved * n_tasks)), n_tasks - m)
    p[m:m + solved] = 1.0
    return p


def detectable_exact_fixed_tasks(flip_rate: float, n_tasks: int, k: int, n_sim: int = 4000,
                                 seed: int = 20261111) -> float:
    """Smallest gain detected with 80% power by the exact stratified test with k runs per system, when the
    gain is g previously never-solved tasks that the new system always solves (the planned alternative).
    Found by simulation over g and interpolated linearly between the two neighbouring task counts."""
    p = planning_tasks(flip_rate, n_tasks)
    free = np.where(p == 0)[0]
    start = max(1, int(detectable_fixed_tasks(flip_rate, n_tasks, k) * n_tasks) - 2)

    def power(g):
        q = p.copy()
        q[free[:g]] = 1.0
        return exact_stratified_power(p, q, k, n_sim=n_sim, seed=seed + 1000 * k + g)

    g = min(start, len(free))
    cur = power(g)
    while cur >= 0.80 and g > 1:                    # started at or above the threshold: step down
        below = power(g - 1)
        if below < 0.80:
            frac = (0.80 - below) / (cur - below) if cur > below else 1.0
            return float((g - 1 + frac) / n_tasks)
        g, cur = g - 1, below
    if cur >= 0.80:
        return float(1.0 / n_tasks)
    while g < len(free):                            # step up
        above = power(g + 1)
        if above >= 0.80:
            frac = (0.80 - cur) / (above - cur) if above > cur else 1.0
            return float((g + frac) / n_tasks)
        g, cur = g + 1, above
    return float("nan")


# --------------------------------------------------------------------------
# Disagreement between two DIFFERENT systems against the size of their difference
# --------------------------------------------------------------------------
def cross_system_discordance(att: pd.DataFrame, min_shared: int = 30) -> pd.DataFrame:
    """For every pair of systems of one lab and one model size that both have repeats.

    On each shared task with at least two runs of each system, with p and q the solve rates of
    the two systems over their k_a and k_b runs:
        within  = unbiased flip rate of each system, 2 k/(k-1) p(1-p), averaged over the two;
        cross   = p(1-q) + q(1-p), the chance that one run of each disagree (unbiased);
        excess  = cross - within, an unbiased estimate of (p_t - q_t)^2.
    Returns one row per pair with the means over tasks and delta = mean(q - p).
    """
    att = first_attempt_per_run(att)
    cell = att.groupby(["lab", "system", "task"], sort=False)["y"].agg(p="mean", k="size").reset_index()
    cell = cell[cell["k"] >= 2]
    model = att.groupby("system")["model"].first() if "model" in att.columns else None
    rows = []
    for lab, g in cell.groupby("lab", sort=False):
        systems = sorted(g["system"].unique())
        wide_p = g.pivot(index="task", columns="system", values="p")
        wide_k = g.pivot(index="task", columns="system", values="k")
        for a, b in itertools.combinations(systems, 2):
            if model is not None and model[a] != model[b]:
                continue
            ok = wide_p[a].notna() & wide_p[b].notna()
            if int(ok.sum()) < min_shared:
                continue
            p, q = wide_p.loc[ok, a].to_numpy(), wide_p.loc[ok, b].to_numpy()
            ka, kb = wide_k.loc[ok, a].to_numpy(), wide_k.loc[ok, b].to_numpy()
            within = 0.5 * (2 * ka / (ka - 1) * p * (1 - p) + 2 * kb / (kb - 1) * q * (1 - q))
            cross = p * (1 - q) + q * (1 - p)
            rows.append({"lab": lab, "system_a": a, "system_b": b, "n_shared": int(ok.sum()),
                         "model": "" if model is None else model[a],
                         "within_flip": float(within.mean()), "cross_discordance": float(cross.mean()),
                         "excess": float((cross - within).mean()), "delta": float((q - p).mean())})
    return pd.DataFrame(rows, columns=["lab", "system_a", "system_b", "n_shared", "model", "within_flip",
                                       "cross_discordance", "excess", "delta"])


def excess_discordance_fit(pairs: pd.DataFrame) -> dict:
    """Weighted least squares of the excess disagreement on |delta| (weights: shared tasks)."""
    if len(pairs) < 3:
        return {"n_pairs": int(len(pairs))}
    x, y, w = pairs["delta"].abs().to_numpy(), pairs["excess"].to_numpy(), pairs["n_shared"].to_numpy(float)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    sxx = float(np.sum(w * (x - xm) ** 2))
    slope = float(np.sum(w * (x - xm) * (y - ym)) / sxx) if sxx > 0 else float("nan")
    return {"n_pairs": int(len(pairs)), "n_labs": int(pairs["lab"].nunique()),
            "mean_within_flip": float(np.average(pairs["within_flip"], weights=w)),
            "mean_cross_discordance": float(np.average(pairs["cross_discordance"], weights=w)),
            "mean_excess": float(ym), "mean_abs_delta": float(xm),
            "mean_delta_squared": float(np.average(pairs["delta"] ** 2, weights=w)),
            "intercept": float(ym - slope * xm), "slope": slope}


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


def rasch_cv(att: pd.DataFrame, n_folds: int = 5, seed: int = 20261111, lam: float = 0.05) -> dict:
    """Cross-validation of the Rasch model over (system, task) cells.

    Every cell (all attempts of one system on one task) is held out once; the model is fitted on
    the other cells and predicts the held-out attempts.  In-sample checks are empty for this
    model: its score equations force predicted and observed totals per system to agree.
    """
    att = att.reset_index(drop=True)
    cell_id = pd.factorize(pd.MultiIndex.from_frame(att[["system", "task"]]))[0]
    n_cells = int(cell_id.max()) + 1
    rng = np.random.default_rng(seed)
    fold_of_cell = rng.permutation(n_cells) % n_folds
    fold = fold_of_cell[cell_id]
    y = att["y"].to_numpy(float)
    p_model, p_task = np.full(len(att), np.nan), np.full(len(att), np.nan)
    for k in range(n_folds):
        test = np.where(fold == k)[0]
        train = att.iloc[np.where(fold != k)[0]]
        fit = rasch_fit(train, lam)
        theta = fit["theta"].reindex(att["system"].iloc[test]).fillna(0.0).to_numpy()
        b = fit["b"].reindex(att["task"].iloc[test]).fillna(0.0).to_numpy()
        p_model[test] = expit(theta - b)
        task_mean = train.groupby("task")["y"].mean()
        p_task[test] = task_mean.reindex(att["task"].iloc[test]).fillna(train["y"].mean()).to_numpy()
    full = rasch_fit(att, lam)
    p_in = expit(full["theta"].reindex(att["system"]).to_numpy() - full["b"].reindex(att["task"]).to_numpy())
    runs = pd.DataFrame({"run": att["run"].to_numpy(), "y": y, "p": p_model}).groupby("run").agg(
        y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    dec = pd.DataFrame({"y": y, "p": p_model})
    dec["bin"] = pd.qcut(dec["p"], 10, labels=False, duplicates="drop")
    cal = dec.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return {"n_folds": n_folds, "n_cells": n_cells, "n_attempts": int(len(att)),
            "brier_out_of_sample": float(np.mean((y - p_model) ** 2)),
            "brier_in_sample": float(np.mean((y - p_in) ** 2)),
            "brier_task_means_only_out_of_sample": float(np.mean((y - p_task) ** 2)),
            "brier_base_rate": float(np.var(y)),
            "run_level_pearson": float(np.corrcoef(runs["y"], runs["p"])[0, 1]),
            "run_level_mean_abs_error": float(np.mean(np.abs(runs["y"] - runs["p"]))),
            "calibration_max_abs_gap_by_decile": float(np.max(np.abs(cal["predicted"] - cal["observed"]))),
            "calibration_by_decile": [[float(a), float(b), int(c)] for a, b, c in
                                      zip(cal["predicted"], cal["observed"], cal["n"])]}


def _ability_given_difficulty(y: np.ndarray, b: np.ndarray, lam: float = 0.05) -> float:
    """Penalised maximum-likelihood ability of one system for known task difficulties."""
    theta = 0.0
    for _ in range(100):
        pr = expit(theta - b)
        grad = float(np.sum(y - pr) - lam * theta)
        hess = float(-np.sum(pr * (1 - pr)) - lam)
        step = grad / hess
        theta -= step
        if abs(step) < 1e-9:
            break
    return theta


def rasch_subset_check(att: pd.DataFrame, subsets: list, min_tasks_system: int = 85, min_in: int = 10,
                       min_out: int = 10, lam: float = 0.05) -> tuple[dict, pd.DataFrame]:
    """Out-of-sample check that the model predicts how a score moves with the task subset.

    For a system s of lab L that attempted at least `min_tasks_system` tasks, and a subset S:
    task difficulties come from a fit WITHOUT lab L; the ability of s is fitted on its attempts
    OUTSIDE S; the model then predicts the rate of s on its tasks INSIDE S.  The naive prediction
    is the rate of s on its other tasks (no subset effect).
    """
    n_tasks = att.groupby("system")["task"].nunique()
    big = n_tasks[n_tasks >= min_tasks_system].index
    lab_of = att.groupby("system")["lab"].first()
    rows = []
    for lab in sorted(set(lab_of[big])):
        b = rasch_fit(att[att["lab"] != lab], lam)["b"]
        for system in [s for s in big if lab_of[s] == lab]:
            g = att[att["system"] == system]
            g = g[g["task"].isin(b.index)]
            bt = b.reindex(g["task"]).to_numpy()
            yy = g["y"].to_numpy(float)
            tasks = g["task"].to_numpy()
            for i, s in enumerate(subsets):
                inside = np.isin(tasks, list(s))
                if len(set(tasks[inside])) < min_in or len(set(tasks[~inside])) < min_out:
                    continue
                theta = _ability_given_difficulty(yy[~inside], bt[~inside], lam)
                rows.append({"lab": lab, "system": system, "subset": i, "n_tasks_in": len(set(tasks[inside])),
                             "n_tasks_out": len(set(tasks[~inside])),
                             "observed_in": float(yy[inside].mean()), "observed_out": float(yy[~inside].mean()),
                             "predicted_in": float(expit(theta - bt[inside]).mean()),
                             "predicted_out": float(expit(theta - bt[~inside]).mean())})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return {"n_pairs": 0}, tab
    eff_obs = tab["observed_in"] - tab["observed_out"]
    eff_pred = tab["predicted_in"] - tab["predicted_out"]
    slope = float(np.sum((eff_pred - eff_pred.mean()) * (eff_obs - eff_obs.mean())) /
                  np.sum((eff_pred - eff_pred.mean()) ** 2))
    # One number per system first, so that systems with many eligible subsets do not dominate.
    per_system = tab.assign(err=(tab["observed_in"] - tab["predicted_in"]).abs(),
                            naive=(tab["observed_in"] - tab["observed_out"]).abs()).groupby("system")[["err", "naive"]].mean()
    out = {"n_pairs": int(len(tab)), "n_systems": int(tab["system"].nunique()), "n_labs": int(tab["lab"].nunique()),
           "min_tasks_system": min_tasks_system,
           "mean_abs_error_model": float((tab["observed_in"] - tab["predicted_in"]).abs().mean()),
           "mean_abs_error_naive": float((tab["observed_in"] - tab["observed_out"]).abs().mean()),
           "mean_abs_error_model_by_system": float(per_system["err"].mean()),
           "mean_abs_error_naive_by_system": float(per_system["naive"].mean()),
           "correlation_of_subset_effects": float(np.corrcoef(eff_obs, eff_pred)[0, 1]),
           "slope_observed_on_predicted_effect": slope}
    return out, tab
