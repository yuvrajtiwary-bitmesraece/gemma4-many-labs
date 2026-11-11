"""Checks of every estimator in mlstats.py against simulated data.

Run before the estimators touch real data (DEVIATIONS.md, Amendment 1c):

    python scripts/test_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlstats as ms  # noqa: E402


def long_table(y: np.ndarray, lab: str = "L1") -> pd.DataFrame:
    """(S, T, r) array of outcomes -> long attempt table for one lab."""
    s, t, r = y.shape
    rows = []
    for i in range(s):
        for k in range(r):
            for j in range(t):
                rows.append((lab, f"{lab}/s{i}", f"{lab}/s{i}/r{k}", f"task_{j}", int(y[i, j, k])))
    return pd.DataFrame(rows, columns=["lab", "system", "run", "task", "y"])


def prob_matrix(rng, s, t, mu=-0.5, theta_sd=0.8, b_sd=1.5, inter_sd=0.7):
    theta = rng.normal(0, theta_sd, s)
    b = rng.normal(0, b_sd, t)
    inter = rng.normal(0, inter_sd, (s, t))
    return expit(mu + theta[:, None] - b[None, :] + inter), theta, b


def population_components(p: np.ndarray) -> dict:
    """What the random-effects estimators converge to for a fixed matrix p.

    With p fixed, the variance of the observed task means is the task
    component plus interaction/S (and likewise for systems), so the expected
    value of each main-effect estimator is the variance of the means of p
    minus that share of the interaction.
    """
    s, t = p.shape
    grand, row, col = p.mean(), p.mean(1), p.mean(0)
    inter = np.sum((p - row[:, None] - col[None, :] + grand) ** 2) / ((s - 1) * (t - 1))
    return {
        "system": np.sum((row - grand) ** 2) / (s - 1) - inter / t,
        "task": np.sum((col - grand) ** 2) / (t - 1) - inter / s,
        "system_x_task": inter,
        "replicate": np.mean(p * (1 - p)),
    }


def test_wilson():
    lo, hi = ms.wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3, (lo, hi)
    lo, hi = ms.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3, (lo, hi)


def test_g_study_unbiased():
    rng = np.random.default_rng(1)
    p, _, _ = prob_matrix(rng, 6, 60)
    truth = population_components(p)
    acc = {k: [] for k in truth}
    for _ in range(400):
        y = (rng.random((6, 60, 3)) < p[:, :, None]).astype(float)
        raw = ms.g_study(y)["raw"]
        for k in truth:
            acc[k].append(raw[k])
    for k, v in truth.items():
        est = float(np.mean(acc[k]))
        se = float(np.std(acc[k]) / np.sqrt(len(acc[k])))
        assert abs(est - v) < 4 * se + 1e-4, (k, est, v, se)
    return truth, {k: float(np.mean(a)) for k, a in acc.items()}


def test_flip_rate_matches_noise():
    rng = np.random.default_rng(2)
    p, _, _ = prob_matrix(rng, 8, 129)
    y = (rng.random((8, 129, 4)) < p[:, :, None]).astype(int)
    att = long_table(y)
    pairs = ms.replicate_pairs(att)
    assert len(pairs) == 8 * 6, len(pairs)
    pooled = ms.summarise_pairs(pairs)["pooled"]
    expected = 2 * np.mean(p * (1 - p))
    assert abs(pooled - expected) < 0.01, (pooled, expected)
    noise = ms.replicate_noise(att)
    assert abs(noise["noise_variance"] - np.mean(p * (1 - p))) < 0.006, noise
    return pooled, expected, noise["noise_variance"]


def test_balanced_block():
    rng = np.random.default_rng(3)
    y = (rng.random((4, 40, 3)) < 0.4).astype(int)
    att = long_table(y)
    # System 3 loses most tasks in its replicates; system 0 gets a single-run sibling.
    drop = (att["system"] == "L1/s3") & (att["task"].isin([f"task_{j}" for j in range(25, 40)]))
    att = att[~drop]
    extra = att[att["run"] == "L1/s0/r0"].assign(system="L1/solo", run="L1/solo/r0")
    att = pd.concat([att, extra], ignore_index=True)
    block, systems, tasks = ms.balanced_block(att, min_tasks=20)
    assert block.shape == (4, 25, 3), block.shape
    assert "L1/solo" not in systems
    block, systems, tasks = ms.balanced_block(att, min_tasks=30)
    assert block.shape == (3, 40, 3) and "L1/s3" not in systems, (block.shape, systems)
    assert ms.balanced_block(att[att["system"] == "L1/solo"]) is None


def test_mcnemar_power():
    rng = np.random.default_rng(4)
    n, p10, p01 = 60, 0.20, 0.05
    exact = ms.mcnemar_power(n, p10, p01)
    from scipy import stats
    hits, sims = 0, 20000
    draws = rng.multinomial(n, [p10, p01, 1 - p10 - p01], size=sims)
    for b, c, _ in draws:
        d = b + c
        if d == 0:
            continue
        pval = min(1.0, 2 * min(stats.binom.cdf(b, d, 0.5), stats.binom.sf(b - 1, d, 0.5)))
        hits += pval <= 0.05
    mc = hits / sims
    assert abs(exact - mc) < 0.012, (exact, mc)
    size = ms.mcnemar_power(129, 0.10, 0.10)
    assert size <= 0.05, size
    mde = ms.minimum_detectable_difference(129, 0.20)
    assert ms.mcnemar_power(129, 0.10 + mde, 0.10) >= 0.80
    assert ms.mcnemar_power(129, 0.10 + mde - 0.0025, 0.10) < 0.80
    return exact, mc, size, mde


def test_r2_against_dense():
    rng = np.random.default_rng(5)
    p, _, _ = prob_matrix(rng, 5, 30)
    y = (rng.random((5, 30, 2)) < p[:, :, None]).astype(int)
    att = long_table(y)
    att = att.sample(frac=0.8, random_state=0).reset_index(drop=True)  # unbalanced
    att["lab"] = np.where(att["system"].isin(["L1/s0", "L1/s1"]), "A", "B")
    w = ms.lab_balanced_weights(att)
    got = ms.r2_models(att, w)
    yv = att["y"].to_numpy(float)
    x = np.hstack([pd.get_dummies(att["task"]).to_numpy(float), pd.get_dummies(att["system"]).to_numpy(float)])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(x * sw[:, None], yv * sw, rcond=None)[0]
    mean = np.sum(w * yv) / np.sum(w)
    r2 = 1 - np.sum(w * (yv - x @ beta) ** 2) / np.sum(w * (yv - mean) ** 2)
    assert abs(got["r2_task_plus_system"] - r2) < 1e-6, (got["r2_task_plus_system"], r2)
    assert got["r2_task_plus_system"] >= max(got["r2_task"], got["r2_system"]) - 1e-9
    assert got["r2_system"] >= got["r2_lab"] - 1e-9  # systems are nested in labs
    assert abs(w.sum() - 2) < 1e-9


def test_rasch_recovers_difficulty():
    rng = np.random.default_rng(6)
    p, theta, b = prob_matrix(rng, 40, 129, inter_sd=0.0)
    y = (rng.random((40, 129, 2)) < p[:, :, None]).astype(int)
    att = long_table(y)
    fit = ms.rasch_fit(att)
    assert fit["converged"]
    b_hat = fit["b"].reindex([f"task_{j}" for j in range(129)]).to_numpy()
    r = np.corrcoef(b_hat, b)[0, 1]
    assert r > 0.97, r
    # Predicted score of the median system on an easy and a hard subset.
    order = np.argsort(b)
    ref = float(np.median(fit["theta"]))
    med_system = int(np.argsort(theta)[len(theta) // 2])
    out = []
    for name, idx in (("easy", order[:30]), ("hard", order[-30:])):
        pred = ms.predict_subset(ref, fit["b"], [f"task_{j}" for j in idx])
        true = float(p[med_system, idx].mean())
        assert abs(pred - true) < 0.05, (name, pred, true)
        out.append((name, pred, true))
    return r, out


def test_split_half():
    rng = np.random.default_rng(7)
    tasks = [f"task_{j}" for j in range(129)]
    b = rng.normal(0, 1.5, 129)

    def make(shared: bool) -> pd.DataFrame:
        frames = []
        for lab in range(14):
            bb = b if shared else rng.normal(0, 1.5, 129)
            p = expit(rng.normal(-0.5, 0.5) - bb)
            keep = rng.random(129) < 0.7
            y = (rng.random((6, 129)) < p[None, :]).astype(int)
            for s in range(6):
                frames.append(pd.DataFrame({
                    "lab": f"L{lab}", "system": f"L{lab}/s{s}", "run": f"L{lab}/s{s}/r0",
                    "task": np.array(tasks)[keep], "y": y[s, keep]}))
        return pd.concat(frames, ignore_index=True)

    res = {}
    for shared in (True, False):
        att = make(shared)
        mat = ms.lab_task_matrix(ms.lab_task_means(ms.cell_means(att)))
        res[shared] = ms.split_half_reliability(mat, n_iter=300)
        pw = ms.pairwise_lab_spearman(mat)
        res[shared]["pairwise_median"] = float(pw["rho"].median())
        cons = ms.consensus(ms.lab_task_means(ms.cell_means(att)))
        assert cons["n_labs"].max() <= 14 and len(cons) == 129
    assert res[True]["median"] > 0.85, res[True]
    assert abs(res[False]["median"]) < 0.25, res[False]
    return res[True]["median"], res[False]["median"]


def main() -> int:
    test_wilson()
    print("wilson: ok")
    truth, est = test_g_study_unbiased()
    print("g_study unbiased: ok   truth", {k: round(v, 4) for k, v in truth.items()})
    print("                       mean of 400 estimates", {k: round(v, 4) for k, v in est.items()})
    pooled, expected, noise = test_flip_rate_matches_noise()
    print(f"flip rate vs 2*noise: ok   pooled flip {pooled:.4f}, expected {expected:.4f}, noise var {noise:.4f}")
    test_balanced_block()
    print("balanced_block: ok")
    exact, mc, size, mde = test_mcnemar_power()
    print(f"mcnemar power: ok   exact {exact:.4f} vs Monte Carlo {mc:.4f}; size under null {size:.4f}; "
          f"MDE(n=129, flip=0.20) = {mde:.4f}")
    test_r2_against_dense()
    print("r2 models: ok (sparse fit equals dense weighted least squares)")
    r, subsets = test_rasch_recovers_difficulty()
    print(f"rasch: ok   corr(b_hat, b) = {r:.3f}; subsets {[(n, round(a, 3), round(b, 3)) for n, a, b in subsets]}")
    shared, indep = test_split_half()
    print(f"split-half: ok   shared difficulty {shared:.3f}, independent difficulty {indep:.3f}")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
