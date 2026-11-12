"""A noise band for a new lab, learned from the other labs. Added after the results; not in the plan.

Question: can the pooled data tell a team how far two runs of ONE agent will
differ on its own task subset, before the team has run a single repeat?

Model: an agent with ability theta solves task t with probability
p_t = expit(theta - b_t), with the task difficulties b_t taken from the pooled
Rasch fit. Two independent runs then disagree on task t with probability
2 p_t (1 - p_t). Real agents are more consistent than that, because part of
what looks like chance in a one-parameter model is a stable strength or
weakness of the agent on that task, so the predicted disagreement is scaled
by a constant c estimated from the pool (observed flips / predicted flips).

Validation is leave-one-lab-out: difficulties and c are fitted without the
held-out lab, and the band is then checked on that lab's own repeated runs.

    python scripts/noise_model.py  -> results/noise_model.json, results/tables/noise_lolo_pairs.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import mlstats as ml  # noqa: E402

ROOT = HERE.parent


def theta_for_rate(rate: float, b: np.ndarray) -> float:
    """Ability at which the mean solve probability on tasks with difficulties b equals rate."""
    rate = min(max(rate, 0.5 / (2 * len(b))), 1 - 0.5 / (2 * len(b)))     # keep away from 0 and 1
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(mid - b).mean() < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pair_table(att: pd.DataFrame, b: pd.Series) -> pd.DataFrame:
    """One row per pair of repeated runs: observed flips and the Rasch prediction before scaling."""
    att = ml.first_attempt_per_run(att)
    rows = []
    for (lab, system), g in att.groupby(["lab", "system"], sort=False):
        runs = list(dict.fromkeys(g["run"]))
        if len(runs) < 2:
            continue
        wide = g.pivot(index="task", columns="run", values="y")
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                pair = wide[[runs[i], runs[j]]].dropna()
                pair = pair[pair.index.isin(b.index)]
                if len(pair) < 10:
                    continue
                ya, yb = pair[runs[i]].to_numpy(), pair[runs[j]].to_numpy()
                bb = b.reindex(pair.index).to_numpy()
                theta = theta_for_rate((ya.mean() + yb.mean()) / 2, bb)
                p = expit(theta - bb)
                rows.append({"lab": lab, "system": system, "run_a": runs[i], "run_b": runs[j], "n": len(pair),
                             "k_a": int(ya.sum()), "k_b": int(yb.sum()), "flips": int((ya != yb).sum()),
                             "pred_flips_raw": float(np.sum(2 * p * (1 - p)))})
    return pd.DataFrame(rows)


def summarise(pairs: pd.DataFrame, c_col: str) -> dict:
    v = pairs[c_col] * pairs["pred_flips_raw"]
    diff = (pairs["k_a"] - pairs["k_b"]).to_numpy(float)
    ok = v.to_numpy() > 1e-9
    z = np.where(ok, diff / np.sqrt(np.where(ok, v, 1.0)), 0.0)
    inside = np.abs(z) <= 1.959964
    per_lab = pd.DataFrame({"lab": pairs["lab"], "inside": inside}).groupby("lab")["inside"].mean()
    return {
        "n_pairs": int(len(pairs)), "n_labs": int(pairs["lab"].nunique()),
        "observed_flips": int(pairs["flips"].sum()), "predicted_flips": float(v.sum()),
        "ratio_observed_to_predicted_flips": float(pairs["flips"].sum() / v.sum()),
        "pearson_predicted_vs_observed_flips": float(np.corrcoef(v, pairs["flips"])[0, 1]),
        "sd_of_standardised_score_difference": float(np.std(z[ok], ddof=1)) if ok.sum() > 2 else None,
        "coverage_of_95_band": float(inside.mean()),
        "coverage_of_95_band_mean_over_labs": float(per_lab.mean()),
        "coverage_by_lab": {k: float(x) for k, x in per_lab.items()},
        "mean_abs_score_difference_observed": float(np.mean(np.abs(diff) / pairs["n"])),
        "mean_abs_score_difference_predicted": float(np.mean(np.sqrt(v) * np.sqrt(2 / np.pi) / pairs["n"])),
    }


def main() -> int:
    att = analyze.load()
    codes = analyze.lab_codes(att)
    fit = ml.rasch_fit(att)
    pairs_all = pair_table(att, fit["b"])
    c_all = float(pairs_all["flips"].sum() / pairs_all["pred_flips_raw"].sum())
    pairs_all["c"] = c_all
    out = {"scale_c_all_labs": c_all, "in_sample": summarise(pairs_all, "c")}
    pairs_all["one"] = 1.0
    out["unscaled_rasch_in_sample"] = summarise(pairs_all, "one")

    # Leave one lab out: difficulties and the scale come from the other labs only.
    held = []
    scales = {}
    for lab in sorted(pairs_all["lab"].unique()):
        rest = att[att["lab"] != lab]
        b = ml.rasch_fit(rest)["b"]
        train = pair_table(rest, b)
        c = float(train["flips"].sum() / train["pred_flips_raw"].sum())
        test = pair_table(att[att["lab"] == lab], b)
        if test.empty:
            continue
        test["c"] = c
        scales[codes[lab]] = c
        held.append(test)
        print(f"held out {codes[lab]}: {len(test)} pairs, scale from the other labs {c:.3f}", flush=True)
    lolo = pd.concat(held, ignore_index=True)
    out["leave_one_lab_out"] = summarise(lolo, "c")
    out["leave_one_lab_out"]["coverage_by_lab"] = {codes[k]: v for k, v in out["leave_one_lab_out"]["coverage_by_lab"].items()}
    out["in_sample"]["coverage_by_lab"] = {codes[k]: v for k, v in out["in_sample"]["coverage_by_lab"].items()}
    out["unscaled_rasch_in_sample"].pop("coverage_by_lab", None)
    out["scale_c_leave_one_lab_out"] = scales

    # The same band applied to the external dataset (difficulties and scale from the pool only).
    ext_path = ROOT / "data" / "derived" / "external_zenodo_attempts.csv"
    if ext_path.exists():
        ext = analyze.load(ext_path)
        test = pair_table(ext, fit["b"])
        test["c"] = c_all
        out["external_dataset"] = summarise(test, "c")
        out["external_dataset"].pop("coverage_by_lab", None)
        by_model = {}
        model = ext.groupby("system")["model"].first()
        test["model"] = test["system"].map(model)
        for m, g in test.groupby("model"):
            s = summarise(g, "c")
            by_model[m] = {"n_pairs": s["n_pairs"], "ratio_observed_to_predicted_flips": s["ratio_observed_to_predicted_flips"],
                           "coverage_of_95_band": s["coverage_of_95_band"]}
        out["external_dataset"]["by_model"] = by_model

    lolo.assign(lab=lolo["lab"].map(codes)).drop(columns=["system", "run_a", "run_b"]).to_csv(
        ROOT / "results" / "tables" / "noise_lolo_pairs.csv", index=False, lineterminator="\n")
    fit["b"].rename("rasch_b").rename_axis("task").to_csv(ROOT / "results" / "tables" / "task_difficulty.csv", lineterminator="\n")

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 5)
        if isinstance(x, np.integer):
            return int(x)
        return x

    out = clean(out)
    (ROOT / "results" / "noise_model.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    # Everything tools/compare_runs.py needs, in one small file next to it.
    band = {"what": "Rasch task difficulties and noise scale from the pooled table of the many-labs study",
            "n_attempts": int(len(att)), "n_labs": int(att["lab"].nunique()), "scale": round(c_all, 5),
            "leave_one_lab_out_coverage_of_95_band": out["leave_one_lab_out"]["coverage_of_95_band"],
            "difficulty": {t: round(float(v), 4) for t, v in fit["b"].sort_index().items()}}
    (ROOT / "tools").mkdir(exist_ok=True)
    (ROOT / "tools" / "noise_band.json").write_text(json.dumps(band, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k, v in out.items():
        print(k, json.dumps(v)[:700], flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
