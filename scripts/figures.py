"""Draw the figures from results/results.json and results/tables/*.csv.

Every figure has a table twin in results/tables/ (named in each function).
Colours follow one validated palette: categorical slots in fixed order,
emphasis as one hue plus grey, text always in ink colours.

    python scripts/figures.py [results_dir [figures_dir]]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlstats as ml  # noqa: E402

ROOT = HERE.parent

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#184f95"
GREY = "#b9b8b1"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 22,
    "axes.labelcolor": INK2, "axes.labelsize": 10, "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.0, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=INK2,
            linespacing=1.3)
    # The subtitle sits between the title and the axes; title pad leaves room for one or two lines.


def pct(ax, axis: str = "y") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


# -- Figure 1: consensus difficulty (table twin: tables/tasks.csv, tables/lab_task_matrix.csv)
def fig_task_difficulty(res, tables: Path, out: Path) -> None:
    t = pd.read_csv(tables / "tasks.csv")
    mat = pd.read_csv(tables / "lab_task_matrix.csv", index_col=0)
    t = t[t["n_labs"] >= 5].sort_values(["d", "task"]).reset_index(drop=True)
    lo = mat[t["task"]].quantile(0.25, axis=0).to_numpy()
    hi = mat[t["task"]].quantile(0.75, axis=0).to_numpy()
    x = np.arange(1, len(t) + 1)
    p1 = res["P1"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.vlines(x, lo, hi, color=GREY, linewidth=1.2, zorder=1, label="Middle half of the labs' solve rates")
    if "audit_status" in t.columns and t["audit_status"].notna().any():
        groups = [("Valid in the sandbox audit", t["audit_status"] == "valid", BLUE),
                  ("Reference patch fails in the audit", t["audit_status"].astype(str).str.startswith("reference_patch_fails"), ORANGE),
                  ("Passes with no patch in the audit", t["audit_status"] == "passes_without_patch", AQUA)]
        for label, mask, color in groups:
            ax.scatter(x[mask], t.loc[mask, "d"], s=30, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label=f"{label} ({int(mask.sum())})")
        x1 = res.get("X1_validity", {})
        sub = (f"{p1['floor_tasks']} of {len(t)} tasks have a consensus solve rate of 10% or less; "
               f"{x1.get('floor_tasks_that_are_valid_in_reference_audit', '?')} of those are valid in the sandbox audit")
    else:
        disc = (t["d"] > 0.10) & (t["d"] < 0.90)
        ax.scatter(x[~disc], t.loc[~disc, "d"], s=30, color=MUTED, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   label="Consensus: floor or ceiling task")
        ax.scatter(x[disc], t.loc[disc, "d"], s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=4,
                   label="Consensus: discriminating task")
        sub = (f"{p1['floor_tasks']} tasks at the floor (consensus at most 10%), {p1['discriminating_tasks']} in "
               f"between, {p1['ceiling_tasks']} at the ceiling (at least 90%)")
    for y in (0.10, 0.90):
        ax.axhline(y, color=BASELINE, linewidth=1.0, zorder=0)
        ax.text(len(t) + 1.5, y, f"{100 * y:.0f}%", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, len(t) + 1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(f"Development-set tasks attempted by at least 5 labs, ordered by consensus solve rate (n = {len(t)})")
    ax.set_ylabel("Solve rate")
    pct(ax)
    ax.grid(axis="x", visible=False)
    ax.set_title("Task difficulty across labs")
    subtitle(ax, sub)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, handletextpad=0.6, columnspacing=2.0)
    save(fig, out / "fig1_task_difficulty.png")


# -- Figure 2: attempts on tasks that cannot pass (table twin: tables/validity_by_lab.csv)
def fig_unpassable(res, tables: Path, out: Path) -> None:
    path = tables / "validity_by_lab.csv"
    if not path.exists():
        return
    v = pd.read_csv(path)
    v["bad"] = v["audit_status"].astype(str).str.startswith("reference_patch_fails")
    tot = v.groupby("lab")["attempts"].sum()
    bad = v[v["bad"]].groupby("lab")[["attempts", "resolved"]].sum().reindex(tot.index).fillna(0)
    d = pd.DataFrame({"share": bad["attempts"] / tot, "attempts": bad["attempts"].astype(int),
                      "resolved": bad["resolved"].astype(int)}).sort_values("share")
    fig, ax = plt.subplots(figsize=(9.2, 0.36 * len(d) + 1.9))
    y = np.arange(len(d))
    ax.barh(y, d["share"], height=0.5, color=BLUE, zorder=2)
    for yi, (share, n, k) in enumerate(zip(d["share"], d["attempts"], d["resolved"])):
        text = "none attempted" if n == 0 else f"{100 * share:.0f}% of attempts  ·  {k} of {n} resolved"
        ax.text(share + 0.008, yi, text, va="center", ha="left", fontsize=8.5, color=INK2, zorder=3)
    ax.set_yticks(y, d.index)
    ax.set_xlim(0, max(0.1, float(d["share"].max()) * 1.55))
    pct(ax, "x")
    ax.set_xlabel("Share of the lab's attempts")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    x1 = res["X1_validity"]
    n_bad = x1["status_counts"].get("reference_patch_fails", 0)
    ax.set_title("Attempts spent on tasks whose reference patch fails in the standard sandbox")
    subtitle(ax, f"{n_bad} of 129 tasks fail with their own reference patch in the sandbox audit; "
                 f"{100 * x1['share_of_attempts_on_tasks_where_reference_patch_fails']:.0f}% of all pooled attempts were made on them")
    save(fig, out / "fig2_unpassable_attempts.png")


# -- Figure 3: variance components per lab (table twin: tables/g_study.csv)
def fig_variance(res, tables: Path, out: Path) -> None:
    g = pd.read_csv(tables / "g_study.csv")
    if g.empty:
        return
    g = g.sort_values("share_replicate").reset_index(drop=True)
    parts = [("share_task", "Task", BLUE, "#ffffff"), ("share_system", "System", ORANGE, "#ffffff"),
             ("share_system_x_task", "System × task", AQUA, "#ffffff"),
             ("share_replicate", "Run-to-run noise", YELLOW, INK)]
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(g) + 1.9))
    y = np.arange(len(g))
    left = np.zeros(len(g))
    for col, label, color, txt in parts:
        v = g[col].to_numpy()
        ax.barh(y, v, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=2, label=label, zorder=2)
        for yi, (l0, w) in enumerate(zip(left, v)):
            if w >= 0.07:
                ax.text(l0 + w / 2, yi, f"{100 * w:.0f}%", ha="center", va="center", fontsize=8.5, color=txt, zorder=3)
        left += v
    labels = [f"{c}  ({s} systems, {t} tasks, {r} runs each)"
              for c, s, t, r in zip(g["code"], g["n_systems"], g["n_tasks"], g["n_replicates"])]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    pct(ax, "x")
    ax.set_xlabel("Share of outcome variance")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Where the variance in outcomes comes from, within each lab")
    subtitle(ax, "Two-way random-effects decomposition on each lab's balanced systems × tasks × repeats block")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.36 if len(g) < 6 else -0.2 - 0.4 / len(g)), ncol=4,
              columnspacing=1.6, handlelength=1.2)
    save(fig, out / "fig3_variance_components.png")


# -- Figure 4: churn against net change (table twin: tables/replicate_pairs.csv)
def fig_replicates(res, tables: Path, out: Path) -> None:
    pairs = pd.read_csv(tables / "replicate_pairs.csv")
    if pairs.empty:
        return
    per_sys = pairs.groupby(["lab", "system"])[["flip_rate", "abs_score_diff"]].mean()
    per_lab = per_sys.groupby("lab").mean()
    n_pairs = pairs.groupby("lab").size()
    per_lab = per_lab.sort_values("flip_rate").reset_index()
    fig, ax = plt.subplots(figsize=(9.2, 0.36 * len(per_lab) + 1.9))
    y = np.arange(len(per_lab))
    ax.hlines(y, per_lab["abs_score_diff"], per_lab["flip_rate"], color=GREY, linewidth=2, zorder=1)
    ax.scatter(per_lab["abs_score_diff"], y, s=58, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="Difference in resolve rate between two repeats")
    ax.scatter(per_lab["flip_rate"], y, s=58, color=BLUE_DARK, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="Share of tasks whose outcome flipped")
    ax.set_yticks(y, [f"{lab}  ({n_pairs[lab]} pair{'s' if n_pairs[lab] != 1 else ''})" for lab in per_lab["lab"]])
    ax.set_xlim(-0.005, max(0.05, float(per_lab["flip_rate"].max()) * 1.12))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    pct(ax, "x")
    ax.set_xlabel("Mean over the lab's pairs of repeated runs")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fr = res["P3"]["flip_rate"]
    ax.set_title("Two runs of the same agent: small change in score, large change in which tasks pass")
    subtitle(ax, f"{fr['n_pairs']} pairs of repeated runs from {fr['n_labs']} labs; "
                 f"pooled flip rate {100 * fr['pooled']:.1f}%")
    ax.legend(loc="lower right", ncol=1, handletextpad=0.5)
    save(fig, out / "fig4_replicate_noise.png")


def mdd_bisect(n: int, flip: float, power: float = 0.80, tol: float = 0.0025) -> float:
    """Minimum detectable difference by bisection (the figure's curve)."""
    lo, hi = 0.0, 1.0 - flip
    if ml.mcnemar_power(n, flip / 2 + hi, flip / 2) < power:
        return float("nan")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if ml.mcnemar_power(n, flip / 2 + mid, flip / 2) >= power:
            hi = mid
        else:
            lo = mid
    return hi


# -- Figure 5: detectable difference against benchmark size (table twin: tables/mdd_curve.csv)
def fig_mdd(res, tables: Path, out: Path) -> None:
    fr = res["P3"]["flip_rate"]
    if not fr.get("pooled"):
        return
    flip = fr["pooled"]
    ns = [20, 30, 40, 50, 60, 80, 100, 129, 160, 200, 250, 300, 400, 500]
    curve = pd.DataFrame({"n_tasks": ns, "mdd": [mdd_bisect(n, flip) for n in ns]})
    curve.to_csv(tables / "mdd_curve.csv", index=False, lineterminator="\n")
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(curve["n_tasks"], curve["mdd"], color=BLUE, zorder=2)
    mark = curve[curve["n_tasks"].isin([30, 60, 129])]
    ax.scatter(mark["n_tasks"], mark["mdd"], s=62, color=BLUE, edgecolor=SURFACE, linewidth=1.8, zorder=3)
    for n, v in zip(mark["n_tasks"], mark["mdd"]):
        ax.annotate(f"{n} tasks: {100 * v:.0f} points", (n, v), xytext=(9, 7), textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.set_xlim(0, 520)
    ax.set_ylim(0, float(np.nanmax(curve["mdd"])) * 1.1)
    pct(ax)
    ax.set_xlabel("Tasks in the comparison")
    ax.set_ylabel("Smallest detectable improvement")
    ax.set_title("What one paired comparison of two agents can detect")
    subtitle(ax, f"Exact two-sided McNemar test, α = 0.05, power 0.80, at the pooled flip rate of {100 * flip:.1f}%")
    save(fig, out / "fig5_detectable_difference.png")


# -- Figure 6: one agent on the task subsets labs used (table twin: tables/subset_predictions.csv)
def fig_subsets(res, tables: Path, out: Path) -> None:
    s = pd.read_csv(tables / "subset_predictions.csv")
    if s.empty:
        return
    p4 = res["P4"]
    full = p4["predicted_full_task_set"]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.axhline(full, color=BASELINE, linewidth=1.2, zorder=1)
    ax.scatter(s["n_tasks"], s["predicted_rate"], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               alpha=0.9)
    ax.annotate(f"all {p4['n_tasks']} tasks: {100 * full:.0f}%", (float(s["n_tasks"].max()), full), xytext=(4, 9),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    lo, hi = s.loc[s["predicted_rate"].idxmin()], s.loc[s["predicted_rate"].idxmax()]
    ax.annotate(f"{100 * hi['predicted_rate']:.0f}% on {int(hi['n_tasks'])} tasks", (hi["n_tasks"], hi["predicted_rate"]),
                xytext=(9, 2), textcoords="offset points", fontsize=9, color=INK2)
    ax.annotate(f"{100 * lo['predicted_rate']:.0f}% on {int(lo['n_tasks'])} tasks", (lo["n_tasks"], lo["predicted_rate"]),
                xytext=(9, -4), textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlim(0, float(s["n_tasks"].max()) * 1.08)
    ax.set_ylim(0, max(0.05, float(s["predicted_rate"].max()) * 1.18))
    pct(ax)
    ax.set_xlabel("Tasks in the subset")
    ax.set_ylabel("Predicted resolve rate")
    ax.set_title("The same agent, scored on each task subset that a lab used")
    subtitle(ax, f"Rasch-model prediction for a median 31B system on the {p4['n_distinct_subsets']} distinct subsets "
                 f"of at least 10 tasks")
    save(fig, out / "fig6_subset_comparability.png")


def main(argv) -> int:
    results = Path(argv[0]) if argv else ROOT / "results"
    out = Path(argv[1]) if len(argv) > 1 else ROOT / "paper" / "figures"
    res = json.loads((results / "results.json").read_text(encoding="utf-8"))
    tables = results / "tables"
    fig_task_difficulty(res, tables, out)
    if res.get('X1_validity'):
        fig_unpassable(res, tables, out)
    fig_variance(res, tables, out)
    fig_replicates(res, tables, out)
    fig_mdd(res, tables, out)
    fig_subsets(res, tables, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
