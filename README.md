# Many labs, one benchmark

A pooled analysis of the per-task results that participants of Kaggle's
[Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent)
competition published on GitHub. Each team sees only its own runs on the 129
public development tasks. Pooling the public record answers questions no single
team can: is task difficulty a stable property across independent labs, how
much of an outcome is run-to-run noise, what difference between two agents can
a comparison detect, how far apart are scores that teams report on different
task subsets, and which tasks can be passed at all in the environment teams
actually used.

Entry for the [Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper).

- **Results:** [`results/RESULTS.md`](results/RESULTS.md) (generated from `results/results.json`).
- **Dataset:** [`data/derived/attempts.csv`](data/derived/attempts.csv), one row per agent run on one task.
- **Plan fixed before the data:** [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md), with every later change in [`DEVIATIONS.md`](DEVIATIONS.md).

## What is here

| Path | What it is |
|---|---|
| `ANALYSIS_PLAN.md` | Questions, inclusion rules, units and analyses, committed before any per-task outcome file was opened (the commit history shows the order). |
| `DEVIATIONS.md` | Every amendment to the plan, with the time, the reason and whether outcome data had been seen. |
| `data/sources/search_results.json`, `manifest.json` | The candidate repositories returned by the searches and the commit each one is pinned to, with the tarball checksum. |
| `scripts/adapters/` | One adapter per examined source: where its per-task outcomes are, how systems, runs and repeats were identified, the evidence for each decision, and the numbers the source itself reports that the adapter reproduces. `README.md` there states the rules. |
| `data/derived/attempts.csv` | The pooled table: `lab, source, system, run, task, y, model, harness, err, status, file`. 15,112 attempts, 14 labs, 260 agent configurations, 346 runs, 129 tasks. |
| `data/derived/runs.csv`, `sources.csv`, `screening.csv` | One row per run, per source with an adapter, and per candidate repository (with the stage at which it left the study). |
| `data/derived/controls.csv` | Control runs that eight labs published per task: the reference patch, or no patch, applied in their environment. Not agent attempts; used only in the validity analysis. |
| `data/sources/leaderboard_pairs.csv` | Development-set rates and public-leaderboard scores that labs state for the same system, compiled by hand with the file and wording for each. |
| `scripts/mlstats.py`, `test_stats.py` | The estimators and their checks against simulated data with known parameters. |
| `scripts/analyze.py`, `validity.py`, `report.py`, `figures.py` | Analyses, the generated results report and the figures. |
| `results/` | `results.json`, `RESULTS.md` and the tables behind every figure. |
| `paper/figures/` | Six figures. |

## Reproducing

Python 3.12, `pip install -r requirements.txt` (numpy, scipy, pandas, matplotlib).

From the committed pooled table (no network, about two minutes):

    python scripts/test_stats.py      # estimator checks on simulated data
    python scripts/analyze.py         # results/results.json and results/tables/
    python scripts/report.py          # results/RESULTS.md
    python scripts/figures.py         # paper/figures/

From the sources (needs the GitHub CLI `gh`, logged in; downloads about 130 MB):

    python scripts/fetch_sources.py   # download every source at its pinned commit
    python scripts/discover.py        # structure report per source
    python scripts/triage.py          # licence evidence, task ids outside JSON/CSV
    python scripts/harmonize.py       # run the adapters, write data/derived/
    python scripts/controls.py        # data/derived/controls.csv
    python scripts/screening.py       # data/derived/screening.csv

Third-party snapshots go to `data/raw/`, which is not committed. Sources are
read in place from their tarballs; nothing from a source is executed.

## Data and attribution

The pooled table contains task identifiers, binary outcomes and run labels
taken from the repositories listed in `data/derived/sources.csv`, each at the
commit given there and under the licence given there (Apache-2.0, MIT, MIT-0
or CC BY 4.0). The credit for every run belongs to the lab that produced it.
Repositories without an explicit licence were not used.

The table holds no competition data: no task statements, patches, tests,
repository snapshots or agent traces.

This study is about the benchmark, not about teams. It does not rank or grade
named participants, and figures use lab codes.

## Limitations

The pooled runs are what teams chose to publish, not a random sample of what
they ran. Labs differ in agent, budget, hardware and scorer, and the study
observes these differences rather than controlling them. See the limitations
section of the paper and `DEVIATIONS.md`.

## AI assistance

The analysis plan, the code in this repository and the generated results
report were drafted with Claude (Anthropic) under the author's direction, and
the per-source adapters were written by Claude and checked against numbers the
sources themselves report. The text of the paper is the author's own.

## Licence

Code: Apache-2.0, see `LICENSE`. The pooled table is a compilation of facts
from the sources above and is offered under the same terms, subject to the
attribution requirements of each source.
