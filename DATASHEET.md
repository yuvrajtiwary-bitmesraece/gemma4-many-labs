# Datasheet: the pooled attempts table

`data/derived/attempts.csv` and the tables beside it. Follows the questions of
"Datasheets for Datasets" (Gebru et al.), shortened.

## Why it exists

Participants of Kaggle's Gemma 4 Developer Agent competition ran coding agents
on the same 129 public development tasks, and many published per-task results
in their own repositories, each in its own format. This table puts the
openly licensed ones into one format so that questions about the benchmark,
not about any one agent, can be answered: which tasks can be passed, how
stable task difficulty is across labs, how much two runs of one agent differ.

## What is in it

One row is one **attempt**: one execution of one agent on one task, with a
pass/fail outcome.

| Column | Meaning |
|---|---|
| `lab` | GitHub account that published the result |
| `source` | repository (`owner/repo`); the pinned commit is in `sources.csv` |
| `system` | one agent configuration, as the lab labels it, prefixed by the lab |
| `run` | one execution of a system; runs of one system are repeats |
| `task` | development-set task, `<library>_<number>` |
| `y` | 1 if the source reports the task as resolved, else 0 |
| `model` | Gemma 4 size: E2B, E4B, 12B, 26B-A4B, 31B (empty if not stated) |
| `harness` | `official` if the competition's harness scored it, `local` if the lab's own scorer |
| `err` | 1 if the source attaches a timeout, budget or error flag, 0 if not, empty if it has no such field |
| `status` | the source's own status text, shortened |
| `file` | path of the file in the source that the row comes from |

Size: 15,112 attempts, 14 labs, 259 systems, 346 runs, 129 tasks; 3,829 resolved.
265 runs use the 31B model, 65 E4B, 16 E2B. 309 runs were scored by the
competition harness. 50 systems have repeated runs.

Beside it:

- `runs.csv`, `sources.csv`: one row per run and per source.
- `screening.csv`: all 128 candidate repositories and why each was or was not used.
- `controls.csv`: per-task control runs that eight labs published (the
  reference patch, or no patch, applied in their environment). These are not
  agent attempts.
- `dev_tasks.txt`: the 129 task identifiers.
- `external_zenodo_attempts.csv`: a separate dataset in the same columns
  (Zenodo record 23834104), used only to test estimates, never pooled.
- `attempts_extended.csv`: the same table plus 9,876 attempts from nine sources
  found by a broader search after the results (column `frame` says which rows
  are which). It is used only for the replication section of the report;
  `screening_extended.csv` lists the candidates of that search.
- `results/tables/tasks.csv`: per task, the consensus solve rate, the number of
  labs, its class, its status in the sandbox audits, its Rasch difficulty and
  how often repeats disagree on it. `results/tables/task_lists.json` holds
  ready-made lists (valid tasks, tasks whose reference patch fails everywhere,
  the informative core).

## What is not in it

No competition data: no task statements, patches, tests, repository snapshots
or agent traces. No results of non-Gemma-4 models, no reference-patch or
empty-patch runs (those are in `controls.csv`), no tasks outside the 129, no
runs with fewer than 10 tasks, no hidden-test results.

## How it was collected

1. Six GitHub repository searches on 2026-11-11, fixed in `ANALYSIS_PLAN.md`
   before any result file was opened. Each repository is pinned to the commit
   that was current then (`data/sources/manifest.json`, with checksums).
2. A source is used only if it carries a licence that permits reuse and holds
   per-task outcomes of Gemma 4 agents on development-set tasks.
3. One adapter per source (`scripts/adapters/`) maps its files to the columns
   above and records the evidence for every decision. The common rules are in
   `scripts/adapters/README.md`.
4. Checks: the adapters reproduce 119 of 131 numbers that the sources state
   about themselves (the rest are explained per adapter); the four largest
   sources (83% of the rows) were re-counted from the raw files with separate
   code, with no disagreement in the outcome. Three of them were audited in
   depth; the corrections that followed are Amendment 4 of `DEVIATIONS.md`.

## Things a user must know

- **Not a random sample.** Labs published what they chose to publish. Three
  labs supply 72% of the rows; weight labs equally (as the analyses here do)
  or the largest lab dominates.
- **A failure is not always the agent's failure.** In the stock sandbox the
  reference patch itself fails on 51 of the 129 tasks in all four independent
  audits (and four audits disagree on 9 more). Some labs left such tasks out of
  their runs, others ran them and recorded failures, and four labs changed the
  test environment and resolve some of them. Use `results/tables/tasks.csv`
  (`audit_status`) to separate these cases.
- **Some outcomes are known to be unreliable.** Eight early runs of one lab
  (480 attempts, all failures) were scored in a test environment that could
  not import the packages of 37 of their 60 tasks. They are kept because the
  lab presents them as valid; the card of that adapter and sensitivity
  analysis S3 give the details. Three attempts recorded as resolved with an
  empty patch are coded as failures.
- **Labs differ** in agent, budget, hardware, harness version and sandbox.
  The table records these differences where the source states them; it does
  not remove them.
- **Repeats are what the lab says they are.** Runs share a `system` only where
  the lab identifies them as the same configuration run again.
- **Labels are the labs' own.** `system` names are not comparable across labs.
- Outcomes are those reported by each source; nothing was re-executed.

## Uses it is suited to, and not

Suited: properties of the benchmark (task validity, difficulty, noise,
comparability of subsets), planning how many tasks and repeats a comparison
needs, checking a new evaluation set-up against known task behaviour.
Not suited: ranking teams or agents against each other (labs ran different
tasks in different environments), or any claim about the hidden test set.

## Licence and credit

The rows are facts taken from repositories under Apache-2.0, MIT or MIT-0
(and CC BY 4.0 for two sources of the extended table);
`sources.csv` names each repository, its commit and its licence. The credit
for every run belongs to the lab that produced it; cite the labs' own work
when you use their runs. The compilation and the code are Apache-2.0.

## Rebuilding and extending

`python scripts/fetch_sources.py` downloads every source at its pinned commit
and `python scripts/harmonize.py` rebuilds the table. To add a source, write
an adapter following `scripts/adapters/README.md` and check it with
`python scripts/harmonize.py --check owner/repo`.
