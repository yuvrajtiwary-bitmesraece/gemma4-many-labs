# Amendments and deviations

Every change to `ANALYSIS_PLAN.md` after its first commit is recorded here, in
order, with the time, the reason, and whether outcome data had been seen.

## Amendment 1 — 2026-11-11, 05:33 UTC — before any outcome data

Made while the source snapshots were still downloading. No per-task outcome
file had been opened and the schema-discovery script had not been run.

**1a. Replicate noise is estimated without bias (changes P2).**
The plan took replicate noise to be the remainder of the saturated
system × task model. In-sample, a cell mean of *n* replicates absorbs part of
the noise (half of it when n = 2), so that remainder is too small. Replicate
noise is instead the unbiased within-cell variance, `n/(n−1) · p(1−p)` per cell
with n ≥ 2, averaged with the same weights. R² values for the other models are
still reported as planned and are described as in-sample.

**1b. A within-lab generalizability study is added (new P2b).**
Reason: R² shares do not say how reliable a score is. Generalizability theory
answers that directly and is the standard tool for it.

For each lab that has replicates, take its systems with at least 2 replicate
runs and the tasks that every one of those systems attempted in at least 2
replicates. If fewer than 20 tasks remain, drop the system with the lowest
task coverage and repeat, stopping when at least 20 tasks remain or fewer than
2 systems are left (the lab is then skipped). Use the first *r* replicates of
each cell in run order, where *r* is the smallest replicate count in the block.

On this balanced systems × tasks × replicates block, estimate the variance
components of a two-way random-effects model with replication from the usual
expected mean squares: task `σ²_t`, system `σ²_s`, system × task `σ²_st`, and
replicate noise `σ²_e`. Negative estimates are set to zero and the raw values
are kept in the output.

Report per lab and summarise across labs (median and range):

- each component's share of the total;
- the generalizability coefficient for comparing that lab's systems on `n`
  tasks with `k` runs each,
  `Eρ² = σ²_s / (σ²_s + σ²_st/n + σ²_e/(n·k))`,
  for n = 30, 60, 129 and k = 1, 3, 5;
- the standard error of the difference between two systems run once on the
  same `n` tasks, `sqrt(2(σ²_st + σ²_e)/n)`.

Consistency check, fixed in advance: the pooled flip rate from P3 should be
close to twice the replicate-noise variance, because two independent runs of a
cell with solve probability *p* disagree with probability `2p(1−p)`.

**1c. Estimators are tested on simulated data first.**
`scripts/test_stats.py` checks every estimator against simulated data with
known parameters before it is applied to the pooled data.

## Amendment 2 — 2026-11-11, 06:15 to 08:00 UTC — after outcome data were opened

Everything below was decided while or after reading the sources. The primary
analyses P1 to P4 and the robustness checks of section 8 were run as written
in the plan and Amendment 1; none of their definitions changed.

**2a. How many sources had usable data.**
The plan expected about 60 repositories with result files. 128 repositories
were returned by the searches, 121 could be fetched, 58 contain
development-set task identifiers, 35 of those carry a licence, and **14 have
at least one eligible run**. The other 21 licensed sources hold retrieval or
localisation metrics, task metadata, totals without per-task rows, other
benchmarks, or simulated results. `data/derived/screening.csv` lists every
repository with the stage at which it left the study and the reason.

**2b. Screening went beyond JSON and CSV files.**
`scripts/triage.py` also searched markdown, notebooks, logs and text files
for task identifiers. Eight more repositories had at least 10 identifiers in
such a file; two of them are licensed and were examined (one is used).
Results that labs keep outside GitHub (Kaggle datasets, Zenodo) were not
collected.

**2c. Licence rule in practice.**
A source counts as licensed if GitHub detects a licence, or if a licence file
or a README statement grants reuse. Three repositories with a licence file
that GitHub does not classify were accepted after reading it. One repository
whose README licenses only its writeup was not used. Repositories with no
licence statement were not opened further.

**2d. Task universe.**
The plan took the union of identifiers across sources. In the event, six
labs list the whole development set in one file or run, they agree exactly on
129 identifiers with the documented composition, and no eligible attempt
carries an identifier outside that set. The development set is these 129
identifiers (`data/derived/dev_tasks.txt`; the harmonizer log names the labs).

**2e. Duplicates.**
Implemented as: byte-identical files within a source are counted once
(sha256), and within a run the first record per task is kept.

**2f. Adapters.**
The 36 adapters were written with AI assistance, in parallel batches, after
the common rules were fixed in `scripts/adapters/README.md`. Each adapter
records its evidence, its judgement calls and the numbers the source reports
that it reproduces. Judgement calls that matter for the results:

- *Repeats.* Runs share a system only where the lab identifies them as
  repeats. Two groups in one lab (L01) rest on pooled reporting and an
  identical agent commit, not on a sentence that says "repeat". In one lab
  (L11) four runs are rebuilt from the per-task flip lists the lab publishes
  against a fully published run. The same configuration on different
  hardware counts as different systems (L02).
- *Attempts a lab marks as unscorable.* Section 4 of the plan drops attempts
  that a source marks as invalid for infrastructure reasons. Labs that
  checked their sandbox mark tasks whose reference patch fails there, and
  those attempts are dropped (L07 23 attempts, L10 54, L14 9; L01 61 rows
  with server errors). Labs that did not check kept such tasks in their runs
  as failures. This follows the plan, and it means the pooled failures are
  not all of one kind. X1 below measures it.
- *One lab's scorer.* L04 uses its own scorer, not the competition harness,
  and eight of its early runs resolve 0 or 1 of 60 tasks. The lab does not
  call them invalid, so they are kept. The "official harness only" check and
  the leave-one-lab-out check both remove this lab.

**2g. Small additions to planned analyses.**
P2: the saturated model is reported on the replicated cells together with the
additive models on the same cells. P4: a model check (predicted against
observed resolve rate per run) is reported. E2: the lab-clustered bootstrap
resamples labs without refitting the Rasch model. E4: replicate disagreement
per task is twice the unbiased within-cell variance, averaged in the levels of
section 6. The figure for P1 shows the middle half of the labs' solve rates.

**2h. E1 is a description, not an analysis.**
Leaderboard scores appear only in the prose of some labs' documents. The
pairs were compiled by hand into `data/sources/leaderboard_pairs.csv`, with
the file and wording for each, and are summarised descriptively. Five of the
seven labs in that table contribute agent attempts; the other two contribute
only these statements.

**2i. New exploratory analysis X1: task validity.**
Not in the plan. Added after the sources showed that eight labs had published
per-task control runs: the reference patch, or no patch, applied to the tasks
in their environment. `scripts/controls.py` collects them and
`scripts/validity.py` compares them with each other and with the agents'
outcomes. Three of the eight labs contribute no agent attempts. X1 is
reported as exploratory, and the status of a task in it comes from one
reference audit, chosen because it is the only one that ran both controls on
all 129 tasks with the competition harness.

**2j. Order of work.**
The analysis code was committed before the pooled table existed and was first
run on synthetic data. It was then run on partial pooled tables (10 and 12
labs) while the last adapters were being written, to find bugs. Between those
runs and the final run the only changes to analysis code were X1, the E1
summary, and figure layout.

## Amendment 3 — 2026-11-11, 08:00 to 09:00 UTC — after the results were known

Additions made after the first complete results. None changes a planned
analysis; all are reported in `results/RESULTS.md` under their own headings.

**3a. Sensitivity and uncertainty** (`scripts/sensitivity.py`). One decision
or definition is changed at a time: repeats only where a lab says "repeat",
runs rebuilt from flip lists removed, one lab's early runs removed, pairs on
identical task sets only, other floor thresholds, task validity from the
agreement of four audits, sampling temperature within one lab. A
cluster-bootstrap interval for the flip rate, and a table of what repeated
runs buy (a normal-approximation extension of the planned McNemar
calculation), are added.

**3b. External validation** (`scripts/external_zenodo.py`,
`scripts/external_validation.py`). A dataset outside the source frame of
section 3 (Zenodo record 23834104, Apache-2.0, from a participant whose
repository the searches did not return) is used to test the pooled estimates.
It is not pooled and no estimate uses it.

**3c. Second reading of the extraction.** The three largest sources (72% of
the attempts) were re-counted from the raw files by a second reader with
separately written code: every pooled row equals its source row (10,859 rows,
no disagreement). Every row of `data/sources/leaderboard_pairs.csv` was
re-checked against the cited file; three rows were corrected and seven added,
and the E1 summary changed accordingly (git history shows the before and
after).
