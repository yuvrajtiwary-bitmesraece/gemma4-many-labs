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

## Amendment 4 — 2026-11-11, 09:30 to 21:00 UTC — after an independent review and three audits

After the results of Amendment 3, four independent checks were run by readers
who had not written the code they checked (AI-assisted, like the rest of the
pipeline; their working notes are not in this repository because they quote the
sources at length):

- a **statistical review** that recomputed every reported number with
  separately written code, from the definitions in the plan and in this file.
  Every number was reproduced. The review then listed where a number was right
  but the claim, the weighting or the derivation built on it was not;
- three **source audits** (L01, L02 and L04: three of the four largest
  sources, 68% of the attempts). Each recounted the source from its raw files
  before reading the adapter, compared every pooled row with its source row
  (10,214 rows, no disagreement in the outcome), and graded the evidence
  behind every group of repeats, every label and every exclusion.

Everything below follows from those four documents. None of the definitions of
P1 to P4 changed. Where a number of the first results was wrong or misleading,
both values are in the table at the end.

**4a. Corrections to three adapters (they change the pooled table).**

- *L01, one merge undone.* Two experiments had been treated as repeats of one
  system on the strength of two script docstrings that list them under one
  version label. The audit found no statement that they are repeats, and found
  that the two sessions ran at different speeds. The plan counts unclear cases
  as different systems, so they are now two systems. A second group in this
  lab (three files, six runs) is kept: its only evidence is a pooled
  percentage in the lab's outline that equals the three files added together.
  The sensitivity analysis S2 splits it.
- *L04, three outcomes recoded.* Three attempts were recorded as resolved with
  an empty patch (one task, three early runs). The lab itself calls this an
  "environmental false positive" and its later scorer refuses it. They are
  now failures.
- *L04, one merge added.* Two runs of one community adapter on two task suites
  were kept apart because the two scripts load it from two directory names.
  The lab's paper lists them as one configuration, exactly as it does for
  another adapter that was already merged. They are now one system.
- *L04, eight early runs: kept, and documented.* The audit showed from the
  result files that these runs (480 attempts, none resolved after the
  recoding) were scored in a test environment that could not import the
  packages needed by 37 of their 60 tasks, with a tool layer that lost most
  edits. One fix that the same scorer accepts 18 times out of 18 in later runs
  is scored as failed five times. The lab presents the runs as valid results,
  and section 4 of the plan drops attempts only where the source marks them
  invalid, so they stay in the pool. S3 removes them, and nothing moves.
- *L02, one merge added.* One third-party configuration was run once in the
  lab's Kaggle stack and four times on its local rig, with the same server
  software and weights. The first version split by rig. The lab pools exactly
  these runs as repeats in its own analysis, so they are now one system.
  Another configuration, which the lab contrasts between the two rigs, stays
  split.
- The wording of the three cards was corrected in several places (what a
  leaderboard file is, which rows a lab counts as failures, which statements
  are inferences). A flag for a modified test environment was added to L01.

**4b. Errors in analysis code and derivations.**

- *Weights on the replicated cells (P2, robustness table).* "Lab-balanced"
  weights were computed on the whole table and then applied to the cells with
  repeats, where they do not give the labs equal weight (one lab had 28% of
  the weight, another 2.5%). They are now computed on those cells.
- *Model check of P4.* The reported check (predicted against observed resolve
  rate per run, in sample) was true by construction: the score equations of
  the model force predicted and observed totals per system to agree. It is
  replaced by two out-of-sample checks: five-fold cross-validation over
  system × task cells, and a test of the subset effect in which task
  difficulties are fitted without the system's lab and its ability is fitted
  on its tasks outside the subset.
- *"What repeats buy" (S9).* The table used a variance of
  (flip rate + gain) / (k n) for the mean difference of two agents run k times
  on n fixed tasks. That is the null variance of a one-run McNemar test, not
  the variance for k runs. With tasks fixed the sampling variance is
  flip rate / (k n), and the null variance of the test that permutes run
  labels within a task is (flip rate / k + gain / (2k − 1)) / n. The table is
  recomputed with the exact stratified permutation test, by simulation, and a
  second table is added for a different question: whether a gain would hold
  on new tasks, where more runs do not remove the task-sampling part.
- *Probability of a wrong order (S11).* The same mistake: the variance of the
  difference of two single runs on fixed tasks is flip rate / n, not
  (flip rate + gain) / n. The corrected probabilities are lower for large
  gains.
- *Lab-balanced resolve rate.* Runs are now averaged within a system before
  systems are averaged within a lab, as section 6 of the plan says. The code
  had pooled a system's attempts.
- *Number of random splits.* The robustness rows used 500 and the
  leave-one-lab-out rows 300 splits, against 2,000 for the headline. All use
  2,000 now.

**4c. Task validity (X1) no longer rests on one audit.**
X1 took a task's status from the one audit that ran both controls on all 129
tasks. The review showed that this audit is the one that disagrees most with
the other three full audits of the same sandbox: the three agree with each
other on 127 to 129 tasks and with it on 121. A task's status now comes from
the agreement of the four: valid if the reference patch resolves in all four
and the task does not resolve with no patch; "reference patch fails" if it
fails in all four; "audits disagree" otherwise. The old definition is kept as
sensitivity analysis S8.

The review also showed that the sentence "28% of pooled attempts were made on
tasks where the reference patch fails" added two different things: labs whose
environment is the stock sandbox, which never resolve such a task, and labs
that changed the test environment, which resolve a fifth of them. The report
now gives the two groups separately and no longer reads the pooled share as
lost effort.

**4d. Additions that show the limits of the planned numbers.**
For P3: the weight of single labs and systems in the pooled flip rate, five
other ways of averaging the same pairs, the share of flips whose failing
attempt carries an error flag, and the planned detectable difference under two
other assumptions about how an improvement spreads over tasks. For the
uncertainty of the flip rate (S1): a t interval on the lab-level rates and a
simulation of the coverage of all three intervals. The bootstrap interval of
the first results was too narrow. For P4: which subsets a lab chose from its
own earlier outcomes (both extreme predictions are such subsets) and the range
without them. For the robustness table: a second error-flag row that re-applies
the 10-task minimum, and a note on why those rows cannot be compared with the
others. For E2: a t interval on the lab means beside the bootstrap interval.

**4e. Extended pool: a partial replication outside the registered frame**
(`scripts/search_extended.py`, `scripts/extended_pool.py`). A broader search,
made after the results, returned 127 further repositories. 33 carry a licence
and development-set task identifiers. 16 of those were examined before the
deadline (the order was set from the structure of their files, before any of
their outcomes was read), 16 were not, and one is the repository of the
external validation dataset. Nine have eligible runs. They are processed by
the same adapters and rules, kept out of every registered number, and reported
in their own section beside the registered frame. The largest of the nine
(2,950 rows, 30% of the added attempts) was recounted from its raw files with
separately written code, with no disagreement; the other eight adapters were
checked only against the numbers their sources report (80 of 84 reproduced).

**4f. Not changed.** The eight early runs of L04 and the three-file group of
L01 stay in the pool (S3 and S2 remove them). Attempts that ended in a
rig-specific cut-off in L02 stay failures, as the lab counts them. The planned
detectable difference keeps its planned alternative; the other assumptions are
reported beside it.

**4g. First results and corrected results.**

| Quantity | First results | After Amendment 4 | Cause |
|---|---|---|---|
| Attempts / systems / runs | 15,112 / 260 / 346 | 15,112 / 259 / 346 | 4a |
| Resolved attempts | 3,832 | 3,829 | 4a (three recoded) |
| Lab-balanced resolve rate | 29.9% | 30.2% | 4b |
| Split-half reliability (Spearman–Brown) | 0.80 (0.89) | 0.80 (0.89) | |
| Floor / discriminating / ceiling tasks | 40 / 58 / 3 | 41 / 57 / 3 | 4a (one task crosses the 10% line) |
| R² task / lab / system (lab-balanced) | 44.8% / 14.6% / 18.5% | 44.9% / 14.6% / 18.5% | 4a |
| Run-to-run noise share (lab-balanced) | 22.6% | 22.8% | 4a, 4b (22.5% with the old weights on the new table) |
| Cells with repeats | 1,861 | 1,760 | 4a |
| P2b, median Eρ² (129 tasks, 1 run) | 0.56 | 0.29 | 4a (one weak system left the block of L01) |
| P2b, median SE of a difference (129 tasks) | 3.2 points | 3.1 points | 4a |
| Pairs of repeats / labs | 123 / 9 | 127 / 9 | 4a |
| Pooled flip rate | 11.2% | 10.9% | 4a |
| Flip rate, median of labs | 10.4% | 10.4% | |
| 95% interval of the flip rate | 8.6% to 12.3% (bootstrap over labs) | 7.0% to 13.9% (t interval on lab-level rates) | 4d |
| Detectable difference, 30 / 60 / 129 tasks | 34.8 / 20.8 / 12.5 points | 34.5 / 20.8 / 12.5 points | 4a |
| Subset predictions, all 52 subsets | 5.7% to 66.7% | 5.6% to 66.7% | 4a |
| ... without subsets chosen from outcomes | not reported | 17.7% to 64.4% (25.5% to 39.3% for 30 tasks or more) | 4d |
| P4 model check | Pearson 0.99, error 1.4 points (in sample) | Pearson 0.986, error 1.9 points (held-out cells) | 4b |
| X1: valid / reference patch fails / other | 68 / 56 / 5 | 67 / 51 / 11 | 4c |
| Attempts on tasks whose reference patch fails | 27.7% | 24.6% (8.2% in ten labs, none resolved; 32.0% in four labs, 21.6% resolved) | 4c |
| Floor tasks that are valid | 24 of 40 | 23 of 41 | 4a, 4c |
| Runs per agent to detect 3 / 5 / 10 points on 129 tasks | 10 / 4 / 2 | 9 / 4 / 2 | 4b |
| Detectable with 3 / 5 / 10 runs per agent, 129 tasks | 5.9 / 4.3 / 2.9 points | 5.3 / 4.1 / 2.8 points | 4b |
| Wrong order on 58 tasks, true gain 5 / 10 points | 17.2% / 4.9% | 12.4% / 1.1% | 4b |
| Noise band: scale; coverage leave-one-lab-out | 0.78; 95.9% | 0.78; 96.1% | 4a |
| External validation: Spearman of task difficulty | 0.84 | 0.84 | |
