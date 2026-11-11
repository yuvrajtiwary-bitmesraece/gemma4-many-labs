# Amendments and deviations

Every change to `ANALYSIS_PLAN.md` after its first commit is recorded here, in
order, with the time, the reason, and whether outcome data had been seen.

## Amendment 1 — 2026-11-11, about 05:55 UTC — before any outcome data

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
