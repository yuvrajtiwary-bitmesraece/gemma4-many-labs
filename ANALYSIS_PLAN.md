# Analysis plan

**Status:** written and committed before any per-task outcome file was opened.
Changes after this commit are listed in `DEVIATIONS.md` with the reason.

Written 2026-11-11, about 05:30 UTC. Drafted by Claude (Anthropic) working with
the author; see the AI-assistance note in `README.md`.

## 1. Question

Hundreds of teams ran Gemma 4 coding agents on the same public development set
of the Kaggle *Gemma 4 Developer Agent* competition (129 Python tasks from
FastAPI, Requests, Rich and HTTPX), and many published their per-task results.
Each team sees only its own runs. Pooling the public record lets us ask what no
single team can:

1. **Consensus.** Is the difficulty of a task a stable property across
   independent labs, and how many tasks actually separate systems?
2. **Variance.** How much of the variation in outcomes comes from the task, the
   lab, the system within a lab, and run-to-run noise?
3. **Noise floor.** How often do two runs of the same system disagree, and what
   difference between two systems can a single paired comparison detect?
4. **Comparability.** Teams report scores on different task subsets. How far
   apart are those numbers for one and the same agent?

## 2. What was known when this was written

- Repository names, descriptions and file listings of the candidate
  repositories (section 3).
- Automated summaries of two READMEs (`yosh95/g4-developer-agent-paper`,
  `CVasilopoulos/gemma4-laptop-agent`), including headline numbers reported
  there, and the titles of several other writeups. Some titles state
  conclusions (for example that most agent improvements are noise).
- **No per-task outcome file had been opened**, and no pooled statistic of any
  kind had been computed.

## 3. Sources

Candidate sources are public GitHub repositories returned on 2026-11-11 by
`gh search repos <query> --limit 100` for these six queries, created on or
after 2026-08-01:

    gemma 4 developer agent
    gemma-4-developer-agent
    gemma4 developer agent paper
    gemma 4 developer agent paper track
    gemma4 agent paper track kaggle
    gemma 4 coding agent kaggle paper

Each source is pinned to the commit that was `HEAD` when it was fetched. The
snapshot is not updated afterwards, even if the repository changes.

**Source-level inclusion.** A source enters the pooled dataset only if

- it carries an explicit licence that permits reuse (for example Apache-2.0,
  MIT, MIT-0, BSD, CC-BY, CC0), stated in a licence file or the README, and
- it contains at least one eligible record (below).

Repositories without such a licence are listed but not used.

## 4. Records

**Eligible record (an *attempt*):** one execution of an agent built on a
Gemma 4 model on one task of the public development set, with a binary
verified outcome (resolved or not), as scored by the competition harness or by
a local re-implementation that the source documents.

**Excluded:**

- tasks that are not in the public development set (other benchmarks, synthetic
  or mutated tasks, private tasks);
- runs of non-Gemma-4 models (for example teacher models), and runs that apply
  the reference patch instead of an agent's patch;
- test fixtures, demos and examples;
- metrics that are not a per-task resolved/not-resolved outcome (localisation
  hits, retrieval ranks, partial test counts);
- hidden-test (leaderboard) results, which are per-submission, not per-task.
  They are used only in the exploratory analysis E1.

**Outcome coding.** `1` if the source reports the task as resolved, `0`
otherwise, including timeouts, exhausted budgets and agent errors, because the
competition scores those as failures. Attempts that a source marks as not run,
skipped or invalid for infrastructure reasons are dropped when the source makes
that distinction. A sensitivity analysis drops every attempt carrying any error
flag.

**Task identity.** Tasks are keyed as `<repo>_<number>` with repo in
{fastapi, requests, rich, httpx}. We do not have the competition data, so the
task universe is the union of such identifiers across sources. Its size is
reported against the documented 129.

## 5. Units

- **Lab:** the GitHub account that owns the source. Repositories of one
  account are merged.
- **System:** one agent configuration within a lab, as the lab labels it.
- **Run:** the attempts that one system produced in one execution.
- **Replicates:** runs that the lab itself identifies as repeats of one system
  (a seed or repeat index in the run name, or a statement in the
  documentation). When this is unclear, runs count as different systems.
- A run needs at least 10 eligible tasks to be used.

**Duplicates.** Within a lab, files with identical content, and runs with
identical run identifiers and identical task-outcome maps, are counted once.

**Metadata** recorded per run where the source states it: model size (E2B,
E4B, 12B, 26B-A4B, 31B), whether the official harness scored it, and the
source's own label. Missing values stay missing; nothing is guessed.

## 6. Primary analyses

Unless stated otherwise, averages are taken in levels so that no lab dominates
by volume: replicates are averaged within a system, systems within a lab, and
labs are then averaged with equal weight.

**P1. Consensus on task difficulty.**
For task *t* and lab *l*, `p[l,t]` is the lab-level solve rate. Consensus
difficulty `d[t]` is the mean of `p[l,t]` over labs, reported for tasks
attempted by at least 5 labs.

- *Reliability:* split the labs at random into two halves 2,000 times, compute
  `d[t]` in each half, and take the Spearman correlation across tasks covered
  by at least 2 labs in both halves. Report the median, the 2.5–97.5
  percentile range, and the Spearman–Brown corrected value.
- *Pairwise agreement:* Spearman correlation of `p[l,·]` between every pair of
  labs sharing at least 30 tasks.
- *Floor and ceiling:* number of tasks with `d[t] <= 0.10` and with
  `d[t] >= 0.90`. The rest are called discriminating.

**P2. Variance decomposition.**
Linear probability models on attempts, with weights that give every lab the
same total weight. Report R² for: task fixed effects; lab fixed effects;
system fixed effects; task + system; and, on cells with replicates, the
saturated system × task model. The remainder of the last model is replicate
noise. Both orders (task first, system first) are reported for the additive
part. Unweighted results are reported as a robustness check.

**P3. Replicate noise and detectable differences.**
For every pair of replicate runs sharing at least 10 tasks: the *flip rate*
(fraction of shared tasks with different outcomes) and the difference in
resolve rate. Pairs are averaged within a system, systems within a lab, and
the median and interquartile range across labs are reported, together with the
pooled value.

From the pooled flip rate `π`, the minimum detectable difference `δ` for one
paired comparison on *n* tasks: discordant-pair probabilities `π/2 + δ` and
`π/2`, exact two-sided McNemar test at α = 0.05, power 0.80 by exact
enumeration, for n = 30, 60 and 129.

**P4. Subset comparability.**
Fit a Rasch (one-parameter logistic) model, `logit P = θ[system] − b[task]`,
by penalised joint maximum likelihood. For every distinct task subset that any
run used (at least 10 tasks), predict the resolve rate of a reference system
(median `θ` among 31B systems) on that subset. Report the range and
interquartile range of these predictions.

## 7. Exploratory analyses (labelled as such in any writeup)

- **E1. Development set versus leaderboard.** Where a lab reports both a local
  development-set score and a public-leaderboard score for the same system:
  rank correlation and typical ratio.
- **E2. Model size.** Mean `θ` by model size with a lab-clustered bootstrap,
  and within-lab contrasts where a lab ran more than one size.
- **E3. Repository.** Consensus solve rate by source library.
- **E4. Instability by difficulty.** Per-task replicate disagreement against
  `d[t]`.

## 8. Robustness checks fixed in advance

- 31B systems only.
- Official-harness runs only.
- Unweighted (attempt-level) instead of lab-balanced.
- Dropping attempts with any error flag.
- Leave-one-lab-out for P1 reliability and the P3 pooled flip rate.

## 9. What this study does not do

- It does not rank or grade named teams. Results are about the benchmark.
  Sources are credited in the source table; figures use lab codes.
- It does not redistribute competition data (task statements, patches, tests,
  repository snapshots, traces). The released table holds only identifiers,
  outcomes and run metadata from licensed sources.
- It does not claim that pooled runs are a random sample of anything. They are
  what teams chose to publish, and selection is discussed as a limitation.
