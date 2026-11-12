# Many Labs, One Benchmark: A Pre-Registered Pooled Analysis of 15,112 Gemma 4 Agent Attempts

*Task difficulty is stable across independent labs, one task in ten flips between identical runs, and the labs were not all running the same benchmark.*

## Abstract

Teams in the Gemma 4 Developer Agent competition test their agents on the same 129 public development tasks, and each sees only its own runs. We pooled the per-task results that 14 teams ("labs") published under open licences: 15,112 attempts from 346 runs of 259 agent configurations, analysed under a plan committed before any result file was opened. (1) Task difficulty is a property of the task: rankings from two random halves of the labs correlate at 0.80, and 41 of 101 well-covered tasks are solved at most 10% of the time. (2) When one configuration is run twice, 10.9% of task outcomes flip. (3) One paired comparison on 129 tasks therefore detects only gains of about 8 to 12.5 points; four runs per agent detect 5 points, but claims about new tasks keep a floor near 3.4 points. (4) Labs used 52 different task subsets, on which the same agent would score between 5.6% and 66.7%. (5) In four independent audits of the stock sandbox, 51 tasks fail with their own reference patch, while labs with modified environments resolve them: the labs measured different benchmarks. An external dataset and nine further labs reproduce the main quantities. We release the pooled table, task lists and a tool that tests a difference between two agents against this noise.

## 1. Introduction

Every team in this competition asks the same question with the same instrument: did my change make the agent better on the development set? Each team answers alone, from a few runs, often on a subset of the tasks, in a test environment it built itself. Is a 4-point gain real? Is a task hard or broken? Can a score on 30 tasks be compared with a score on 129? One team's runs cannot settle these questions.

Everyone's runs can. Many teams published per-task outcomes in public repositories. The "Many Labs" projects in psychology [1] pooled independent replications to separate a finding from the laboratory that produced it. We do the same for a benchmark: each team is a lab, and the development set is the shared instrument.

Our contributions:

- A pooled table of 15,112 attempts by 14 labs in one format, with documented adapters checked against the numbers each source reports.
- A pre-registered analysis of task difficulty, run-to-run noise, detectable differences and subset comparability, with an exploratory analysis of task validity and two replications on unused data.
- Planning numbers: how many repeated runs a comparison needs, what repeats cannot fix, and a standard-library tool that applies both.

## 2. Related work

Liu et al. [2] pooled the published per-task verdicts of 254 SWE-bench [3] submissions and showed that the leaderboard cannot order its top entries, the closest design to ours. Bjarnason et al. [4] measured the run-to-run randomness of agents on SWE-bench Verified within one group. Ge et al. [5] fitted item response models across agentic coding benchmarks. Miller [6] and Card et al. [7] argue for error bars, paired comparisons and power analysis. SWE-bench Verified [8] showed that many tasks of the original benchmark could not be solved as specified. Several teams in this competition saw parts of our findings in their own data, such as noise between repeats and failing reference patches. New here is the combination: repeated runs from many independent labs, validity audits by four of them, one scale across 52 task subsets, and a plan fixed in advance.

## 3. Data and method

**Sources.** Six GitHub searches returned 128 repositories. We fetched 121 at a pinned commit; 58 contained development-set task identifiers, 35 of those carried an explicit licence, and 14 had at least one eligible run (ten or more tasks with a per-task verdict). One adapter per source maps its files to a common row: lab, configuration, run, task, resolved or not, model, scorer, error flag. The adapters reproduce 119 of the 131 numbers that the sources state about themselves; each difference is explained in the adapter's card.

**Units.** An *attempt* is one agent execution on one task, a *run* one pass over a task set, a *configuration* one agent set-up as its lab labels it, and a *lab* one GitHub account. Of the 15,112 attempts, 25.3% are resolved. Most runs use the 31B model (265), the others E4B (65) and E2B (16). Three labs supply 72% of the attempts, so every average is lab-balanced: repeats are averaged within a configuration, configurations within a lab, and labs count equally.

**Plan.** Questions, inclusion rules, units and estimators were committed to git before any outcome file was opened, and the estimators were first tested on simulated data. Four analyses were planned: consensus task difficulty with split-half reliability; a variance decomposition with an unbiased estimate of run-to-run noise; the flip rate between repeated runs, with the smallest improvement that an exact McNemar test [9] detects at 80% power and the 5% level; and a one-parameter Rasch model [10] that places runs on different subsets on one scale. The validity analysis, the sensitivity analyses and both replications came after data had been seen and are labelled exploratory. Every change is logged in `DEVIATIONS.md`.

**Review.** An independent review then recomputed every reported number with separately written code, and three audits compared 10,214 pooled rows with the raw files. The audits found no transcription error in an outcome. The review found two wrong derivations, one check that was true by construction, one weighting error, one interval that was too narrow and one misleading claim. All were corrected before the numbers below were produced, and the before-and-after table is published.

## 4. Results

### 4.1 Task difficulty is a property of the task (planned)

We split the 14 labs into two random halves 2,000 times and ranked the tasks by solve rate in each half. The median Spearman correlation is 0.80. Labs with different agents, budgets and hardware agree on which tasks are hard. Which task is attempted explains 44.9% of the outcome variance; which lab attempts it explains 14.6%.

Of the 101 tasks attempted by at least five labs, 41 sit at the floor (solved at most 10% of the time; 25 were never solved by any lab), 3 at the ceiling (at least 90%) and 57 in between (Figure 1).

![Figure 1](figures/fig1_task_difficulty.png)

Figure 1. Consensus solve rate of the 101 tasks attempted by at least five labs, with the middle half of the labs' rates. Colour gives the task's status in four sandbox audits (Section 4.5).

### 4.2 One task in ten flips between identical runs (planned)

Forty-nine configurations in nine labs were run more than once, giving 127 pairs of repeated runs. Between two runs of the same configuration, 10.9% of the shared tasks change outcome. Two labs carry 86% of the weight of that pooled figure, so the honest interval counts each lab once: mean 10.4%, 95% interval 7.0% to 13.9%. The score hides the churn. Two repeats differ by 3.3 points on average, yet only 14% of the pairs agree on every task. Run-to-run noise is 22.8% of the outcome variance in the cells with repeats.

In about half of the flips (51%) the failing attempt carries a timeout, budget or tool-error flag; where neither attempt is flagged, the flip rate is 7.9%. Much of the noise is an agent that sometimes finishes in time and sometimes does not.

### 4.3 What one comparison can detect, and what repeats buy (planned, extended after the review)

Suppose a team runs the old and the new agent once on the same tasks and applies an exact McNemar test. With 80% power it detects an improvement of 12.5 points on 129 tasks, 20.8 on 60 and 34.5 on 30. That is the planned worst case, in which gained tasks go from never solved to always solved. If the improvement adds no disagreement of its own, the figure on 129 tasks is 8.2 points (Figure 2). The median run in the pool covers 30 tasks.

![Figure 2](figures/fig5_detectable_difference.png)

Figure 2. Smallest improvement that one paired comparison detects, by number of tasks. The band runs from an improvement that adds no disagreement to the planned worst case.

Repeated runs help, but they answer only one of two questions (Table 1). *Is B better than A on these tasks?* Here noise is the only obstacle, and a 5-point gain needs about 4 runs per agent on 129 tasks, 7 on 60 and 14 on 30. *Would B be better on new tasks, such as the hidden test set?* Then the tasks are a sample, and the part of a difference that depends on which tasks were drawn does not shrink with more runs. Variance components [11] from the four labs with a balanced block put that floor at 3.4 points on 129 tasks and 5.1 on 58, with a wide range across labs.

Table 1. Smallest improvement, in points, detected with 80% power at the 5% level. Column 2 is the exact test's worst case; columns 3 and 4 use the disagreement observed between configurations.

| Runs per agent | Better on these 129 tasks? | Better on new tasks? (129 drawn) | Better on new tasks? (58 drawn) |
|---|---|---|---|
| 1 | 12.5 (8.2 to 12.5) | 8.6 | 12.9 |
| 3 | 5.3 | 5.7 | 8.5 |
| 5 | 4.1 | 4.9 | 7.4 |
| 10 | 2.8 | 4.2 | 6.3 |
| no run-to-run noise | 0 | 3.4 (0 to 6.0) | 5.1 (0 to 8.9) |

An example from the pool: two configurations of one lab, each run four times on the same 61 tasks, differ by 6.6 points. With one run each the difference lies inside the noise band (exact McNemar p = 0.29); with three runs each it lies outside (p = 0.020).

### 4.4 Scores on different task subsets are not comparable (planned)

The runs use 52 different task subsets, and no subset is shared by two labs. The Rasch model puts agents and tasks on one scale, so we can ask what one agent would score on each subset. For a median 31B configuration the answer ranges from 5.6% to 66.7%, against 32.9% on all 129 tasks (Figure 3). Both extremes are subsets that a lab chose from its own earlier outcomes. Without the six subsets of that kind the range is 17.7% to 64.4%, and 25.5% to 39.3% among those of at least 30 tasks. Choosing tasks from earlier results distorts a score most; small subsets come next. On held-out cells the model predicts a run's resolve rate within 1.9 points on average (Brier score 0.074, against 0.105 for task means alone).

![Figure 3](figures/fig6_subset_comparability.png)

Figure 3. Predicted score of one median 31B configuration on each of the 52 task subsets. Orange: subsets a lab chose from its earlier outcomes.

### 4.5 The labs were not running the same benchmark (exploratory)

Eight labs published control runs: the reference patch that ships with each task, or no patch, applied in their own environment. Four of them audited all 129 tasks in the competition's stock sandbox. These audits agree on 120 tasks: the reference patch passes in all four on 69 and fails in all four on 51. We call a task *valid* when its reference patch passes in all four audits and the task does not pass without a patch; 67 tasks are valid. The failures come from the environment, not from the tasks: with per-task dependencies, one lab's Docker audit passes the reference patch on 105 tasks, among them 33 of the 51.

A quarter of all pooled attempts (24.6%) were made on those 51 tasks, and that figure hides two groups (Figure 4). Ten labs resolved none of them: five made no such attempt, and the other five failed all 388. In the stock sandbox these attempts could not succeed. Four labs resolved 21.6% of their 3,331 attempts on them, and all four document a test environment that differs from the stock sandbox. The same task identifiers stood for different benchmarks.

![Figure 4](figures/fig2_unpassable_attempts.png)

Figure 4. Share of each lab's attempts made on the 51 tasks whose reference patch fails in the stock sandbox. Blue: the lab resolved some of them. Grey: it resolved none.

Of the 41 floor tasks of Section 4.1, 23 are valid: hard for these models, not broken. Only 41 of the 129 tasks are both valid and discriminating. On the valid tasks alone the picture holds: reliability 0.86, flip rate 11.9%.

## 5. Replication

Neither check was planned, and neither dataset entered the numbers above. First, a pre-registered study by another participant [12] published 4,056 attempts on 104 tasks with three model sizes (E4B, 12B and 26B-A4B) in a repaired environment. Its per-task solve rates correlate at 0.84 with our consensus over the 91 shared tasks (Figure 5), and its flip rate between seeds is 13.3%.

Second, a broader search made after the results found nine further labs with 9,876 attempts. On their own they give a split-half reliability of 0.88, 46 of 99 tasks at the floor, a noise share of 22.1% and a lab-level flip rate of 9.4% (95% interval 5.5% to 13.2%). Task difficulty estimated from the 14 registered labs correlates at 0.89 with difficulty estimated from the nine new ones. This check is partial: 16 of 33 licensed candidates were examined, in an order fixed before their outcomes were read, and only the largest new source was recounted independently.

A noise band built from the pooled difficulties covers 96.1% of the repeat pairs of a lab that was left out of the fit, and 95.7% of those in the external dataset.

![Figure 5](figures/fig7_external_validation.png)

Figure 5. Consensus solve rate in the pooled table against the solve rate in the external dataset, one dot per task.

## 6. What teams and organisers can do

For teams:

1. Before any agent run, apply the reference patch and no patch to every task in your environment, and report the valid set.
2. Report repeated runs and their flip rate, never a single run.
3. Compare agents on the same tasks with a paired exact test, and plan the runs: about four per agent for a 5-point gain on 129 tasks.
4. Name the task subset, do not choose it from earlier outcomes, and prefer 50 tasks or more.
5. Treat a development-set gain of about 3 points or less as unproven for new tasks.
6. Publish per-task outcomes with a licence. This study exists because 23 labs did.

For organisers: ship per-task environments, since 51 of 129 public tasks fail their own reference patch in the stock sandbox. On a leaderboard split of about 58 tasks, the size participants infer, two agents that truly differ by 5 points finish in the wrong order 12% of the time, and resubmitting one unchanged agent five times raises its best score by 3.6 points on average.

## 7. Limits

The study is observational: labs differ in agent, budget, hardware, test environment and scorer, and nothing was controlled. Teams publish what they choose, so the pool is not a random sample of what was run. "Same configuration" rests on each lab's labels; under the strictest reading the flip rate is 10.2%. Task status rests on four audits of the stock sandbox. The variance between agents within a lab rests on 2 to 7 degrees of freedom, so the floor for new tasks is uncertain. The validity analysis, the sensitivity analyses and the replications were not pre-registered. Nothing here measures the hidden test set.

## Reproducibility and use of AI

Repository: https://github.com/yuvrajtiwary-bitmesraece/gemma4-many-labs (Apache-2.0). Ten commands rebuild every number, table and figure from the committed pooled table; a second path rebuilds the table from the sources at pinned commits. The repository holds the pooled table with a datasheet, the task lists (67 valid tasks; 41 valid and discriminating) and `tools/compare_runs.py`, which applies the noise band to anyone's result files.

The analysis plan, code, adapters, figures and results report were produced with Claude (Anthropic) under the author's direction. So were the independent review and the audits, by instances that had not written the code they checked. This text was drafted with Claude from the generated results report and reviewed by the author.

## Acknowledgements

The credit for every run belongs to the lab that produced it; the repository lists the 23 sources with commit and licence. We thank T. Toeda for the external dataset.

## References

1. Klein et al. Investigating Variation in Replicability: A "Many Labs" Replication Project. Social Psychology 45(3), 2014.
2. Liu et al. Coding Agents Have Converged. arXiv:2609.17394, 2026.
3. Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024.
4. Bjarnason, Silva, Monperrus. On Randomness in Agentic Evals. arXiv:2602.07150, 2026.
5. Ge et al. Agent psychometrics: Task-level performance prediction in agentic coding benchmarks. arXiv:2604.00594, 2026.
6. Miller. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
7. Card et al. With Little Power Comes Great Responsibility. EMNLP 2020.
8. Chowdhury et al. Introducing SWE-bench Verified. OpenAI, 2024.
9. Dietterich. Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation 10(7), 1998.
10. Rasch. Probabilistic Models for Some Intelligence and Attainment Tests. Copenhagen, 1960.
11. Brennan. Generalizability Theory. Springer, 2001.
12. Toeda. Scaffolding Helps the Smallest Gemma 4 Agent, Not the Larger Ones. Zenodo, doi:10.5281/zenodo.23834104, 2026.
