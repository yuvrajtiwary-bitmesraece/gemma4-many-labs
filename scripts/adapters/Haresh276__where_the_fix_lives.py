"""Adapter for Haresh276/where-the-fix-lives ("Where the Fix Lives: Code-Graph Structure and the Triage Gap for
Local Coding Agents").

Eligible data: two full passes of Gemma 4 31B over the 129 public tasks in a Kaggle 4xL4 notebook that drives the
competition's swegemma Evaluator: the default sample agent, and the same agent with an "edit-plan checkpoint"
appended to the prompt.  The rest of the repository is a SWE-bench Verified analysis and fix-location features.
"""
import re

REPO = "Haresh276/where-the-fix-lives"

# The lab flags 27 fastapi tasks whose tests import packages that the sandbox lacks and reports every result on the
# remaining 102 ("I exclude them, leaving 102").  True follows the lab (README rule: attempts the source marks as
# invalid for infrastructure reasons are dropped); False would yield all 129 rows per run (29/129 and 32/129).
DROP_FLAGGED_UNSOLVABLE = True

CARD = {
    "eligible": True,
    "reason": "",
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License Apache License 2.0.'; "
                            "paper/paper.md '## 6. Release Apache-2.0 at github.com/Haresh276/where-the-fix-lives'"},
    "models": ["31B"],
    "harness": "official",
    "summary": ("Two single runs of Gemma 4 31B (QAT) on all 129 public tasks in one Kaggle 4xL4 session each, with the "
                "competition's sample agent (dummy LoRA adapters removed) under 6 min / 40 tool calls / 70 turns per task, "
                "2 tasks in flight: a base run and a run with an edit-plan checkpoint added to the prompt."),
    "evidence": {
        "model": "kaggle/wfl-gemma31b-129-tasks.ipynb cell 4: MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'  # research run: the 31B "
                 "model the paper studies; 'always strip the sample's DUMMY LoRA adapters so we measure the base model'",
        "harness": "kaggle/wfl-gemma31b-129-tasks.ipynb cell 10: 'from swegemma.evaluate import Evaluator', EvalConfig(..., "
                   "sandbox='subprocess', max_time_minutes=6.0, max_tool_calls=40, max_turns=70, ...), "
                   "rec.update(resolved=bool(r.resolved), test_exit_code=r.test_exit_code, ...); paper/paper.md s2: 'I ran Gemma 4 "
                   "31B (QAT, the competition's default agent and tools, vLLM on 4xL4) on all 129 of them in one Kaggle session'",
        "systems": "README.md: 'kaggle/wfl-gemma31b-checkpoint-arm.ipynb | The same notebook plus one cell adding the edit-plan "
                   "checkpoint to the prompt'; paper/paper.md s4: 'A second full run, identical except for the checkpoint text in the prompt'",
        "replicates": "none: one run per arm. paper/paper.md s4: 'one run per arm cannot separate it from run-to-run noise'; "
                      "s5: 'My Gemma run is one sample per task on 102 tasks'",
        "task_sets": "paper/paper.md s2: '27 fastapi tasks have tests that import packages missing from the sandbox, as the forum "
                     "reports [6]; Gemma solved 0 of them, and I exclude them, leaving 102'; flag = data/features_gemma_own129.csv "
                     "column needs_missing_pkg (src/gemma_tasks.py: test files of the fix commit mention inline_snapshot or dirty_equals)",
    },
    "decisions": [
        "two systems, one run each: 'base' (data/gemma/own/gemma31b_task_results_compact.csv) and 'checkpoint' (gemma31b_checkpoint_results_compact.csv); the prompt differs, so they are not replicates",
        "the 27 fastapi tasks flagged needs_missing_pkg are dropped from both runs (54 attempts, all resolved = 0) because the lab excludes them as unsolvable in the sandbox and reports 29/102 and 32/102; the flag is a static heuristic of the lab, not a per-attempt reference-patch check; set DROP_FLAGGED_UNSOLVABLE = False to yield 129 rows per run (29/129, 32/129)",
        "harness = official: the competition's swegemma Evaluator scored the attempts, but in the lab's own Kaggle notebook with sandbox='subprocess', research budgets (6 min, 40 calls, 70 turns) and concurrency 2, not on the scoring server",
        "err = non-empty 'err' column (timeout: 6-minute limit, ctx: 32k context overflow, turns, other); rows with test_exit_code -1 have an empty patch and are agent failures (y = 0), not infrastructure drops",
    ],
    "excluded": [
        "data/features_gemma_own129.csv, data/features_gemma_public55.csv: fix-location features of the tasks (task metadata, no outcomes); only the needs_missing_pkg flag of the first is read",
        "data/numbers.json: aggregated paper numbers; its 'gemma' block summarises another lab's runs (yosh95, 55 tasks, arms v8_noanz@ollama55 and v9_verify@ollama55, mean solve rate 0.318, 0.402 on 41 solvable); the per-task file data/gemma/pub/yosh_task_outcomes.csv is fetched by the lab and not in this repository",
        "data/features_verified.csv, data/localize_verified.csv, data/per_agent_coefs.csv: SWE-bench Verified tasks (features, BM25 localisation hits, regression coefficients of 38 non-Gemma-4 leaderboard agents), outside the development set",
    ],
    "dropped": {"task flagged needs_missing_pkg (tests import packages missing from the sandbox; lab excludes them)": 54},
    "flags": ["sandbox_unsolvable_tasks_dropped"],
    "reported_numbers": [
        {"what": "base run, resolved on solvable tasks", "value": "29/102", "where": "paper/paper.md s3.4; data/numbers.json gemma_own.solved / n_solvable",
         "adapter": "29/102", "match": True},
        {"what": "checkpoint run, resolved on solvable tasks", "value": "32/102 (4 gained, 1 lost)",
         "where": "paper/paper.md s4; README.md; data/numbers.json gemma_own.checkpoint_arm", "adapter": "32/102 (4 gained, 1 lost)", "match": True},
        {"what": "base run over all 129 rows", "value": "rate_all 0.225 (= 29/129), 0 of the 27 flagged tasks solved, 56 timeouts, 10 context overflows",
         "where": "data/numbers.json gemma_own; paper/paper.md s3.4",
         "adapter": "raw file before the drop: 29/129, flagged solved 0, err timeout 56, ctx 10 (after the drop: 40 timeouts in 102)", "match": True},
        {"what": "checkpoint run over all 129 rows", "value": "46 timeouts, 11 context overflows, 0 flagged solved",
         "where": "data/numbers.json gemma_own.checkpoint_arm; paper/paper.md s4",
         "adapter": "raw file before the drop: 32/129, err timeout 46, ctx 11, flagged solved 0", "match": True},
    ],
}

FLAGS = r"^data/features_gemma_own129\.csv$"
RUN = re.compile(r"^data/gemma/own/(gemma31b_(task|checkpoint)_results)_compact\.csv$")
SYSTEM = {"task": "base", "checkpoint": "checkpoint"}


def attempts(src):
    flagged = set()
    if DROP_FLAGGED_UNSOLVABLE:
        for _, rows in src.csv(FLAGS):
            flagged = {r["task_id"] for r in rows if r["needs_missing_pkg"].strip().lower() == "true"}
    for path, rows in src.csv(RUN.pattern):
        run, arm = RUN.match(path).groups()
        for row in rows:
            if row["instance_id"] in flagged:
                continue  # counted in CARD["dropped"]
            y = int(row["resolved"])
            err = row["err"].strip()
            yield {
                "file": path,
                "system": SYSTEM[arm],
                "run": run,
                "task": row["instance_id"],
                "y": y,
                "model": "31B",
                "harness": "official",
                "err": bool(err),
                "status": f"{'resolved' if y else (err or 'failed')}; test_exit_code={row['test_exit_code']}",
            }
