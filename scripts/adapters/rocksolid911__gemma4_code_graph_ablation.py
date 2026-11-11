"""Adapter for rocksolid911/gemma4-code-graph-ablation ("Finding the Code Isn't the Problem")."""
import json
import re

REPO = "rocksolid911/gemma4-code-graph-ablation"

# The lab states that some dev20 tasks could not be scored in its Kaggle notebook runs (see CARD["decisions"]).  True
# drops those attempts (README rule: attempts the source marks as invalid for infrastructure reasons are dropped);
# False would yield all 20 rows of every run, as the lab's own tables count them (740 attempts).
DROP_UNSCOREABLE = True
# requests_6644 is the one requests/httpx task of dev20 that is reported resolved in every Kaggle run (12 of 12), so its
# tests evidently pass on the host's copy of requests.  The README names only requests_7309 and httpx_3672 (the two that
# never pass there); True applies the notebook's own explanation ("a src/ repo (requests, httpx) would import the host's
# copy and never see the agent's patch") to all three.  False keeps the 12 resolved requests_6644 attempts of the Kaggle
# runs.
DROP_SRC_LAYOUT_ALWAYS_RESOLVED = True

CARD = {
    "eligible": True,
    "licence": {"spdx": "Apache-2.0",
                "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: '**Licence:** Apache 2.0 (see `LICENSE`).'"},
    "models": ["E4B", "31B", "12B"],
    "harness": "official",
    "summary": "37 runs of YAML-only agents on a frozen 20-task subset (dev20): a placebo-controlled ablation of the "
               "code-graph tools with Gemma 4 E4B on a laptop (Ollama, 25 runs) and 12 Kaggle vLLM runs (31B, one 12B, "
               "one E4B), all scored by the organizers' swegemma evaluator.",
    "evidence": {
        "model": "experiments/<run>/summary.json states the model that ran: \"model\": \"e4b-qat-ollama\" / "
                 "\"e4b-qat-w4a16\" / \"31b-qat-w4a16\" / \"12b-qat-w4a16\" with \"backend\": \"ollama\" / \"vllm-kaggle\". "
                 "Most laptop submissions still declare 'model: gemma-4-31b-it-qat-w4a16-ct' in agent.yaml; "
                 "harness/models.ollama.yaml: 'every competition alias goes to Ollama's Gemma 4 E4B (QAT) ... The "
                 "submission keeps declaring the real model name; only the local endpoint changes.' README.md: 'Laptop "
                 "runs use Ollama's E4B QAT build on an 8 GB GPU: 15 minutes, 60 calls, no compaction.'",
        "harness": "README.md: '`harness/` | Our glue around the organizers' evaluator, which is used unchanged'; "
                   "harness/run_eval.py: 'Run the official `swegemma eval` with our local sandbox fix-ups and the local "
                   "Ollama model.'; kaggle/make_eval_notebook.py: 'calls swegemma's `Evaluator.run()` on the frozen dev20 "
                   "ids with the subprocess sandbox and the submission's own eval_config budgets'",
        "systems": "make_tables.py groups the run folders into arms (dicts E4, O1, SINGLES, KAGGLE_E4, LATER_31B, "
                   "LATER_E4B) and each summary.json carries a `config_diff` sentence, e.g. 'E4 off: graph tools removed "
                   "from agent.yaml; Locate step uses grep instead of search_similar_code (base E6+E7 v2)'",
        "replicates": "README.md: '`seed` on E4B is a repeat index at temperature 0.2, not a sampling seed.'; "
                      "harness/postprocess.py: '`seed` is a repeat index at T > 0'; summary.json config_diff of the later "
                      "repeats: 'E4 on: repeat of E6+E7 v2 (graph tools on), new repeat index', 'Repeat (index 1) of the "
                      "8-min/28-call config', 'Repeat (index 1) of merge + 40-call cap', 'Repeat 1 of 20261101-e4b-merge-s0 "
                      "(identical snapshot)', 'Repeat of oracle O1: identical snapshot and data/variants/O1, repeat index "
                      "1'; make_tables.py lists the repeats of an arm together, e.g. '\"on\": [\"20261023-e4b-e6e7v2-s0\", "
                      "\"20261024-e4b-e4on-s1\", \"20261024-e4b-e4on-s2\"]', and tables_output.txt prints them as 'on: "
                      "runs=3 scores=[6, 4, 3]'. The submission folders of all repeats of an arm are byte-identical "
                      "(where a snapshot is published).",
        "task_sets": "harness/subsets/dev20.json: '\"method\": \"stratified by repo quota and gold-patch size tertile; "
                     "env-unstable tasks excluded\"', frozen 2026-10-21, seed 20261021; harness/subsets/build_dev20.py: "
                     "'Drop tasks whose gold patch is known to fail in the sandbox environment (ENV_UNSTABLE), so dev20 "
                     "measures the agent, not environment noise.' (22 ids taken from another team's gold sweep, "
                     "github.com/happyc0der/gemma-swe-agent). All 37 runs use these 20 tasks.",
    },
    "decisions": [
        "system = the lab's arm = run folder name without date and repeat suffix (-s<k>); the run folder is the run label",
        "20261023-e4b-e6e7v2-s0 is repeat 0 of arm 'on' (make_tables.py E4['on']; 'E4 on: repeat of E6+E7 v2'), so its "
        "system label is e4b-e4on",
        "20261028-31b-kaggle-baseline20-s0 uses the byte-identical submission of 31b-kaggle-budget8m28 ('Current "
        "submission unchanged; notebook now ships the fastapi test deps, so all 20 tasks can score') but the lab lists it "
        "as its own row with repeat index 0 and the test environment changed, so it stays a separate system",
        "e4b-e4skillunused (skill listed, never loaded, no prompt pointer) is not a repeat of e4b-e4skill: different "
        "system prompt, listed separately by the lab ('skill never loaded')",
        "Kaggle runs: requests_7309 and httpx_3672 are dropped from all 12, because the tests never saw the agent's patch "
        "(README.md: 'all of them predate the 2026-11-02 fix for `requests_7309` and `httpx_3672`'; "
        "kaggle/make_eval_notebook.py: 'This sandbox skips `pip install -e /workspace`, so a src/ repo (requests, httpx) "
        "would import the host's copy and never see the agent's patch.'). All 24 dropped attempts are unresolved.",
        "Kaggle runs: requests_6644, the third requests/httpx task of dev20, is dropped from all 12 for the same reason "
        "(DROP_SRC_LAYOUT_ALWAYS_RESOLVED). The README does not name it; the notebook's explanation covers the whole "
        "repository, and the lab's records fit it: reported resolved in 12 of 12 Kaggle runs (E4B and 12B included) while "
        "the agent edited the gold file in 1 of 12 (edit_hit), against 11 of 25 laptop runs where the workspace copy is "
        "imported. These 12 dropped attempts are all reported resolved, so the adapter's Kaggle counts are one below the "
        "lab's. This is a judgement call; the laptop attempts on requests_6644 are kept.",
        "Kaggle runs before 2026-10-28 (7 runs): README.md 'Runs before 2026-10-28 could score only 13 of 20 tasks'; "
        "kaggle/make_eval_notebook.py: 'Without them 7 of the 13 dev20 fastapi tasks can't even collect tests' (dev20 has "
        "10 fastapi tasks). The 7 ids are not published. Two fastapi tasks were resolved in these runs (fastapi_14463, "
        "fastapi_5077) and are therefore scoreable; the other 8 contain the 7 unscoreable ones and one scoreable task that "
        "cannot be told apart from this source, so all 8 are dropped in these runs (56 attempts, all unresolved; 7 of "
        "them are genuine failures that are lost). These runs keep 9 tasks (2 fastapi, 7 rich), which is below the "
        "pool's 10-task minimum, and carry '@13scoreable' in the system label. Set DROP_UNSCOREABLE = False to yield all "
        "20 rows per run as the lab's tables count them.",
        "the two oracle-O1 runs are read from the organizers' raw task_results.jsonl (the only runs whose raw output is "
        "published; identical outcomes in tasks/*.json); O1 repeat 1 ran in two parts split by a power-off "
        "(raw-first15 + raw-last5 = one run); raw/task_results.jsonl of that run is the concatenation and is skipped",
        "err: raw rows = the evaluator's error string is non-empty; tasks/*.json rows (the published records omit the "
        "error string) = stop_reason is timeout / max_tool_calls / max_turns / error, or 'ended' with no patch, or "
        "had_error is True in results/timing_200_runs.csv. Against the 240 rows where the evaluator's error flag is "
        "published, the stop_reason rule alone differs in 2 rows (submitted, then error).",
        "dev20 is not flagged outcome_selected_task_subset: it was drawn at random (stratified, fixed seed) after "
        "removing tasks whose gold patch fails, not from agent outcomes",
    ],
    "excluded": [
        "experiments/*/traces/*.jsonl: per-tool-call counts, no outcome",
        "results/timing_200_runs.csv: derived table of 10 runs already yielded (used only for its had_error flag)",
        "experiments/20261104-e4b-oracleO1-s1/raw/task_results.jsonl: concatenation of raw-first15 and raw-last5",
        "experiments/20261024-e4b-oracleO1-s0/tasks, experiments/20261104-e4b-oracleO1-s1/tasks: same runs as their raw files",
        "harness/subsets/dev20.json: task list with gold-patch size, no outcome",
        "leaderboard scores (tables_output.txt section I): hidden-test aggregates",
        "no control runs are published: harness/gold_check.py and the notebook's gold/null cell exist, their results "
        "(experiments/20261021-gold-dev20, experiments/20261021-envcheck-dev20) are not in the repository",
    ],
    "dropped": {
        "Kaggle runs, src/-layout tasks requests_7309 and httpx_3672 (tests imported the host's copy, not the patched workspace; all unresolved)": 24,
        "Kaggle runs, src/-layout task requests_6644 (same mechanism, not named by the README; all reported resolved)": 12,
        "Kaggle runs before 2026-10-28, 8 fastapi tasks never resolved there (7 could not collect tests, 1 cannot be told apart)": 56,
    },
    "flags": ["oracle_hints", "partial_runs", "sandbox_unsolvable_tasks_dropped", "interrupted_run_two_parts",
              "ollama_backend_for_laptop_runs", "fixed_20_task_subset_gold_valid"],
    "sampling": "temperature 0.2, top_p 0.95, no sampling seed in every published submission "
                "(experiments/*/submission/configs/sampling.yaml); README.md: '`seed` on E4B is a repeat index at "
                "temperature 0.2, not a sampling seed.' Of the 7 runs without a published submission, the merge "
                "configurations state 'same budgets and sampling' (31B) and 'sampling kept' (E4B) in summary.json "
                "config_diff; the temperature of the public Budget-Fit bundle is not stated ('thinking 4096, output 8192').",
    "leaderboard": [
        "tables_output.txt section I / make_tables.py LEADERBOARD ('Leaderboard (public, 58 tasks; one draw each)'; "
        "'Read from the Kaggle submissions page; recorded in each run's notes.md. Not derived from files here.')",
        "31B kagglefix, 5 min / 60 calls, graph tools on: LB 0.08 (submission 57553747); dev20 run "
        "20261024-31b-kaggle-kagglefix-s0 5/20 with 13 of 20 scoreable",
        "31B 8 min / 28 calls, tools on: LB 0.13 (57639721); dev20 runs 20261025-31b-kaggle-budget8m28-s0/-s1 7/20 and "
        "6/20 with 13 scoreable; the same submission later 8/20 (20261028-31b-kaggle-baseline20-s0)",
        "31B merge, 8 min / 40 calls, tools removed: LB 0.20 (12 of 58; 57907569) and 0.12 (7 of 58; 58013493, 'same "
        "files as 57907569, resubmitted'); dev20 runs 20261101-31b-kaggle-merge40-s0/-s1 10/20 and 11/20",
        "E4B E4 off, 5 min / 60 calls: LB 0.03 (2 of 58; 57867361); dev20 laptop runs e4b-e4off 5, 6, 5 of 20 and, with "
        "the scorer's compaction, e4b-e4off-compact 3 of 20 (Ollama, 15-minute limit, not the scorer's conditions)",
        "E4B merge, 5 min / 60 calls: LB 0.05 (3 of 58; 57974203); dev20 laptop runs e4b-merge 4 and 6 of 20 and, with "
        "the scorer's compaction, e4b-merge-compact 5 of 20 (Ollama, 15-minute limit)",
    ],
    "controls": [],
    "reported_numbers": [
        {"what": "E4B laptop arms, resolved of 20 per repeat (off / on / skill / placebo / O1)",
         "value": "[5, 6, 5] / [6, 4, 3] / [3, 6, 5] / [5, 3, 3] / [2, 4]", "where": "tables_output.txt section A",
         "adapter": "5,6,5 / 6,4,3 / 3,6,5 / 5,3,3 / 2,4 of 20", "match": True},
        {"what": "single E4B laptop arms, resolved of 20 (skill never loaded / E8 / O2 / O3)",
         "value": "4 / 3 / 6 / 6", "where": "tables_output.txt section B", "adapter": "4 / 3 / 6 / 6 of 20", "match": True},
        {"what": "other E4B laptop runs, resolved of 20 (kagglefix 60 calls / kagglefix 28 calls / merge / E4 off + "
                 "compaction / merge + compaction)", "value": "[5] / [3, 2] / [4, 6] / [3] / [5]",
         "where": "tables_output.txt section H", "adapter": "5 / 3,2 / 4,6 / 3 / 5 of 20", "match": True},
        {"what": "Kaggle runs before 2026-10-28, resolved count (31B off / 31B on / 31B on + skill / 12B on / E4B on); "
                 "the lab counts 20 tasks with 13 scoreable, the adapter keeps 9 tasks and drops the always-resolved "
                 "requests_6644",
         "value": "5 / 5 / 5 / 4 / 3", "where": "tables_output.txt section F",
         "adapter": "raw files before the drop: 5 / 5 / 5 / 4 / 3 of 20; after the drop: 4 / 4 / 4 / 3 / 2 of 9",
         "match": True},
        {"what": "later 31B Kaggle runs, resolved count (8 min / 28 calls, 13 scoreable / baseline20 / Budget-Fit / merge "
                 "/ merge 40 calls); the lab counts 20 tasks, the adapter keeps 9 and 17 tasks and drops requests_6644",
         "value": "[7, 6] / [8] / [8] / [7] / [10, 11]", "where": "tables_output.txt section G",
         "adapter": "raw files before the drop: 7,6 / 8 / 8 / 7 / 10,11 of 20; after the drop: 6,5 of 9 / 7 of 17 / 7 of "
                    "17 / 6 of 17 / 9,10 of 17",
         "match": True},
        {"what": "task by arm table: number of repeats that resolved each task (off, on, skill, placebo, O1), 20 x 5 cells",
         "value": "e.g. fastapi_14303 3 3 3 2 1; rich_2725 3 0 2 1 0", "where": "tables_output.txt section D",
         "adapter": "all 100 cells equal", "match": True},
        {"what": "every run: resolved task ids", "value": "summary.json resolved_ids (37 runs)",
         "where": "experiments/<run>/summary.json",
         "adapter": "identical id lists in all 37 runs before the drop (after it, requests_6644 is missing from the 12 "
                    "Kaggle lists)", "match": True},
        {"what": "timing table: task runs and solved", "value": "200 rows, 70 resolved",
         "where": "results/timing_200_runs.csv; timing_output.txt (23 + 21 + 26 solved)",
         "adapter": "the same 10 runs before the drop: 200 rows, 70 resolved; the adapter keeps 185 rows, 65 resolved "
                    "(15 Kaggle attempts on requests_6644, requests_7309 and httpx_3672 dropped)", "match": True},
    ],
}

MODEL = {"e4b-qat-ollama": "E4B", "e4b-qat-w4a16": "E4B", "31b-qat-w4a16": "31B", "12b-qat-w4a16": "12B"}
RUN_ID = re.compile(r"^(\d{8})-(.+)-s(\d+)$")
ARM_ALIAS = {"e4b-e6e7v2": "e4b-e4on"}                 # make_tables.py: first repeat of arm "on"
DATA = re.compile(r"^experiments/([^/]+)/(?:tasks/[^/]+\.json|(raw[^/]*)/task_results\.jsonl)$")
# Runs whose raw evaluator output is published, and the raw folders that make up one execution.
RAW_PARTS = {"20261024-e4b-oracleO1-s0": {"raw"}, "20261104-e4b-oracleO1-s1": {"raw-first15", "raw-last5"}}
SRC_LAYOUT_FAIL = {"requests_7309", "httpx_3672"}      # README: every Kaggle run predates the fix for these two
SRC_LAYOUT_PASS = {"requests_6644"}                    # same mechanism, the host's copy passes the tests
# Resolved in at least one run before the fix, hence scoreable there.  (Not used here: the reference-patch audit of
# another source on the same Kaggle image, sundarshahi/gemma4-agents-rarely-lose-a-fix results/audit/verdicts.jsonl,
# passes exactly these two and fastapi_14986 among the ten dev20 fastapi tasks; adding fastapi_14986 would give these
# runs 10 tasks.)
SCOREABLE_FASTAPI_BEFORE_FIX = {"fastapi_14463", "fastapi_5077"}
FASTAPI_FIX = "20261028"                               # README: "Runs before 2026-10-28 could score only 13 of 20 tasks"
ERROR_STOPS = {"timeout", "max_tool_calls", "max_turns", "error"}


def _meta(src):
    """run id -> (system, model, backend, date) from the per-run summary.json files."""
    out = {}
    for path, s in src.json(r"^experiments/[^/]+/summary\.json$"):
        run = path.split("/")[1]
        date, arm, _ = RUN_ID.match(run).groups()
        system = ARM_ALIAS.get(arm, arm)
        if s["backend"] == "vllm-kaggle" and date < FASTAPI_FIX:
            system += "@13scoreable"
        out[run] = (system, MODEL[s["model"]], s["backend"], date)
    return out


def _unscoreable(task, backend, date):
    if not DROP_UNSCOREABLE or backend != "vllm-kaggle":
        return False
    if task in SRC_LAYOUT_FAIL or (DROP_SRC_LAYOUT_ALWAYS_RESOLVED and task in SRC_LAYOUT_PASS):
        return True
    return date < FASTAPI_FIX and task.startswith("fastapi_") and task not in SCOREABLE_FASTAPI_BEFORE_FIX


def attempts(src):
    meta = _meta(src)
    had_error = {}
    for _, rows in src.csv(r"^results/timing_200_runs\.csv$"):
        for r in rows:
            had_error[(r["run"], r["task"])] = r["had_error"] == "True"
    for path, raw in src.read_many(DATA.pattern):
        run, raw_dir = DATA.match(path).groups()
        system, model, backend, date = meta[run]
        if raw_dir is None:
            if run in RAW_PARTS:
                continue                                   # yielded from the raw evaluator output instead
            rec = json.loads(raw.decode("utf-8"))
            stop = rec["stop_reason"]
            err = stop in ERROR_STOPS or (stop == "ended" and not rec["has_patch"]) \
                or had_error.get((run, rec["task_id"]), False)
            rows = [(rec["task_id"], rec["resolved"], err, stop)]
        else:
            if raw_dir not in RAW_PARTS.get(run, ()):
                continue                                   # merged copy of the two parts
            rows = []
            for line in raw.decode("utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.append((r["instance_id"], r["resolved"], bool(r.get("error")),
                                 r.get("error") or ("resolved" if r["resolved"] else "failed")))
        for task, resolved, err, status in rows:
            if _unscoreable(task, backend, date):
                continue
            yield {
                "file": path,
                "system": system,
                "run": run,
                "task": task,
                "y": int(resolved is True),
                "model": model,
                "harness": "official",
                "err": bool(err),
                "status": status,
            }
