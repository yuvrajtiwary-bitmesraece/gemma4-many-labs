"""Adapter for lspecian/gemma4-moving-failures ("Moving Failures Around: a small coding agent acts on
information it is shown, not on instructions to find it")."""
import re

REPO = "lspecian/gemma4-moving-failures"

CARD = {
    "eligible": True,
    "licence": {"spdx": "MIT", "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Luis Specian'; "
                                           "README.md '## License MIT, see [LICENSE](LICENSE).'"},
    "models": ["31B", "E4B"],
    "harness": "official",
    "summary": "One lab's whole development log: 108 local experiments (ids E001-E113, 90 of them usable here) of "
               "~35 versions of a Google-ADK agent with a narrow scripted tool interface (plus a public stock-tools "
               "agent, LoRA adapters and a few E4B probes), run with the competition's swegemma harness on subsets "
               "of the dev set (mostly the 113-task list 'eval113'), one CSV of per-task outcomes per experiment.",
    "evidence": {
        "model": "README.md: 'The agent drives Gemma 4 31B-it (QAT w4a16, served by vLLM)'; paper/draft.md s7 "
                 "'Limitations: one agent model'; outcomes/runs.tsv names the exceptions: E097-E102 'model "
                 "gemma-4-e4b-it-qat-w4a16-ct', E082-E085 DeepSeek. Runs before E016 (no '_vllm' in the name) were "
                 "served by llama.cpp (scripts/serve_local.sh: 'Q4_0 GGUF instead of W4A16 compressed-tensors')",
        "harness": "README.md: 'The harness is the competition's `swegemma` package (0.2.11)'; "
                   "scripts/swegemma_local.py: '`swegemma` CLI with Kaggle's scorer settings for local runs ... "
                   "Per-task test environments ... The scorer's App context management'; scripts/run_ab.sh: "
                   "'Default: the stock published environment, as in all runs before 2026-10-03'",
        "systems": "README.md: 'outcomes/ One CSV per local run ...; runs.tsv maps each run id (E075_..., as cited "
                   "in the paper) to its agent version and pass count'. The run name carries the agent version and "
                   "the evaluation condition (kaggleslow = paced at 12 tok/s, real = 35 tok/s, neb = Nebius, b512 = "
                   "thinking budget, rft1/rft2/rft3 = LoRA adapter, e4b = model)",
        "replicates": "(1) column `repeat` (r1..r3) inside one CSV; scripts/run_ab.sh: 'Each run dir holds the exact "
                      "agent it evaluated ... and one sub-dir per repeat: results/<run>/r<k>/<task>/', "
                      "scripts/run_chunked.sh '<list> <rounds> ...'. "
                      "(2) E069+E070: paper/draft.md s1 'the same agent scored 46/113 twice with 6 tasks flipping "
                      "each way' and F1 table row 'E069/E070 -> E073 | 46 -> 47'; runs.tsv both 'agent-v22 c98bd1a'. "
                      "(3) E013+E014: docs/architecture.md s4 'Noise: the same agent scored 8/12 and 2/12 on the "
                      "same tasks on different nights'; paper/outline.md 'Same-agent noise: 8/12 vs 2/12 on "
                      "different nights'. "
                      "(4) E037+E038+E040, an INFERENCE, not a statement: runs.tsv all three 'agent-v16 e314d8b'; "
                      "paper/outline.md F4 'A local \"win\" (v16 > v14, 40.3% vs 35.8%)', where 35.8% is 81/226 "
                      "(E036, two repeats) and 40.3% is (92+88+93)/678, the three files pooled. 273/678 equals 91/226 "
                      "exactly, so a single file with 91 passes would give the same percentage, but the published "
                      "files have 92, 88 and 93 (40.7%, 38.9%, 41.2%); the pool is the only reading that fits the "
                      "published counts. No sentence calls the three files repeats. Same task list and order, same "
                      "4.5 min budget, 44-48 passes per repeat; flips across files 9-19 of 113, inside files 11-18. "
                      "The sensitivity analysis S2 splits this group. "
                      "NOT merged: E049 and E053 (scripts/rft3_relabel.py 'E048 v17b, E049/E053 v18c, E054 v18c+b256'; "
                      "scripts/rft2_build.py 'E053 (v18c), E049 (v18c), E048 (v17b); all eval113 x 1 round, "
                      "concurrency 4, compaction on, Kaggle's June template, ~10-12 tok/s'; runs.tsv both "
                      "'agent-v18c 5cde5fb'): one version label and no statement that they are repeats, and the "
                      "two sessions differ in speed (duration p50/p90/max 280/291/298 s against 337/436/525 s; "
                      "scripts/pace_check.py puts the E048/E049 session at about 10 tok/s against a 12 tok/s target)",
        "task_sets": "paper/draft.md s2: 'the competition's 129 tasks (fastapi, rich, requests, httpx); with per-task "
                     "environments 115 pass with the gold patch, and we use 113 (eval113)'; experiments/eval113.txt; "
                     "smaller lists: dev12 (12 tasks), a 4-task set (E009-E015), eval20, the 37 LoRA held-out tasks "
                     "(heldout / ho35), every third eval113 task (38, E103-E113), 23 tasks 'solved by some DeepSeek "
                     "run and by none of four Gemma runs' (E092-E094)",
    },
    "decisions": [
        "run = one experiment id + one value of the `repeat` column (E036_v14_eval113/r1); repeats inside one CSV "
        "are replicate runs of one system",
        "system = the experiment id (E064_v20b_real) by default: two experiments with the same agent tag are NOT "
        "merged unless the lab says they are the same configuration, because the name does not carry every "
        "condition (server, pacing, compaction, harness version changed between sessions)",
        "merged on the lab's own statements (see evidence.replicates): E069+E070 (v22_real), E013+E014 (v5); "
        "merged on an inference from a pooled percentage, not on a sentence that says 'repeat': E037+E038+E040 "
        "(v16_eval113). An independent audit graded the first two STRONG and the third WEAK; the sensitivity "
        "analysis S2 splits it",
        "E049 and E053 (v18c_kaggleslow) were merged in the first version of this adapter on two script docstrings "
        "that list both under one version label. The audit graded that evidence as unclear (no statement of "
        "repetition, different speed of the two sessions), and the plan counts unclear cases as different systems, "
        "so they are two systems (DEVIATIONS.md, Amendment 4)",
        "kept apart although runs.tsv gives the same agent tag and the name suffix is the same, because no "
        "statement calls them repeats: E003/E007 (sample_20m), E010/E012 (v4), E027/E028 (v13_vllm; empty patches "
        "21/36 vs 10/36), E043/E048 (v17b_kaggleslow; scripts/rft3_states.py groups only E048-E058 as one harness "
        "config), E049/E053 (v18c_kaggleslow), E064/E068 (v20b_real), E073/E075 (v24 on vast.ai vs Nebius), "
        "E103/E106/E108 (v32), E110/E112 (v37)",
        "y = 1 iff outcome == 'pass'; 'wrong' (non-empty patch that fails) and 'empty' (no patch) are y = 0",
        "err = the `error` column is non-empty (session timeout, call/turn budget, unknown tool, context overflow, "
        "test patch failed to apply)",
        "dropped as infrastructure: the 61 rows whose error is 'Sandbox execution error: "
        "litellm.InternalServerError' (the model server stopped answering; 43 rows) or 'Unexpected evaluation "
        "worker error: UnixHTTPConnectionPool' (Docker daemon; 18 rows). 44 of them never received a model "
        "response (llm_calls = 0) and 17 were in flight when the server or the daemon failed (2 of those carry a "
        "patch and the outcome 'wrong'). The lab is not consistent about such rows: in two paired comparisons it "
        "restricts itself to the tasks reached before the hang (paper/draft.md F1 '(9 vs 8 of 23 tasks)' for "
        "E112/E113, and 'On the 24 unseen tasks both runs reached before the server serving the adapter hung' for "
        "E095/E096), while outcomes/runs.tsv, the F1 row 'E106 -> E107' and the 296-attempt baseline count them as "
        "failures. Its own rule is 'tasks with a model response in both runs', not the error text; the adapter "
        "uses the error text because it is recorded per row. So runs.tsv task_runs exceed the adapter's n for "
        "E060, E061, E106-E109, E112, E113",
        "'@partial' in the system label: the run stopped before its task list was finished (E021 8 of 12, E059 87, "
        "E060 90, E061 91 and E071 19 of 113) or lost rows to the server failures above (E106-E109, E112, E113)",
        "LoRA runs are kept with the adapter in the label (E033 rft1, E056 rft2, E063 rft3; E031 is a zero, no-op "
        "adapter); rft1/rft2 were trained on the agent's own passing episodes on dev tasks, E056 and E063 are "
        "evaluated on the 37 tasks held out from that training",
        "E067_pub018_real is another team's public agent (verracodeguacas Budget-Fit) run by this lab on Gemma 4 "
        "31B: kept as a system of this lab",
        "E092-E094 run on 23 tasks selected because four earlier Gemma runs failed them: kept, flagged",
        "harness = official: the competition's swegemma graded every attempt, but locally, behind a pacing proxy "
        "and (from 2026-10-03) with per-task Python environments that the stock harness does not have",
        "two test environments: experiments up to E035 were graded in the stock published environment, experiments "
        "from E036 on (4,660 of the 5,320 pooled rows) with per-task environments (scripts/run_chunked.sh 'Per-task "
        "environments by default (TASK_ENV=1 ...)', scripts/run_ab.sh 'Default: the stock published environment, as "
        "in all runs before 2026-10-03'; README.md: the reference patch passes on 52 of 129 tasks in the stock "
        "environment and on 115 with per-task environments). One task shows the break in the data: rich_3468 "
        "passes in 33 of 69 attempts up to E035 and in 0 of 39 Gemma attempts from E036 on. No multi-run system "
        "spans the break",
        "handoff runs E088-E091 are excluded as mixed-model runs: a DeepSeek trajectory is replayed up to its first "
        "edit and Gemma only writes the fix",
    ],
    "excluded": [
        "outcomes/E076, E077, E080, E081, E087, E095, E096 (*_unseen*) and E078, E079 (*_ccs): 'unseen100', 99 "
        "tasks from 99 other repositories (experiments/unseen100.txt), not the development set; v24 34/99 twice",
        "outcomes/E082_ds_v4pro_eval113, E082_smoke, E083_flash_v24_paced, E084_flash_v24_generous, "
        "E085_flash_bundle_generous: non-Gemma teacher models (DeepSeek V4 Pro / V4.1 Flash) driving the agent; "
        "49, 1 of 3, 50, 67 and 75 of 113",
        "outcomes/E088_handoff_ours, E089_handoff_stock, E090_handoff_ours_info, E091_handoff_stock_info: DeepSeek "
        "V4.1 Flash exploration replayed, then Gemma 4 31B continues (mixed-model, 23 outcome-selected tasks); "
        "9/14, 6/20, 6/14, 6/20",
        "outcomes/runs.tsv: index derived from the per-run CSVs (run, agent, task_runs, passed); used for "
        "verification only",
        "gold-patch control: no per-task file; README.md 'The harness's shared environment lets 52 of 129 gold "
        "patches pass; per-task environments raise that to 115'",
        "leaderboard scores (0.12-0.24 on the 58 public hidden tasks, paper F6): hidden tests, aggregates only",
        "raw traces: not published ('Results' raw traces are not included because they embed the tasks' problem "
        "statements')",
    ],
    "dropped": {"not run / infrastructure": 61},
    "flags": ["fine_tuned", "partial_runs", "outcome_selected_task_subset", "modified_test_environment"],
    "reported_numbers": [
        {"what": "passed per run, all 90 included experiment ids", "value": "runs.tsv column `passed`",
         "where": "outcomes/runs.tsv", "adapter": "identical for all 90", "match": True},
        {"what": "task_runs per run, all 90 included experiment ids", "value": "runs.tsv column `task_runs`",
         "where": "outcomes/runs.tsv",
         "adapter": "identical for 82; lower by 5, 3, 5, 2, 10, 8, 13, 15 in E060, E061, E106, E107, E108, E109, "
                    "E112, E113 (the 61 infrastructure rows dropped)",
         "match": False},
        {"what": "E069, E070 (same agent twice)", "value": "46/113 twice, 6 tasks flipping each way",
         "where": "paper/draft.md s1", "adapter": "46/113, 46/113; 6 solved only by E069, 6 only by E070",
         "match": True},
        {"what": "v16 vs v14 pooled pass rate", "value": "40.3% vs 35.8%", "where": "paper/outline.md F4",
         "adapter": "273/678 = 40.3% (E037+E038+E040; equal to 91/226, see evidence.replicates), 81/226 = 35.8% "
                    "(E036)", "match": True},
        {"what": "E013, E014 (same agent, different nights)", "value": "8/12 and 2/12",
         "where": "docs/architecture.md s4", "adapter": "8/12, 2/12", "match": True},
        {"what": "E064, E065, E066 pass", "value": "43, 43, 42 of 113", "where": "paper/draft.md F1",
         "adapter": "43, 43, 42", "match": True},
        {"what": "E073; E075 pass", "value": "47; 45 of 113", "where": "paper/draft.md F1, F3",
         "adapter": "47; 45", "match": True},
        {"what": "E106 -> E107 pass (38 tasks)", "value": "13 -> 12", "where": "paper/draft.md F1",
         "adapter": "13/33 -> 12/36 (5 and 2 server-failure rows dropped; the paper's wrong / empty columns, "
                    "15 -> 20 and 10 -> 6, count those rows)", "match": True},
        {"what": "E112 vs E113", "value": "9 vs 8 of 23 tasks", "where": "paper/draft.md F1",
         "adapter": "E113 9/23, E112 9/25; 8 vs 9 on the 22 tasks valid in both",
         "match": False},
        {"what": "E092 (30 min / 100 calls on the 23 tasks)", "value": "4/23", "where": "paper/draft.md F4",
         "adapter": "4/23", "match": True},
        {"what": "rft1 E033 vs E032; rft2 E056 vs E055; rft3 E063 vs E062", "value": "17 vs 16/36; 8 vs 5/37; "
         "14 vs 16/37", "where": "docs/lora.md s4", "adapter": "17 vs 16/36; 8 vs 5/37; 14 vs 16/37", "match": True},
    ],
}

RUN = re.compile(r"^outcomes/(E\d{3}_\w+)\.csv$")

# Tasks outside the development set (unseen100 and its co-change subset).
NOT_DEV_TASKS = {"E076_neb_v24_unseen", "E077_neb_v25_unseen", "E078_neb_v25_ccs", "E079_neb_v26_ccs",
                 "E080_bundle_unseen", "E081_v24_unseen_rerun", "E087_v27_unseen", "E095_v30_unseen",
                 "E096_v30ds1_unseen"}
# The agent is driven by DeepSeek, or DeepSeek's trajectory is replayed before Gemma continues.
NOT_GEMMA = {"E082_ds_v4pro_eval113", "E082_smoke", "E083_flash_v24_paced", "E084_flash_v24_generous",
             "E085_flash_bundle_generous", "E088_handoff_ours", "E089_handoff_stock", "E090_handoff_ours_info",
             "E091_handoff_stock_info"}
# Experiments the lab itself describes as one configuration run again (CARD["evidence"]["replicates"]).
SAME_SYSTEM = {
    "E013_v5": "E013+E014_v5", "E014_v5": "E013+E014_v5",
    "E037_v16_eval113": "E037+E038+E040_v16_eval113", "E038_v16_eval113": "E037+E038+E040_v16_eval113",
    "E040_v16_eval113": "E037+E038+E040_v16_eval113",
    "E069_v22_real": "E069+E070_v22_real", "E070_v22_real": "E069+E070_v22_real",
}
# Stopped before the task list was finished, or lost rows to a server failure.
PARTIAL = {"E021_v9_vllm", "E059_v18d_real", "E060_v19b_real", "E061_v19b512_real", "E071_v23_real", "E106_v32",
           "E107_v35", "E108_v32", "E109_v36", "E112_v37", "E113_v38b"}
INFRASTRUCTURE = ("Sandbox execution error: litellm.InternalServerError", "Unexpected evaluation worker error")


def attempts(src):
    for path, rows in src.csv(RUN.pattern):
        name = RUN.match(path).group(1)
        if name in NOT_DEV_TASKS or name in NOT_GEMMA:
            continue
        system = SAME_SYSTEM.get(name, name) + ("@partial" if name in PARTIAL else "")
        model = "E4B" if name.endswith("_e4b") else "31B"
        for row in rows:
            error = (row.get("error") or "").strip()
            if error.startswith(INFRASTRUCTURE):
                continue
            outcome = row["outcome"].strip()
            yield {
                "file": path,
                "system": system,
                "run": f"{name}/{row['repeat'].strip()}",
                "task": row["instance_id"].strip(),
                "y": int(outcome == "pass"),
                "model": model,
                "harness": "official",
                "err": bool(error),
                "status": f"{outcome}: {error}" if error else outcome,
            }
