"""Adapter for veljjanoski/gemma4-developer-agent ("Better calls, not more calls")."""
import re

REPO = "veljjanoski/gemma4-developer-agent"

CARD = {
    "eligible": True,
    "licence": {"spdx": "MIT", "evidence": "LICENSE: 'MIT License Copyright (c) 2026 Daniel Veljjanoski'; README.md '## License': 'MIT, see `LICENSE`.'"},
    "models": ["31B", "E4B"],
    "harness": "official",
    "summary": ("60 runs of ADK submissions (own versions v3-v15 plus three public third-party configurations; single-agent "
                "except v15, e4b_bf_seq and e4b_loc, which add a locator sub-agent) "
                "on three rigs: a local llama.cpp rig (31B, 40-task dev set and a 23-task swing bench), Kaggle notebooks in "
                "the scorer's stack (31B and E4B, 61 gold-checked held-out tasks) and a local vLLM rig (E4B, the same 61 tasks)."),
    "evidence": {
        "model": ("README.md: 'drives the fixed model `gemma-4-31b-it-qat-w4a16-ct` (or, for the Small Model Prize, "
                  "`gemma-4-e4b-it-qat-w4a16-ct`)'; APPENDIX.md E: local rig serves 'Google's QAT Q4_0 GGUF of Gemma 4 31B' with "
                  "llama.cpp; APPENDIX.md G: 'E4B runs (Small Model Prize), the same rig with every configuration on "
                  "`gemma-4-e4b-it-qat-w4a16-ct`'; APPENDIX.md I: local_e4b = 'The E4B model (`gemma-4-e4b-it-qat-w4a16-ct`, W4A16) "
                  "served by vLLM 0.19.1 in Docker on one RTX 3090'"),
        "harness": ("rig/run_agent.py: 'from swegemma.evaluate import Evaluator' (swegemma 0.2.7, docker sandbox, local patches "
                    "rig/local_env.py, rig/local_pin.py); APPENDIX.md G: 'each task run and verified by the harness's `Evaluator` "
                    "(swegemma 0.2.11, adk-submission 0.2.13, subprocess sandbox ...)'; rig/export_local_e4b.py reads "
                    "'runs/{run}/task_results.jsonl' of the 'swegemma harness'"),
        "systems": ("README.md Results table (one row per version v3..v15); APPENDIX.md F 'Run index' (run -> version, task list, "
                    "minutes); APPENDIX.md G and I tables 'Configurations (first = reference)' per run; scorer_stack/build_ab.py and "
                    "build_small.py RUNS give the exact text replacements of the arms of runs ab1-ab5 and small1-small2; ab6, "
                    "small3 (build_ab_lora.py VERSIONS) and the local E4B arms are described in README / APPENDIX prose only"),
        "replicates": ("v11@kaggle x4 and v10@kaggle x4: scorer_stack/ab_stats.py \"RUNS = {'v11': ['eval_base', 'ab1_base', 'ab2_base', "
                       "'ab3_base'], 'v10': ['ab2_think2048', 'ab3_think2048', 'ab4_v10', 'ab5_v10']}\" and 'flips per pair of identical "
                       "runs'; README.md 'v11 averages 18.25 of 61 in four runs, v10 22.00 in four (ab2 to ab5)'. "
                       "v19@kaggle x2: APPENDIX.md G 'ab5_v19 ... | davis0178's public v19' and 'ab6_v19, ab6_v19_issue_script | v19; v19 "
                       "with the issue text templated ...'. "
                       "e4b_v10@kaggle x3: scorer_stack/build_small.py 'run 2: spend it, from e4b_v10: the reference again (noise)'; "
                       "APPENDIX.md I 'the same configuration solved 7, 7 and 4 in the scorer's stack (small1 to small3)'. "
                       "v10@local x2: APPENDIX.md F 'v10_dev40_s6, v10_dev40_s6_r2 | v10, thinking 2,048, twice'. "
                       "v11@local-swing x2, v15@local-swing x2: APPENDIX.md C 'each candidate runs twice on the bench, unseeded, "
                       "alternating (v15, v11, v15, v11), local 6.0 minutes'. "
                       "e4b_v10@local-vllm x3: APPENDIX.md I 'e4b_local1, e4b_local2, e4b_local3 | v10 (`submissions/e4b_v10`), three "
                       "draws'. e4b_bf@vllm x5 (Kaggle small3_e4b_bf and local ab1..ab4_e4b_bf, pooled by the lab in "
                       "analysis/bestofn_oracle2.py, see decisions): README.md '(6, 6, 4 and 5, 4, 5 of 61 against 6 and 5 for the unchanged files)', "
                       "'solved 10 and 7 of 61 in two paired runs against 7 and 6 for the unchanged files'; APPENDIX.md I 'the two "
                       "Budget-Fit draws [differ] on 3'. e4b_bf_seq@local-vllm x2: APPENDIX.md I 'a read-only locator sub-agent ... "
                       "before the same fixer, two draws'. e4b_loc@local-vllm x2: APPENDIX.md I 'ab4_e4b_bf, ab4_e4b_loc | the same "
                       "pair, second draw'"),
        "task_sets": ("README.md Contents: 'the 40-task development set, the swing bench (17 swing + 6 always-solved tasks), the 65 "
                      "held-out tasks and the 61 of them that pass the gold/empty check in the scorer's stack' (data/dev40.txt, "
                      "swing23.txt, heldout65.txt, valid61.txt); dev40 and heldout65 together are the 105 public tasks that passed "
                      "the lab's local gold/empty check (APPENDIX.md H: gold passes 107/129, empty passes 2/129)"),
    },
    "decisions": [
        "system = '<configuration>@<rig>'; rigs: 'local' = llama.cpp Q4_0 GGUF 31B on one RTX 3090 (results/*.csv), 'kaggle' = "
        "scorer's stack on 4x L4, vLLM W4A16 (results/scorer_stack), 'local-vllm' = E4B W4A16 on vLLM on one RTX 3090 "
        "(results/local_e4b). The same agent files on two rigs are different systems (hardware, sandbox image and harness "
        "version differ; the llama.cpp rig also uses another weights format, time budget and task set), with one exception "
        "made after an independent audit on 2026-11-11: Budget-Fit's unchanged files on E4B are one system 'e4b_bf@vllm' "
        "across the Kaggle run small3_e4b_bf and the local vLLM runs ab1..ab4_e4b_bf, because the lab itself pools them as "
        "repeats (analysis/bestofn_oracle2.py:8 and analysis/bestofn_selector.py:107: 'E4B bf x3 (same files)' = Kaggle "
        "small3 e4b_bf + local ab1_e4b_bf + ab2_e4b_bf; APPENDIX.md:180 reports 'E4B Budget-Fit 6.3 of 61 per run', which only "
        "that cross-rig group reproduces). e4b_v10 stays split by rig: the lab contrasts the rigs for it (APPENDIX.md:192, "
        "rig/export_local_e4b.py:3 'rig validation') and never pools those six runs",
        "harness = official for all three rigs: every attempt is run and verified by the competition's swegemma Evaluator; the "
        "local rigs add environment patches (rig/local_env.py, rig/local_pin.py) and scale minutes by 0.75 on the llama.cpp rig",
        "the four reference arms 'base' of eval/ab1/ab2/ab3 are one system (v11@kaggle) and think2048 of ab2/ab3 plus v10 of "
        "ab4/ab5 are one system (v10@kaggle), exactly as the lab pools them in scorer_stack/ab_stats.py",
        "eval_base and small3_e4b_v10 ran on a server with LoRA adapters loaded ('LoRA enabled, which made every configuration of "
        "that run slower', APPENDIX.md G); the lab still counts them as the same configuration, so they keep the pooled label",
        "ab5_v19 and ab6_v19 are one system (v19@kaggle): both are labelled 'v19' (davis0178's public v19, the reference arm) in "
        "APPENDIX.md G; the lab gives no pooled statistic for this pair and ab6 had two arms on the server instead of three",
        "v11_swing_r1/r2 are kept apart from v11_dev40_s6 (same submissions/v11 and 6.0 local minutes per rig/swing_queue.sh, but "
        "the lab never states they are repeats of the dev40 run, the rig was patched in between (cutoff_tool_call_local fix) and "
        "the swing bench is an outcome-selected task subset: APPENDIX.md C 'It was chosen after the fact from the runs it now judges')",
        "v3_dev40_defaults is its own system (llama.cpp default sampling top_k 40 / min_p 0.05 and unscaled 4.0 minutes); "
        "APPENDIX.md F: '16 tasks re-run after the shadowing fix', i.e. the file mixes two executions",
        "v6_redo14964.csv (one task re-run after a Docker error) is a shard of run v6_dev40_s3; the original row with "
        "end=docker_error_local is dropped as an infrastructure failure",
        "end=cutoff_tool_call_local (7 rows; 'local-only failures', a tool call cut off at the output limit ended the task on the "
        "llama.cpp rig) is kept as y=0 with err=True because the lab counts these tasks as unsolved in its totals of 40 and "
        "does not re-run them, although rig/local_env.py describes the cut-off as a property of the local rig ('a "
        "non-retryable model error that ends the task')",
        "the two swing-bench systems (v11@local-swing, v15@local-swing) are repeats on a task subset chosen for unstable "
        "outcomes; their flip rates (30% and 35%) are not typical, and the sensitivity analysis S2b removes them",
        "end=unknown_tool (ab3_e4b_loc, ab4_e4b_loc: 'One task per draw was lost to the locator calling an edit tool outside its "
        "list') is an agent error: y=0, err=True",
        "third-party public configurations measured by this lab (davis0178's v19, Hanh Tran's Gemma 4 Baseline v4, Wilmer E. "
        "Henao's Budget-Fit files) are Gemma 4 agents run by this lab and are kept as systems v19, hanh_v4, e4b_v19*, e4b_bf*",
        "err = the column `end` is non-empty (session_timeout, call_cap, turn_limit/turn_cap, no_submit, test_patch_conflict, "
        "patch_apply_failed, unknown_tool, cutoff_tool_call_local)",
    ],
    "excluded": [
        "results/scorer_stack/eval_gold.csv: control, gold patch in place of the agent step; passes 62 of 65 held-out tasks",
        "results/scorer_stack/eval_empty.csv: control, empty patch; passes 1 of 65 held-out tasks (61 tasks kept = data/valid61.txt)",
        "APPENDIX.md H (no per-task file): local gold runs over all 129 public tasks pass 19/129 without pins, 97/129 with pins, "
        "107/129 after the shadowing fix (FastAPI 62/67, Rich 39/48, Requests 5/13, httpx 1/1); local empty patch passed 16 of the "
        "40 dev tasks before the shadowing fix and 2 of 129 after it",
        "data/*.txt: task-id lists, no outcomes",
        "README.md leaderboard column (hidden-test scores 0.05-0.18): not per-task",
        "APPENDIX.md B replay tables: counts of first tool calls in replayed moments, not task outcomes",
    ],
    "dropped": {"not run / infrastructure": 1},
    "flags": ["fine_tuned", "outcome_selected_task_subset", "tasks_restricted_to_gold_checked", "composite_runs"],
    "reported_numbers": [
        {"what": "local llama.cpp rig, solved of 40: v3, v4, v5, v6, v7, v8, v9, v10 (twice), v11, v13, v14",
         "value": "5, 9, 14, 14, 15, 12, 11, 18 then 15, 15, 15, 13", "where": "README.md Results table",
         "adapter": "5, 9, 14, 14, 15, 12, 11, 18 and 15, 15, 15, 13", "match": True},
        {"what": "v3 with llama.cpp defaults (v3_dev40_defaults)", "value": "13/40", "where": "APPENDIX.md H",
         "adapter": "13/40", "match": True},
        {"what": "swing bench, solved of the 17 swing tasks: v15 r1, r2; v11 r1, r2", "value": "8, 9; 10, 12",
         "where": "README.md Results table", "adapter": "8, 9; 10, 12 (on data/swing17.txt; 13, 13; 15, 18 of all 23)", "match": True},
        {"what": "scorer's stack, solved of 61: eval (base, lora_e1, lora_e2); ab1 (base, calls40, temp10); ab2 (base, think2048, "
                 "think4096); ab3 (base, think2048, think2048_7min); ab4 (v10, think1024, think2048_calls24); ab5 (v19, hanh_v4, v10); "
                 "ab6 (v19, v19_issue_script)",
         "value": "18, 16, 17; 19, 18, 17; 18, 23, 18; 18, 20, 17; 24, 23, 20; 18, 18, 21; 18, 19", "where": "APPENDIX.md G",
         "adapter": "18, 16, 17; 19, 18, 17; 18, 23, 18; 18, 20, 17; 24, 23, 20; 18, 18, 21; 18, 19", "match": True},
        {"what": "pooled means: v11 four runs, v10 four runs", "value": "18.25 and 22.00 of 61", "where": "README.md",
         "adapter": "18.25 and 22.00", "match": True},
        {"what": "flips between identical runs in the scorer's stack", "value": "6 to 11 tasks (v11 pairs), 5 to 8 (v10 pairs)",
         "where": "APPENDIX.md G", "adapter": "v11 pairs 6..11, v10 pairs 5..8", "match": True},
        {"what": "E4B scorer's stack, solved of 61: small1 (v19, v19_script, v10); small2 (v10, verify, think8k); small3 (v10, bf, "
                 "bf_ep1, bf_final)", "value": "5, 5, 7; 7, 7, 7; 4, 8, 4, 6", "where": "APPENDIX.md G",
         "adapter": "5, 5, 7; 7, 7, 7; 4, 8, 4, 6", "match": True},
        {"what": "local E4B rig, solved of 61: e4b_local1-3; ab1 (bf, locate, grepfix); ab2 (bf, seq_r1, seq_r2); ab3 (bf, loc); "
                 "ab4 (bf, loc)", "value": "6, 4, 5; 6, 6, 4; 5, 4, 5; 7, 10; 6, 7", "where": "APPENDIX.md I",
         "adapter": "6, 4, 5; 6, 6, 4; 5, 4, 5; 7, 10; 6, 7", "match": True},
        {"what": "identical configurations flipped (local rig)", "value": "5 of 40 (v10 twice); 7 and 8 of 23 (v11 and v15 twice)",
         "where": "APPENDIX.md C", "adapter": "v10: 5 of 40; v11: 7 of 23; v15: 8 of 23", "match": True},
        {"what": "local E4B draws differ", "value": "e4b_loc draws on 7 tasks, the two Budget-Fit draws (ab3, ab4) on 3",
         "where": "APPENDIX.md I", "adapter": "e4b_loc 7, e4b_bf ab3 vs ab4 3", "match": True},
        {"what": "controls (excluded): gold and empty patch in the scorer's stack", "value": "62 and 1 of 65",
         "where": "APPENDIX.md G", "adapter": "not yielded (file counts 62 and 1 of 65)", "match": True},
    ],
}

# file stem -> configuration label, per rig.  Every released result file is listed; nothing is inferred from scores.
LOCAL = {  # results/<stem>.csv: llama.cpp rig, 31B Q4_0 GGUF
    "v3_dev40_defaults": "v3-defaults@local",
    "v3_dev40_s3": "v3@local",
    "v4_dev40_s3": "v4@local",
    "v5_dev40_s3": "v5@local",
    "v6_dev40_s3": "v6@local",
    "v6_redo14964": "v6@local",
    "v7_dev40_s3": "v7@local",
    "v8_dev40_s3": "v8@local",
    "v9_dev40_s3": "v9@local",
    "v10_dev40_s6": "v10@local",
    "v10_dev40_s6_r2": "v10@local",
    "v11_dev40_s6": "v11@local",
    "v13_dev40_s375": "v13@local",
    "v14_dev40_s6": "v14@local",
    "v11_swing_r1": "v11@local-swing",
    "v11_swing_r2": "v11@local-swing",
    "v15_swing_r1": "v15@local-swing",
    "v15_swing_r2": "v15@local-swing",
}
KAGGLE = {  # results/scorer_stack/<stem>.csv: scorer's stack, vLLM W4A16
    "eval_base": ("v11@kaggle", "31B"),
    "eval_lora_e1": ("v11+lora-e1@kaggle", "31B"),
    "eval_lora_e2": ("v11+lora-e2@kaggle", "31B"),
    "ab1_base": ("v11@kaggle", "31B"),
    "ab1_calls40": ("v11-calls40@kaggle", "31B"),
    "ab1_temp10": ("v11-temp10@kaggle", "31B"),
    "ab2_base": ("v11@kaggle", "31B"),
    "ab2_think2048": ("v10@kaggle", "31B"),
    "ab2_think4096": ("v11-think4096@kaggle", "31B"),
    "ab3_base": ("v11@kaggle", "31B"),
    "ab3_think2048": ("v10@kaggle", "31B"),
    "ab3_think2048_7min": ("v10-7min@kaggle", "31B"),
    "ab4_v10": ("v10@kaggle", "31B"),
    "ab4_think1024": ("v10-think1024@kaggle", "31B"),
    "ab4_think2048_calls24": ("v10-calls24@kaggle", "31B"),
    "ab5_v19": ("v19@kaggle", "31B"),
    "ab5_hanh_v4": ("hanh_v4@kaggle", "31B"),
    "ab5_v10": ("v10@kaggle", "31B"),
    "ab6_v19": ("v19@kaggle", "31B"),
    "ab6_v19_issue_script": ("v19-issue_script@kaggle", "31B"),
    "small1_e4b_v19": ("e4b_v19@kaggle", "E4B"),
    "small1_e4b_v19_script": ("e4b_v19_script@kaggle", "E4B"),
    "small1_e4b_v10": ("e4b_v10@kaggle", "E4B"),
    "small2_e4b_v10": ("e4b_v10@kaggle", "E4B"),
    "small2_e4b_v10_verify": ("e4b_v10_verify@kaggle", "E4B"),
    "small2_e4b_v10_think8k": ("e4b_v10_think8k@kaggle", "E4B"),
    "small3_e4b_v10": ("e4b_v10@kaggle", "E4B"),
    "small3_e4b_bf": ("e4b_bf@vllm", "E4B"),
    "small3_e4b_bf_ep1": ("e4b_bf+lora-ep1@kaggle", "E4B"),
    "small3_e4b_bf_final": ("e4b_bf+lora-final@kaggle", "E4B"),
}
LOCAL_E4B = {  # results/local_e4b/<stem>.csv: vLLM W4A16 E4B on one RTX 3090
    "e4b_local1": "e4b_v10@local-vllm",
    "e4b_local2": "e4b_v10@local-vllm",
    "e4b_local3": "e4b_v10@local-vllm",
    "ab1_e4b_bf": "e4b_bf@vllm",
    "ab1_e4b_bf_locate": "e4b_bf_locate@local-vllm",
    "ab1_e4b_bf_grepfix": "e4b_bf_grepfix@local-vllm",
    "ab2_e4b_bf": "e4b_bf@vllm",
    "ab2_e4b_bf_seq_r1": "e4b_bf_seq@local-vllm",
    "ab2_e4b_bf_seq_r2": "e4b_bf_seq@local-vllm",
    "ab3_e4b_bf": "e4b_bf@vllm",
    "ab3_e4b_loc": "e4b_loc@local-vllm",
    "ab4_e4b_bf": "e4b_bf@vllm",
    "ab4_e4b_loc": "e4b_loc@local-vllm",
}
CONTROLS = {"eval_gold", "eval_empty"}
SHARD_OF = {"v6_redo14964": "v6_dev40_s3"}          # one task re-run after a Docker error
INFRASTRUCTURE = {"docker_error_local"}              # dropped, see CARD["dropped"]

PATH = re.compile(r"^results/(?:(scorer_stack|local_e4b)/)?([^/]+)\.csv$")


def attempts(src):
    for path, rows in src.csv(PATH.pattern):
        folder, stem = PATH.match(path).groups()
        if folder == "scorer_stack":
            if stem in CONTROLS:
                continue
            system, model = KAGGLE[stem]
        elif folder == "local_e4b":
            system, model = LOCAL_E4B[stem], "E4B"
        else:
            system, model = LOCAL[stem], "31B"
        run = f"{folder or 'local'}/{SHARD_OF.get(stem, stem)}"
        for row in rows:
            end = (row.get("end") or "").strip()
            if end in INFRASTRUCTURE:
                continue
            y = int(row["resolved"].strip() == "1")
            yield {
                "file": path,
                "system": system,
                "run": run,
                "task": row["instance_id"].strip(),
                "y": y,
                "model": model,
                "harness": "official",
                "err": bool(end),
                "status": end or ("resolved" if y else "failed"),
            }
