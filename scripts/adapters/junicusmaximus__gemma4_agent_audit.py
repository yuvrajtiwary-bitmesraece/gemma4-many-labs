"""Adapter for junicusmaximus/gemma4-agent-audit ("Different Collapse Outcomes After Finetuning on Successful
Trajectories").

The package holds no raw harness files (task_results.jsonl is not released); the per-task resolved flags of
every run are embedded in the sealed analysis reports:

    analyses/v02-v025/report.json        per_task[task][arm][seed]["resolved"]  arms A0, A1, A2
    analyses/<two-arm analysis>/report.json
        populations.all48_sensitivity.primary.per_task = [{instance_id, <slot>: [s1, s2, s3], ...}]

16 arms (Table 1 of the paper) x 3 decoding seeds x the 48 Rich tasks of the development set = 48 runs,
2,304 attempts.  A comparator slot is repeated in several reports (v043_M0 in four, v055_M0qmc in five,
v066_A0mc in four); every arm is read from ONE report (table ARMS below); the copies were checked to be
identical flag by flag.  The list position of a flag is the decoding seed: the lab's analyzers build
`[[runs[arm, s][task] for s in (1, 2, 3)] for task in ids]` (code/tools/analyze_mask_control_v042.py).
"""
import re

REPO = "junicusmaximus/gemma4-agent-audit"

CARD = {
    "eligible": True,
    "licence": {"spdx": "CC-BY-4.0",
                "evidence": "README.md '## Licence and scope': 'Own documents and figures and tables are released under "
                            "Creative Commons Attribution 4.0 International (`LICENSE-CC-BY-4.0`). This covers the reports "
                            "and per-task tables and contracts and pre-registrations and figures'; 'Own code is released "
                            "under the Apache License 2.0 (`LICENSE`). This covers `code/` and `verify/` and the plotting "
                            "scripts in `figures/`'; NOTICE.md repeats both. The records used here come from the reports "
                            "(CC-BY-4.0); GitHub detects the repository as Apache-2.0 from LICENSE"},
    "models": ["E4B"],
    "harness": "official",
    "summary": ("Controlled fine-tuning study of Gemma 4 E4B (public 'Budget-Fit' agent, 4 min / 28 calls / 80 turns): "
                "the untuned QAT base in three serving conditions and 13 LoRA arms (teacher trajectories, masked targets, "
                "QAT-base retraining, history pack, self-generated trajectories, second training seeds), each run with "
                "decoding seeds 1-3 on all 48 Rich tasks of the development set with the swegemma harness on one local "
                "RTX 5090; 2,304 attempts, most trained arms resolve nothing."),
    "evidence": {
        "model": ("README.md subtitle: 'A controlled study of Gemma 4 E4B: self generated versus teacher trajectories at "
                  "equal updates with QAT and serving format audits'; report.json slot_runs: 'adapter_alias': "
                  "'gemma-4-e4b-it-qat-w4a16-ct', 'adapter_source': null for the base arms and "
                  "'PROJECT_ROOT/gemma4/sft/runs/A1_...' for the LoRA arms; "
                  "preregistrations/contracts/paper-experiment-contract-v0.2.md: 'A0: supported E4B w4a16 base, no adapter'"),
        "harness": ("paper-experiment-contract-v0.2.md: 'Let y[a,i,s] be the harness resolution indicator for arm a, task i "
                    "and decoding seed s'; 'Effective CLI budgets override the stored eight-minute YAML: max_time_minutes4, "
                    "max_tool_calls28, max_turns80, timeout_seconds240, verification timeout300, subprocess sandbox, "
                    "concurrency4'; 'These are local single-5090 measurements'; NOTICE.md: 'Harness packages: `swegemma` ... "
                    "and `adk_submission` and `adk_eval_core`'; code/tools/analyze_contract_v02_v025.py reads "
                    "'results/a0_e4b_s{seed}/task_results.jsonl'; code/tools/run_mask_evaluations_v043.py pins "
                    "'swegemma/evaluate.py', 'swegemma/harness/verification.py', 'swegemma/sandbox/subprocess.py'"),
        "systems": ("verify/verify_paper_numbers.py TABLE1 lists the 16 rows of the paper's Table 1 with their analysis folder "
                    "and arm key: A0, A1, A2 (v02-v025), M0, M1 (mask-control-v043), M0mc (v051), A0cap (v052), M0q (v054), "
                    "M0qmc (v055), Hq1mc (v058), M1qmc (v062), M1qmc-s2 (v067), A0mc (v066 / v064a), A_onpol (v064a), "
                    "T_sub (v064t), A_onpol-s2 (v064c); report.json 'slots' / 'generation_runs' describe each slot"),
        "replicates": ("paper-experiment-contract-v0.2.md: 'Same existing prompt and seed-specific configuration in all arms; "
                       "only adapter assignment varies. Decoding seeds1,2,3; temperature0.2, top_p0.95'; run names "
                       "<slot>_s1, _s2, _s3 with 'decoding_seed': 1, 2, 3 in report.json slot_runs / generation_runs and "
                       "'allowed_config_changes': 'sampling seed is the paired seed above, temperature remains 0.2; all other "
                       "sampling bytes unchanged'; analyses/matched-context-v064a/report.md: '120 rollouts = 40 tasks x 3 "
                       "decoding seeds'. Training seeds are NOT replicates for the lab: analyses/matched-context-v067/"
                       "report.md 'M1q (v062, seed 1) and M1q-s2 are reported side by side, never pooled'"),
        "task_sets": ("paper-experiment-contract-v0.2.md: 'Primary population: the following40 Rich tasks with "
                      "gold_resolved=True and empty_resolved=False in the hashed controls CSV'; 'All48 original Rich tasks "
                      "remain evaluated and reported as a secondary population. The eight primary exclusions are rich_3064, "
                      "rich_3130, rich_3296, rich_3469, rich_3486, rich_3782, rich_4070, rich_4079'"),
    },
    "decisions": [
        "system = the paper's Table 1 row (arm); LoRA arms carry the suffix '@lora' (README rule: fine-tuned condition in "
        "the label); the three untuned rows A0 (historical uncapped sandbox, default chat template), A0cap (4 GiB "
        "per-sandbox memory cap, LoRA-off server) and A0mc (cap + the lab's 'matched' chat template) are three systems, as "
        "the lab reports them as separate rows and says 'A0mc vs A0cap is a template difference across separate runs'",
        "the two training seeds of one recipe (M1qmc / M1qmc-s2, A_onpol / A_onpol-s2) are different adapters and stay "
        "different systems; the lab never pools them",
        "run = the lab's run name (<generation>_<arm>_s<decoding seed>); the three decoding seeds of an arm are replicates",
        "all 48 Rich tasks are yielded per run (the lab's 'all48' / 'secondary48' population), not only the lab's "
        "control-qualified primary40: the eight other tasks were run and reported; none of them is ever resolved "
        "(their reference patch fails in the lab's sandbox)",
        "every arm is read from one report; A0mc is read from analyses/matched-context-v064a/report.json because the "
        "generation's own report (matched-context-v066) prints only counts per seed (4, 3, 5 - equal to the flags used)",
        "outcomes are harness rows copied into analysis reports, not raw harness files: flagged no_raw_per_run_files",
        "err: analyses/v02-v025/report.json keeps the harness 'error' string per attempt (A0, A1, A2); the two-arm reports "
        "keep only the resolved flag, so err is None there",
        "attempts ending with 'Sandbox execution error: litellm.ContextWindowExceededError' (A1 9, A2 14) are kept as "
        "failures: the lab counts them unresolved ('False+error is unresolved') and they are model-context overflows, not "
        "sandbox faults",
        "A_onpol / A_onpol-s2 were trained on the base model's own resolved rollouts on 57 fastapi tasks of the public "
        "set; they are evaluated only on Rich tasks ('The 48 Rich holdout tasks are never generated on and never used for "
        "data selection'), so no evaluated task was trained on",
        "harness 'official': swegemma run locally (subprocess sandbox, concurrency 4, one RTX 5090) with a 4-minute "
        "session limit instead of the default 8",
    ],
    "excluded": [
        "analyses/*/per_task.jsonl and analyses/mechanism-v031/per_task_{all48,legacy48,primary40}.jsonl: trace metrics "
        "per run and task (loop, submit_called, opener, think rate); no resolved flag",
        "legacy runs a1_e4b_s1-3 and a2_e4b_s1-3 (analyses/mechanism-v031/per_task_legacy48.jsonl) and the v021-v024 "
        "retries: declared 'sealed diagnostic artifacts' by paper-experiment-contract-v0.2.1-retry.md, outcomes not released",
        "analyses/probes-v032/report.md: four serving probes (A1 alpha 8, A2 alpha 8, A1 temperature 0.7, A0 temperature "
        "0.7; one decoding seed, 48 tasks each): only paired differences are printed, report.json is not in the package "
        "(aggregated only)",
        "on-policy generation v064_gen_A0_s1..s10 (analyses/onpolicy-v064/migration/scale/evaluation/completed.json, "
        "contracts/onpolicy-gen-v064r2-runtime-PINNED.json): ten seeded runs of the untuned base on 57 control-eligible "
        "fastapi tasks; per-task outcomes are not released (only: 30 trajectories kept from 13 tasks, cap 3 per task)",
        "analyses/external-e4b-v068/per_task.jsonl (144 rows) and analyses/codex-v070-git-r2/readout/per_task.jsonl "
        "(108 rows): 12 tasks of external repositories (mpmath, pypika, attrs, jinja, ...), not the development set",
        "analyses/codex-v070-git-r2/controls-evidence/control-report.json: reference/empty controls of the external panel",
        "analyses/oom-controls-v043/final48/report.json: reference-patch and empty-patch controls on the 48 Rich tasks "
        "(see 'controls'); contracts/onpolicy-gen-v064r2-runtime-PINNED.json: control eligibility of 69 non-Rich tasks",
        "analyses/teacher-behaviour-boss-v045/per_trajectory.jsonl: 1,670 teacher trajectories of a third-party LoRA kit "
        "(Qwen3-Coder-480B and Gemma 4 31B) on kit tasks, all pre-selected as resolved; not runs of this lab",
        "analyses/first-step-probe-boss-v048, qat-transfer-v053, serialization-audit, sft-audit-v033, compaction-audit, "
        "repro-definition-audit, mechanism-ci-v031, m1q-vs-a0mc-v067, onpolicy-cells-v064, figures/figure_data.csv: "
        "token-level probes, training-data audits and aggregates over the runs already yielded",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["fine_tuned", "no_raw_per_run_files", "rich_tasks_only", "partly_aggregated_only",
              "reduced_time_budget_4min", "local_single_gpu"],
    "sampling": ("temperature 0.2, top_p 0.95, max_output_tokens 8192, thinking_budget 4096; decoding (sampling) seeds 1, 2, "
                 "3 = the three runs of every arm, same seeds in all arms "
                 "(preregistrations/contracts/paper-experiment-contract-v0.2.md 'Fixed local evaluation configuration'; "
                 "report.json slot_runs / generation_runs: 'temperature': 0.2, 'decoding_seed': 1..3)"),
    "leaderboard": ("none (paper-experiment-contract-v0.2.md: 'These are local single-5090 measurements. They do not "
                    "demonstrate equivalence to the sequential four-L4 scorer or a safe full-run leaderboard budget')"),
    "controls": [
        {"path": "analyses/oom-controls-v043/final48/report.json", "kind": "reference_patch",
         "environment": ("local swegemma harness, subprocess sandbox under the lab's v043 memory guard (4 GiB per sandbox), "
                         "one workstation"),
         "tasks": 48, "resolved": 40,
         "how": "primary40.ids = the 40 tasks that pass ('all_gold_pass': true); original8.rows = the 8 that fail"},
        {"path": "analyses/oom-controls-v043/final48/report.json", "kind": "empty_patch",
         "environment": ("local swegemma harness, subprocess sandbox under the lab's v043 memory guard (4 GiB per sandbox), "
                         "one workstation"),
         "tasks": 48, "resolved": 0,
         "how": "all48.empty_resolved_count = 0; exit codes of the four non-assertion cases in pathological_controls"},
        {"path": "contracts/onpolicy-gen-v064r2-runtime-PINNED.json", "kind": "reference_patch_and_empty_patch_joint",
         "environment": "same local harness, capped sandboxes (Stage 0 of prereg v064-r2)",
         "tasks": 69, "resolved": 57,
         "how": ("candidate_ids (61 fastapi, 7 requests, 1 httpx) versus eligible.task_ids = 'gold resolved and empty not "
                 "resolved and no error and no OOM kill' (57, all fastapi). The 12 others are named in "
                 "preregistrations/notes/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md from the lab's earlier "
                 "uncapped controls: 8 gold-fail (fastapi_14962, fastapi_14964, fastapi_15763, fastapi_15785, httpx_3672, "
                 "requests_6644, requests_7205, requests_7505) and 4 empty-pass (requests_7309, requests_7315, "
                 "requests_7328, requests_7427)")},
    ],
    "reported_numbers": [
        {"what": "Table 1 repair counts on primary40 per decoding seed (16 rows)",
         "value": ("A0 3,2,2; A1 0,0,0; A2 0,0,0; M0 0,0,0; M1 0,0,0; M0mc 0,0,0; A0cap 5,5,5; M0q 0,0,0; M0qmc 0,0,0; "
                   "Hq1mc 0,0,0; M1qmc 4,2,4; M1qmc-s2 3,2,1; A0mc 4,3,5; A_onpol 5,3,2; T_sub 0,0,0; A_onpol-s2 3,2,4"),
         "where": "verify/verify_paper_numbers.py TABLE1 (values 'printed in the paper'); report.md of each analysis",
         "adapter": "identical for all 16 rows when restricted to the 40 primary ids", "match": True},
        {"what": "all-48 resolved per run", "value": "v066_A0mc_s1 4/48, s2 3/48, s3 5/48; A0cap [5, 5, 5]; M1qmc [4, 2, 4]",
         "where": "analyses/matched-context-v066/report.md; a0-capped-v052/report.md; matched-context-v062/report.md",
         "adapter": "4, 3, 5; 5, 5, 5; 4, 2, 4 of 48", "match": True},
        {"what": "A0 resolution rate per seed on all 48 tasks", "value": "0.0625, 0.0417, 0.0417",
         "where": "analyses/v02-v025/report.json populations.secondary48.arms.A0.seed_resolution_rates",
         "adapter": "3/48, 2/48, 2/48", "match": True},
        {"what": "pooled resolved on all 48 tasks", "value": "A_onpol 10/144, A0mc 12/144, T_sub 0/144",
         "where": "analyses/matched-context-v064a/report.md, matched-context-v064t/report.md",
         "adapter": "10, 12, 0 of 144", "match": True},
        {"what": "attempts with a harness error on all 48 tasks (A0 / A1 / A2)", "value": "0 / 136 / 140",
         "where": "analyses/v02-v025/report.json populations.secondary48.arms.<arm>.task_error_count",
         "adapter": "err=True rows: 0 / 136 / 140", "match": True},
        {"what": "controls on the 48 Rich tasks", "value": "reference patch 40/48, empty patch 0/48",
         "where": "analyses/oom-controls-v043/final48/report.json", "adapter": "not yielded (controls)", "match": True},
    ],
}

# report folder -> [(key of the arm in that report, system label, run-name prefix)]
ARMS = {
    "mask-control-v043": [("M0", "M0@lora", "v043_M0"), ("M1", "M1@lora", "v043_M1")],
    "matched-context-v051": [("v051_M0mc", "M0mc@lora", "v051_M0mc")],
    "a0-capped-v052": [("A0cap", "A0cap", "v052_A0cap")],
    "m0q-eval-v054": [("M0q", "M0q@lora", "v054_M0q")],
    "matched-context-v055": [("v055_M0qmc", "M0qmc@lora", "v055_M0qmc")],
    "matched-context-v058": [("v058_Hq1mc", "Hq1mc@lora", "v058_Hq1mc")],
    "matched-context-v062": [("v062_M1qmc", "M1qmc@lora", "v062_M1qmc")],
    "matched-context-v067": [("v067_M1qs2mc", "M1qmc-s2@lora", "v067_M1qs2mc")],
    "matched-context-v064a": [("v066_A0mc", "A0mc", "v066_A0mc"), ("v064_A_onpol", "A_onpol@lora", "v064_A_onpol")],
    "matched-context-v064t": [("v064_T_sub", "T_sub@lora", "v064_T_sub")],
    "matched-context-v064c": [("v064_A_onpol_c2", "A_onpol-s2@lora", "v064_A_onpol_c2")],
}
# analyses/v02-v025/report.json: arm -> (system label, run-name prefix)
V025 = {"A0": ("A0", "a0_e4b"), "A1": ("A1@lora", "v025_a1_e4b"), "A2": ("A2@lora", "v025_a2_e4b")}
SEEDS = (1, 2, 3)
REPORT = re.compile(r"^analyses/([^/]+)/report\.json$")


def attempts(src):
    wanted = "|".join(re.escape(name) for name in list(ARMS) + ["v02-v025"])
    for path, rep in src.json(rf"^analyses/({wanted})/report\.json$"):
        folder = REPORT.match(path).group(1)
        if folder == "v02-v025":
            for task, by_arm in rep["per_task"].items():
                for arm, (system, prefix) in V025.items():
                    for seed in SEEDS:
                        row = by_arm[arm][str(seed)]
                        resolved = bool(row["resolved"])
                        error = row.get("error")
                        yield {
                            "file": path,
                            "system": system,
                            "run": f"{prefix}_s{seed}",
                            "task": task,
                            "y": int(resolved),
                            "model": "E4B",
                            "harness": "official",
                            "err": bool(error),
                            "status": error or ("resolved" if resolved else "unresolved"),
                        }
            continue
        for row in rep["populations"]["all48_sensitivity"]["primary"]["per_task"]:
            for key, system, prefix in ARMS[folder]:
                for seed, resolved in zip(SEEDS, row[key]):
                    yield {
                        "file": path,
                        "system": system,
                        "run": f"{prefix}_s{seed}",
                        "task": row["instance_id"],
                        "y": int(bool(resolved)),
                        "model": "E4B",
                        "harness": "official",
                        "err": None,
                        "status": "resolved" if resolved else "unresolved",
                    }
