"""Adapter for gautamhardik/mr-ast ("MR-AST: Multi-Resolution AST Context Selection ...").

Not eligible: the repository is a context-selection study (retrieval recall under
token budgets).  It contains no agent run with a verified resolved / not-resolved
outcome, so nothing is yielded.
"""

REPO = "gautamhardik/mr-ast"

CARD = {
    "eligible": False,
    "reason": (
        "no per-task resolved/not-resolved outcome: ~21,800 records are context-selection metrics "
        "(gold-symbol recall, hit@k, syntactic validity, token budgets); the Gemma 4 repair experiment was not "
        "executed (all 24 rows 'HARDWARE_CONSTRAINED_NO_L4_GPU'), and the only live Gemma 4 calls (18 single-shot "
        "diffs, 6 tasks x 3 context methods) were checked with `git apply` only (0/18 applied), never tested"
    ),
    "licence": {
        "spdx": "Apache-2.0",
        "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md s8: 'Project Source Code: "
                    "Distributed under the [Apache License 2.0](LICENSE).'",
    },
    "models": [],
    "harness": None,
    "summary": (
        "Offline benchmark of repository-context compilers (BM25 raw/scoped, Aider RepoMap, MR-TopoS/MR-AST MILP) on "
        "the 129 dev tasks, scored by gold-file / gold-symbol recall under 2k-16k token budgets with the Gemma 4 "
        "tokenizer. No agent was run through the competition harness."
    ),
    "evidence": {
        "model": "research/gemma_pilot_model_identity.md s3: 'Inference Provider: OpenRouter API ... Model Identifier: "
                 "google/gemma-4-31b-it' (pilot only; 18 single-shot calls, no agent loop)",
        "harness": "research/critical_issues.md Issue 6: 'Gemma 4 Repair Inference Was NOT EXECUTED ... zero LLM "
                   "inference was executed'; results/gemma_repair/per_task.csv: model_execution_status = "
                   "'HARDWARE_CONSTRAINED_NO_L4_GPU' in 24/24 rows",
        "systems": "n/a (nothing yielded)",
        "replicates": "n/a (nothing yielded)",
        "task_sets": "results/dataset_audit/audit_summary.md: 129 tasks (fastapi 67, rich 48, requests 13, httpx 1); "
                     "pilot_execution.csv: 6 tasks; gemma_repair/per_task.csv: 8 tasks",
        "no_outcome": "README.md s6: 'end-to-end bug resolution (SWE-bench pass@1) has not been demonstrated'; "
                      "research/gemma_pilot_model_identity.md s4.1: 'must be formally classified as a "
                      "generation-latency and prompt-efficiency pilot, NOT a validated SWE repair benchmark ... "
                      "Clean `git apply` Rate: 0.0% (0/18 trials applied cleanly)'",
    },
    "decisions": [
        "pilot_execution.csv rows are not yielded as failures: the only outcome is a patch-applies flag "
        "(applied_cleanly), tests were never run on the model patches, and the lab itself says it is not a repair "
        "benchmark; the README rule excludes patch-applies flags",
        "the 18 pilot calls are single-shot diff generations through a cloud API (no agent loop, no harness); each "
        "method covers 6 tasks only",
        "syntactic_validity / is_budget_compliant / gold_recall columns describe the compiled context, not an agent "
        "attempt, and are ignored",
    ],
    "excluded": [
        "results/strategic_reset/experiment_ledger.csv + .json (5,160 rows = 129 tasks x 4 budgets x 10 conditions; "
        "the .json is the same ledger): context-selection metrics (gold_recall_all, retained symbols, "
        "syntactic_validity, serialized_tokens), no agent run",
        "results/final_benchmark/per_task.csv (2,580 rows = 129 x 4 budgets x 5 methods), "
        "results/final_benchmark/budget_audit.csv, results/phase3/per_task_results.csv (2,064 rows), "
        "results/phase3/budget_utilization_audit.csv, results/parent_reconstruction_cost_audit.csv: retrieval "
        "recall / hit@k / token-budget metrics, not outcomes",
        "results/retrieval_benchmark/lexical_per_task.csv, results/ugrace_benchmark/ugrace_per_task.csv, "
        "results/*heldout*.csv, results/dense_hybrid_benchmark.csv, results/phase3/structural_baseline_comparison.csv: "
        "localisation ranks / recall, not outcomes",
        "results/gemma_repair/per_task.csv == results/phase4/repair_per_task_results.csv (identical files, 24 rows = "
        "8 tasks x 3 methods): model not run ('HARDWARE_CONSTRAINED_NO_L4_GPU'); only reports that test patch and "
        "gold patch apply (TEST_PATCH_APPLIED / GOLD_PATCH_APPLIED 24/24) - a gold-patch application check, no "
        "test outcome",
        "results/gemma_repair/pilot_execution.csv (18 rows = 6 dev tasks x bm25_raw/bm25_scoped/mrtopos_milp at "
        "B=2000): single-shot diffs from google/gemma-4-31b-it via OpenRouter; has_diff 18/18, applied_cleanly "
        "0/18; patch-applies flag only, no tests run",
        "getting-started-gemma-4-developer-agent.ipynb: stock competition starter notebook with a 2-task demo of "
        "the sample submission (gemma-4-12b-it-qat-w4a16-ct): fastapi_15661 and fastapi_15588 both "
        "'resolved=False, exit_code=-1' after agent timeouts; demo output in notebook stdout, not a lab experiment",
        "results/dataset_audit/*, results/phase1/*, results/phase2/*, results/validation/*, "
        "results/*audit*.json, results/statistics/*: dataset / tokenizer / solver audits without outcomes",
    ],
    "dropped": {},
    "flags": [],
    "reported_numbers": [
        {"what": "Gemma 4 pilot: diffs that applied cleanly", "value": "0/18",
         "where": "README.md s6; research/gemma_pilot_model_identity.md s4.1",
         "adapter": "not yielded; results/gemma_repair/pilot_execution.csv has applied_cleanly=False in 18/18 rows",
         "match": True},
        {"what": "repair rows with model executed", "value": "0/24 (NOT EXECUTED)",
         "where": "research/critical_issues.md Issue 6",
         "adapter": "not yielded; results/gemma_repair/per_task.csv has HARDWARE_CONSTRAINED_NO_L4_GPU in 24/24 rows",
         "match": True},
    ],
}


def attempts(src):
    return
    yield
