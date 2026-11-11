"""Adapter for aridiosilva/Gemma4_Google_Deepmind_S1 ("Frozen paper evidence: supplemental edition")."""

REPO = "aridiosilva/Gemma4_Google_Deepmind_S1"

CARD = {
    "eligible": True,
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md "
                                                  "'(c) 2026 Aridio Silva - Apache-2.0', 'All new text, extracts "
                                                  "and verification code in this repository are author-generated "
                                                  "supporting material under Apache-2.0'"},
    "models": ["31B"],
    "harness": "official",
    "summary": "Public extracts of a private evidence archive for a paper on a controlled agent layer ('P' arm). "
               "Exactly one real Gemma 4 31B attempt on a dev task is recorded (rich_3953, not resolved); everything "
               "else is a synthetic-task pilot, reference-patch rehearsals with a scripted fake model, or logs.",
    "evidence": {
        "model": "evidence/ref-12.md: 'On November 6, Gemma 4 31B processed rich_3953 under P on Kaggle. This task "
                 "had already been exposed during development. There was no B arm.'",
        "harness": "not named for this run; inferred from evidence/source-catalog.json member "
                   "'tasks/rich_3953-bug33-restart02-P/official/traces/trace_rich_3953.json' and from the same "
                   "pipeline's pilot, evidence/ref-08.md: 'using Gemma 4 31B, vLLM 0.19.1 and swegemma 0.2.11'",
        "systems": "evidence/v12-real-worker.json: run_id 'rich_3953-bug33-restart02-P', integration_revision "
                   "'P33_V12_AGENT_LAYER_20261105' -> system 'V12-P'",
        "replicates": "none: one attempt; evidence/ref-12.md 'No automatic retry was performed'",
        "task_sets": "one dev task (rich_3953); evidence/ref-12.md: 'This real case does not establish "
                     "effectiveness on the 129-task benchmark'",
    },
    "decisions": [
        "the single evaluated real-model attempt is yielded as its own run; with one task it falls under the "
        "harmonizer's minimum run size, which is the harmonizer's decision, not the adapter's",
        "err = terminal_guard_reason is set (CONTEXT_EXHAUSTED: the context preflight stopped the session before "
        "model request 19)",
        "harness 'official' is an inference from file names (see evidence.harness); the worker extract itself only "
        "says evaluation_completed=true, test_exit_code=1, resolved=false",
        "the run name contains 'restart02' and the progress log shows positions '1/32' and '1/1'; the source does "
        "not publish outcomes of earlier starts, so nothing else can be yielded",
    ],
    "excluded": [
        "pilot-bp.csv, evidence/pilot-receipts.json: synthetic task 'B01-page-boundary' (not a competition task), "
        "arms B and P once each with Gemma 4 31B, both resolved (2/2 tests, same 214-byte patch)",
        "evidence/task16-verdict.json, task16-junit.json, task16-proof.md: reference patch of fastapi_14186 "
        "replayed by a mock server ('This was not a Gemma result'): test exit code 0 but resolved=false, because "
        "required tests are skipped under either Pydantic version",
        "evidence/v12-k2-local.json, v12r1-k2-local.json: scripted fake model server replaying reference patches, "
        "model_calls_real 0: 32 mock tasks started, 32 resolved, fastapi_14186 deferred",
        "evidence/v12-progress-events.jsonl (840 heartbeat records), v12-trace-metadata.json, "
        "v12-junit-identities.json, v12-test-summary.json: logs and test details of the one yielded attempt",
        "evidence/v12-unit-local.json, v12r1-unit-local.json, parser-*.json, w18-synthetic-receipt.json: local unit "
        "and parser checks, no model",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["single_attempt"],
    "reported_numbers": [
        {"what": "V12 real run, rich_3953 under P", "value": "resolved=false, test_exit_code=1, 1 failed / 55 passed",
         "where": "evidence/ref-12.md; evidence/v12-real-worker.json; evidence/v12-test-summary.json",
         "adapter": "1 attempt, y=0", "match": True},
    ],
}


def attempts(src):
    for path, obj in src.json(r"^evidence/v12-real-worker\.json$"):
        data = obj["data"]
        if data.get("status") != "EVALUATED" or not data.get("evaluation_completed"):
            continue
        guard = data.get("terminal_guard_reason")
        yield {
            "file": path,
            "system": "V12-P",
            "run": data["run_id"],
            "task": data["instance_id"],
            "y": int(bool(data["resolved"])),
            "model": "31B",
            "harness": "official",
            "err": bool(guard),
            "status": f"{data['status']}; {guard or 'no guard'}; test exit {data.get('test_exit_code')}",
        }
