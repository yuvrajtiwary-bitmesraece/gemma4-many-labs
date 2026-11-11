"""Adapter for Finerium/axi-paper-release (W1 "What Does a Fixed Model Gain from the System Around It?" and
W2 "AXI-Fresh: Post-Cutoff Tasks for Measuring Developer Agents"; GitHub mirror of the Kaggle dataset
ghaisank/axi-paper-release).

Eligible data: the `run` rows of ledger-public.jsonl with mode "agent" and split "dev".  Every such row is one
execution of one agent bundle (`cand`, identified by `cand_sha`) on development-set tasks and carries a `per_task`
map {task id: 0/1} copied from that run's harness file (`results`: runs/<session>/<arm>/rep1/task_results.jsonl,
not released).  36 runs, 1,121 attempts, 71 distinct tasks.

Not yielded: gold-patch rows (mode "gold", controls), rows of the lab's own AXI-Fresh task set (split "fresh",
ids fresh_*), leaderboard / submission / gate / analysis rows.
"""
import re

REPO = "Finerium/axi-paper-release"

CARD = {
    "eligible": True,
    "licence": {"spdx": "CC-BY-4.0",
                "evidence": "README.md: 'Licences: data (this folder's csv, jsonl, md, txt, json) CC BY 4.0; code (code/, "
                            "analysis/*.py) Apache-2.0.'; LICENSE: 'Apache License Version 2.0, January 2004' (the file "
                            "GitHub detects). The records used here are in ledger-public.jsonl and e3-session-exit.csv "
                            "(data, CC BY 4.0)"},
    "models": ["31B", "12B", "E4B"],
    "harness": "official",
    "summary": ("Public ledger of a five-week campaign with the author's single-loop 'AXI' agents: 36 agent runs on the "
                "lab's 71-task development split of the public set (9 full runs), on a seeded 18-task draw of it "
                "(15 runs) or on smaller / partial / hinted subsets (12 runs), Gemma 4 31B except one 12B and four E4B "
                "runs, scored by the competition harness (swegemma 0.2.7, later 0.2.11) on Kaggle 4x L4 with four test "
                "packages added to the sandbox."),
    "evidence": {
        "model": ("W1-Best-Paper.pdf s1: 'We use Gemma 4 31B with a 32k-token context and no adapter'; W1-supplement.pdf "
                  "S2: 'the model google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2 (a 4-bit QAT checkpoint) served by "
                  "vLLM on four L4 GPUs'; ledger-public.jsonl run rows from r0013 on carry instrument.model "
                  "('gemma-4-31b-it-qat-w4a16-ct', 'gemma-4-12b-it-qat-w4a16-ct' in r0023, 'gemma-4-e4b-it-qat-w4a16-ct' "
                  "in r0037-r0039 and r0051); r0005-r0010 have no model field and are the 31B 'five single-loop "
                  "configurations of the patch pool' (W1 s3, W2 Table 2)"),
        "harness": ("ledger-public.jsonl instrument: 'harness': {'swegemma': '0.2.7' | '0.2.11', 'adk-submission': ..., "
                    "'vllm': '0.19.1'}, 'sandbox': 'subprocess', 'gpu_model': 'NVIDIA L4', 'gpu_count': 4, "
                    "'concurrency': 1; 'results': 'runs/<session>/<arm>/rep1/task_results.jsonl'; W1-supplement.pdf S2: "
                    "'The inner level consists of the competition's harness and wheelhouse ... Every run from r0044 on "
                    "used 0.2.11'"),
        "systems": ("ledger-public.jsonl: every run row names its agent bundle in 'cand' (axi-mini/<name>) with its hash "
                    "'cand_sha'; 'hyp' names the experiment arm, 'note' and 'tags' the conditions (hint tier, capture, "
                    "starter sandbox, partial)"),
        "replicates": ("r0039 note: 'the E4B control arm of e4b-04 (its second read on these ids; r0037 the first)'; r0037 "
                       "note: 'against the 31B control's reads on the same ids (k1-01-control 11)'; r0031 / r0033 notes: "
                       "'K5 control arm, the same 12 ids' / 'K6 control arm, the same 12 ids'; W1-Best-Paper.pdf s6: "
                       "'Each arm is a single run; two runs of one Budget-Fit bundle on 29 shared AXI-Fresh tasks resolved "
                       "25 and 22 (r0047, r0045; rollout versions differ)'; W2-New-Resource.pdf s5: 'run-to-run "
                       "variation: two fitted-budget runs resolved 25 and 22 on their 29 shared tasks'. All rows of one "
                       "system share 'cand' and 'cand_sha' (ref-budget-fit 6677052a..., e4b-control f1d606a6...). The "
                       "ledger's own 'replicate' field is 1 in every agent row (one execution per arm and session)"),
        "task_sets": ("W1-Best-Paper.pdf s1: 'The development tasks are 71 of the 105 public tasks whose reference fix "
                      "passes the competition's sandbox. A seeded split drew them and held out one third per repository "
                      "(34 tasks, read by no agent)'; r0018 note: '18 dev tasks drawn by random.Random(20261012)'; r0030 "
                      "note: '12 of gap-01's 18 ids ran before the 1.6 h wall cap'; r0041 note: 'the 23 dev ids outside "
                      "hint-01's hard 48 (agent prior 0.72)'; r0042 note: '6 hard dev ids never solved by any agent read'"),
    },
    "decisions": [
        "system = agent bundle ('cand' without the 'axi-mini/' prefix); rows with the same cand and cand_sha run under "
        "the same condition are runs of one system: ref-budget-fit (r0019, r0024 on the 18-task draw, r0049 on all 71), "
        "ref-budget-fit@partial (r0031, r0033) and e4b-control (r0037, r0039); every other system has one run",
        "the lab's runner script ('rollout_sha') differs between some runs of one bundle (r0019/r0024 42424fa8, "
        "r0031/r0033 02aa6b0c, r0049 91551fd1; r0037 e642d80a, r0039 08441e88); the lab itself counts such runs as 'two "
        "runs of one Budget-Fit bundle' while noting 'rollout versions differ', so the rollout version is not part of the "
        "system label",
        "conditions that change what is measured go into the label: '@partial' (run stopped by a wall cap: r0030-r0033, "
        "r0051), '@starter-sandbox' (r0034, r0035: sandbox without the four added packages, on tasks whose reference fix "
        "needs them), '@<hyp>' for the four hinted runs (r0036 cap-01-tier1 and r0040 hint-01-tier1: file-path hints; "
        "r0042 hint-02-tier3: gold hunk in the prompt; r0043 hint-03-tier3: gold diff verbatim), '@capture' (r0041)",
        "r0041 (cap-02-data: the control bundle with the lab's capture proxy on, no hint) is kept apart from "
        "ref-budget-fit as 'ref-budget-fit@capture' although its note says 'also a control reading on these ids': the "
        "execution path differs (proxy) and its 23 tasks were chosen from earlier outcomes",
        "outcome-selected task subsets (flag outcome_selected_task_subset): r0040 (the 'hard 48': 40 of them unsolved "
        "in all five early full runs), r0041 (the 23 other ids, all solved at least once before), r0042 and r0043 (ids "
        "'never solved by any agent read'); r0036 (2 hinted ids, selection not stated). The 18-task draw is seeded "
        "random and the 12-task runs are its first 12 ids before a wall cap, so they are not outcome-selected; the K7 "
        "ids (r0034, r0035) were chosen from reference-patch controls, not from agent outcomes",
        "run label = '<row_id>:<hyp>'",
        "model: instrument.model where present; r0005-r0010 (no model field) are 31B per the papers",
        "err: only the five early full runs have a per-attempt exit flag (e3-session-exit.csv: 'timeout' = session cut "
        "off by the cap -> err True, 'terminal' -> False); the other runs have no such field (err None)",
        "the 71 development tasks include fastapi_14851, which the lab reports as resolved by an empty patch in its "
        "sandbox; its rows are kept as recorded",
        "harness 'official': competition harness and subprocess sandbox on Kaggle; the lab added four wheels "
        "(inline_snapshot, asttokens, executing, dirty_equals) to the sandbox ('repaired instrument') - flagged "
        "modified_test_environment",
    ],
    "excluded": [
        "ledger-public.jsonl run rows with mode 'gold' (r0001-r0004, r0008 on public tasks; r0011-r0013 on AXI-Fresh): "
        "reference-patch controls (see 'controls')",
        "ledger-public.jsonl run rows with split 'fresh' (r0014-r0017, r0044-r0048, r0054) and p6-tasks.csv: the lab's "
        "own AXI-Fresh task set (ids fresh_*), not the competition's development set",
        "ledger-public.jsonl rows of kind 'lb' and 'submit' (public-leaderboard scores, hidden tasks), 'gate' (paired "
        "tests between two run rows), 'e3' and 'fam' (analyses)",
        "e3-verdicts.csv: 182 judged non-empty patches of runs r0005-r0007, r0009, r0010 with their hidden-test outcome; "
        "repeats attempts already yielded (0 disagreements) and adds in-task check verdicts",
        "e3-session-exit.csv: timeout / terminal per session of the same five runs; used only for the err flag",
        "public-task-audit.csv: 24 public tasks whose reference fix fails in the sandbox and one that an empty patch "
        "passes; summary of the gold rows",
        "public-trajectories.csv, analysis/e3-public/: Qwen3-Coder and Llama trajectories of public Nebius datasets on "
        "SWE-rebench tasks (other models, other tasks)",
        "analysis/parser-tax, trace-fail-01, e3-terminal-01, e3-combo-01, mde, statement-cues: trace statistics and "
        "aggregates over the runs already yielded; statements.md, hard-subset.txt: AXI-Fresh material",
    ],
    "dropped": {"not run / infrastructure": 0},
    "flags": ["tasks_restricted_to_gold_checked", "modified_test_environment", "partial_runs", "oracle_hints",
              "outcome_selected_task_subset", "runs_under_10_tasks", "no_raw_per_run_files"],
    "outcome_selected_runs": ["r0040:hint-01-tier1", "r0041:cap-02-data", "r0042:hint-02-tier3", "r0043:hint-03-tier3"],
    "sampling": ("temperature 0.2, top_p 0.95, thinking budget 4096, max_output_tokens 8192 stated for the early "
                 "single-loop agents (preregistration/P5-E3-H3-H6.md A1: 'the agent's sampling (temperature 0.2, top_p "
                 "0.95, thinking budget 4096 with thoughts included, max_output_tokens 8192)'); not stated for the later "
                 "bundles (agent configurations are withheld until 2026-12-02; cs-v1 uses 'the host's sampling "
                 "settings'); no seed is reported anywhere: a repeated run is the same bundle executed again in another "
                 "Kaggle session"),
    "leaderboard": [
        "source: ledger-public.jsonl rows of kind 'lb' (public_score per submission, joined by cand_sha) and "
        "W2-New-Resource.pdf Table 2; the public leaderboard scores 58 hidden tasks, so 0.15 = 9/58",
        "floor-v0 (host baseline, first prompt): public 0.06 (2026-10-04); development 19/71 (r0005)",
        "floor-v3 (baseline, prompt fixes): public 0.13 (2026-10-06); development 19/71 (r0010)",
        "floor-v5-8: public 0.15 (2026-10-12); development 10/18 (r0018)",
        "ref-budget-fit (public Budget-Fit configuration): public 0.15, 0.18, 0.15 (three submissions of the same "
        "bundle, 2026-10-14, 10-21, 10-24); development 31/71 (r0049), 11/18 and 10/18 (r0019, r0024)",
        "floor-v8: public 0.15 (2026-10-16); development 6/18 (r0020)",
        "ref-v2: public 0.13 (2026-10-17); development 9/18 (r0028)",
        "ctl-cfg: public 0.15 (2026-10-20); development 6/12 (r0030, partial run)",
        "e4b-control (Gemma 4 E4B): public 0.03 (2026-10-23); development 4/18 twice (r0037, r0039)",
        "cs-v1 (shared default stack): public 0.13 (2026-11-02); development 31/71 (r0050)",
        "bundles with leaderboard scores but no development run in the ledger: ctl-both-v2 0.15, 0.18, 0.15, 0.13 (four "
        "submissions of one bundle), ctl-edit 0.15, 0.15, floor-v5-8-spec-verify 0.15, 0.12, adapter-01 0.13, "
        "axi-mini-a4 (five-role pipeline) 0.12",
    ],
    "controls": [
        {"path": "ledger-public.jsonl", "row": "r0001 (hyp gold-subprocess)", "kind": "reference_patch",
         "environment": "competition harness swegemma 0.2.7, subprocess sandbox as shipped, Kaggle CPU kernel",
         "tasks": 129, "resolved": 75, "per_task_visible": 95, "resolved_visible": 49,
         "note": "34 holdout ids are redacted from per_task (per_task_redacted)"},
        {"path": "ledger-public.jsonl", "row": "r0002 (hyp gold-subprocess-repaired)", "kind": "reference_patch",
         "environment": ("competition harness swegemma 0.2.7, subprocess sandbox with four added wheels (inline_snapshot, "
                         "asttokens, executing, dirty_equals), Kaggle CPU kernel"),
         "tasks": 129, "resolved": 105, "per_task_visible": 95, "resolved_visible": 71,
         "note": "the 24 visible failures are listed with a diagnosis in public-task-audit.csv; the 34 redacted ids all pass"},
        {"path": "ledger-public.jsonl", "row": "r0003 (hyp gold-l4-probe01)", "kind": "reference_patch",
         "environment": "same harness, sandbox as shipped, Kaggle 4x L4 kernel", "tasks": 7, "resolved": 3},
        {"path": "ledger-public.jsonl", "row": "r0004 (hyp gold-l4-dev01)", "kind": "reference_patch",
         "environment": "same harness, sandbox with the four added wheels, Kaggle 4x L4 kernel", "tasks": 71, "resolved": 71},
        {"path": "ledger-public.jsonl", "row": "r0008 (hyp gold-l4-dev03)", "kind": "reference_patch",
         "environment": "same harness, sandbox with the four added wheels, Kaggle 4x L4 kernel", "tasks": 71, "resolved": 71},
    ],
    "reported_numbers": [
        {"what": "n and resolved of every agent run row on the development split (36 rows)",
         "value": "the row's own 'n' and 'resolved' fields", "where": "ledger-public.jsonl",
         "adapter": "identical for all 36 rows", "match": True},
        {"what": "development resolve counts per configuration",
         "value": "19 (r0005), 19 (r0010), 31 (r0049), 31 (r0050), 4 of 18 (r0037)", "where": "W2-New-Resource.pdf Table 2",
         "adapter": "19, 19, 31, 31, 4/18", "match": True},
        {"what": "two-stage split against control: resolved, change-only, control-only", "value": "12 against 31, b 0, c 19",
         "where": "W1-Best-Paper.pdf Table 2 (r0052/r0049)", "adapter": "12 vs 31, 0, 19", "match": True},
        {"what": "session-state notes against control", "value": "24 against 31, b 3, c 10",
         "where": "W1-Best-Paper.pdf Table 2 (r0053/r0049)", "adapter": "24 vs 31, 3, 10", "match": True},
        {"what": "shared default stack against control", "value": "31 against 31, b 4, c 4",
         "where": "W1-Best-Paper.pdf Table 2 (r0050/r0049)", "adapter": "31 vs 31, 4, 4", "match": True},
        {"what": "Budget-Fit against baseline with prompt fixes", "value": "31 of 71 (r0049) against 19 (r0010); 13 gained, 1 lost",
         "where": "W1-Best-Paper.pdf s4", "adapter": "31 vs 19; 13 gained, 1 lost", "match": True},
        {"what": "sessions of the five single-loop configurations that hit the cap / resolutions",
         "value": "205 of 355 timed out, 0 of them resolved; 90 resolutions", "where": "W1-Best-Paper.pdf s4",
         "adapter": "err=True 205 of 355, none with y=1; 90 resolved (19+16+15+21+19)", "match": True},
        {"what": "18-task subset: floor-v8", "value": "6 of 18 (r0020)", "where": "W2-New-Resource.pdf s5",
         "adapter": "6/18", "match": True},
        {"what": "reference fix on the 129 public tasks", "value": "24 fail with the four added packages (r0002), 54 without (r0001)",
         "where": "W2-New-Resource.pdf s1", "adapter": "not yielded; ledger rows give 105/129 and 75/129", "match": True},
    ],
}

MODEL = {"gemma-4-31b-it-qat-w4a16-ct": "31B", "gemma-4-12b-it-qat-w4a16-ct": "12B", "gemma-4-e4b-it-qat-w4a16-ct": "E4B"}
RESULTS = re.compile(r"^runs/([^/]+)/([^/]+)/rep\d+/task_results\.jsonl$")


def system_of(row):
    """Agent bundle plus the conditions the ledger records for this run."""
    label = row["cand"].split("/", 1)[-1]
    instrument = row.get("instrument") or {}
    tags = row.get("tags") or []
    if not (instrument.get("harness") or {}).get("extra_wheels"):
        label += "@starter-sandbox"      # r0034, r0035: 'STARTER sandbox (extra_wheels false)'
    if instrument.get("hint_tier"):
        label += "@" + row["hyp"]
    elif "capture" in tags:
        label += "@capture"
    if row.get("partial"):
        label += "@partial"
    return label


def attempts(src):
    exits = {}
    for _, rows in src.csv(r"^e3-session-exit\.csv$"):
        for row in rows:
            exits[(row["configuration"], row["task"])] = row["exit"]
    for path, rows in src.jsonl(r"^ledger-public\.jsonl$"):
        for row in rows:
            if row.get("kind") != "run" or row.get("mode") != "agent" or row.get("split") != "dev":
                continue
            model = MODEL[(row.get("instrument") or {}).get("model", "gemma-4-31b-it-qat-w4a16-ct")]
            system = system_of(row)
            run = f"{row['row_id']}:{row['hyp']}"
            m = RESULTS.match(row.get("results") or "")
            session = f"{m.group(1)}/{m.group(2)}" if m else None
            for task, value in row["per_task"].items():
                exit_state = exits.get((session, task))
                yield {
                    "file": path,
                    "system": system,
                    "run": run,
                    "task": task,
                    "y": int(bool(value)),
                    "model": model,
                    "harness": "official",
                    "err": None if exit_state is None else exit_state == "timeout",
                    "status": ("resolved" if value else "not resolved") + (f" ({exit_state})" if exit_state else ""),
                }
