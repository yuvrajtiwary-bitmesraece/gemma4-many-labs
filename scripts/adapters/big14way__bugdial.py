"""Adapter for big14way/bugdial ("How Far Is the Bug? Symptom-to-Cause Distance and Code Graphs in Small Agents"). Not eligible."""

REPO = "big14way/bugdial"

CARD = {
    "eligible": False,
    "reason": ("localisation benchmark without resolved/not-resolved outcomes: the Gemma 4 E4B/E2B agents only name up to three "
               "suspect functions and are scored by the rank of the faulty function (no patch is written, no test is run). The "
               "field `resolved` of the episode files is the list of graph ids the submitted names resolve to, not a test "
               "outcome; 8,568 of the 10,220 episodes are on synthetic one-token mutations, the other 1,652 are localisation "
               "episodes on 118 public tasks"),
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md '## License': 'Apache-2.0'; release/dataset/README.md: 'Annotations and results: CC BY 4.0.'"},
    "models": [],
    "harness": None,
    "summary": ("Controlled bug-localisation benchmark: 454 synthetic tasks (one-token bugs planted in rich, requests, fastapi at "
                "fixed commits) plus derived labels for the 129 public tasks; Gemma 4 E4B and E2B localisation agents with seven "
                "tool/context arms, scored by top-k rank of the faulty function."),
    "evidence": {
        "model": "release/dataset/README.md: '`episodes_gemma-4-{e4b,e2b}-it-qat-w4a16-ct.jsonl` | 5,110 each | Gemma 4 agent episodes (7 arms; synthetic tasks x clue off/on, public tasks)'",
        "harness": ("bugdial/agent.py docstring: 'A small localization agent for Gemma 4 served by vLLM ... then must call `submit` "
                    "with up to 3 fully qualified function names, best first'; own driver bugdial/evalrun.py, no competition harness"),
        "systems": "release/dataset/README.md: '`arm` (`text`, `graph`, `callee`, `ctx_callee`, `ctx_short_bm25`, `ctx_short_undir`, `ctx_short_dir`), `variant` (`issue` / `issue_clue`)'",
        "replicates": "none: one episode per (task, arm, variant, model)",
        "task_sets": ("release/dataset/README.md: '`{rich,requests,fastapi}_tasks.jsonl` | 282 / 82 / 90 | synthetic tasks (one per "
                      "buggy function)'; 'For the competition's public tasks we publish only derived labels'"),
    },
    "decisions": [
        "release/dataset/README.md defines `rank` as '1-based position of the faulty function in the agent's up-to-3 answers; null "
        "if absent' and bugdial/evalrun.py computes it from `ep.resolved` (resolved names); neither is a verified fix, so no "
        "episode is converted into y",
    ],
    "excluded": [
        "release/dataset/episodes_gemma-4-e4b-it-qat-w4a16-ct.jsonl, episodes_gemma-4-e2b-it-qat-w4a16-ct.jsonl, kind=real "
        "(826 rows each: 118 public tasks x 7 arms): localisation rank only (E4B top-1: text 0.254, ctx_short_bm25 0.441; README: "
        "'a BM25 shortlist in the prompt lifts top-1 on the public tasks from 0.21 to 0.40')",
        "the same files, kind=synth (4,284 rows each): synthetic planted-bug tasks and localisation rank only",
        "release/dataset/{rich,requests,fastapi}_tasks.jsonl: 454 synthetic mutated tasks (task definitions)",
        "release/dataset/{rich,requests,fastapi}_locate.jsonl: model-free retrieval ranks (BM25, embeddings, graph expansion)",
        "release/dataset/real_tasks.jsonl: derived labels of the 129 public tasks (gold function ids, distance d, method ranks); no outcomes",
        "release/dataset/graph_stats.jsonl: edge audit of the released code graphs (every `calls` edge is reciprocal); no tasks",
    ],
    "dropped": {},
    "flags": ["localisation_only", "synthetic_tasks"],
    "reported_numbers": [],
}


def attempts(src):
    return
    yield
