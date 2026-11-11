"""Adapter for CVasilopoulos/gemma4-laptop-agent ("What the context window forgets: a Gemma 4 coding agent on a laptop chip").

Not eligible: the per-task results are in a Kaggle dataset, not in this repository.
"""

REPO = "CVasilopoulos/gemma4-laptop-agent"

PAPER = "paper/paper.md"

CARD = {
    "eligible": False,
    "reason": "per-task outcomes are not in the repository (README.md: 'The agent traces and result files live in the Kaggle "
              "dataset `chrisvas123/gemma4-laptop-agent-traces`. Raw results, the 42 GB of competition data and the models are "
              "not in this repository.'); the only data file, paper/fig_timeline_fastapi_14372.csv (603 rows), holds prompt-"
              "token counts per turn for two runs of one task; the paper gives aggregate tables only",
    "licence": {"spdx": "Apache-2.0", "evidence": "LICENSE: 'Apache License Version 2.0, January 2004'; README.md: 'The code, "
                                                  "paper and pre-registration are released under the Apache-2.0 licence (see "
                                                  "`LICENSE`).'"},
    "models": [],
    "harness": None,
    "summary": "Gemma 4 E2B, E4B and 26B-A4B (QAT Q4_0 GGUF, llama.cpp Vulkan on a Ryzen AI 9 HX 370) on a stratified sample "
               "of 60 of the 125 usable dev tasks with the lab's own Docker grader; context-policy arms (adk, cached, adk_task, "
               "adk_calls) and thinking arms; a pre-registered paired test of cached against adk on 50 tasks.",
    "evidence": {
        "model": PAPER + " 2.1: 'Google's official QAT Q4_0 GGUF files for Gemma 4 E2B (3.1 GiB), E4B (4.8 GiB) and 26B-A4B "
                         "(13.4 GiB)'",
        "harness": PAPER + " abstract: 'We scored them with a local copy of the grader: its reference fixes pass on 127 of the "
                           "129, and an empty patch passes on 2.' (it would be 'local')",
        "systems": PAPER + " Tables 2-5: model x context policy (adk, cached, adk_task, adk_calls) x thinking",
        "replicates": PAPER + " section 8: 'One machine and one seed.'; Table 4: 'one run per cell'",
        "task_sets": PAPER + " 2.4: 'we drew a stratified random sample of 60 of the 125 usable tasks (seed 20261101). The "
                             "first 10 are the exploratory set (E) ... The other 50 are the confirmatory set (C).'",
    },
    "decisions": [
        "nothing is yielded: the repository gives counts per arm but no per-task rows, and runs are not invented from counts",
        "three attempts can be read from the prose (fastapi_14372 solved under cached and not under adk; one thinking-arm solve "
        "of rich_2725) but they are examples, not a results table, and are not turned into records",
    ],
    "excluded": [
        "paper/fig_timeline_fastapi_14372.csv: plotted data of Figure 1 (policy, turn, prompt_tokens, compaction, repeat run) "
        "for one exploratory task; no outcome column",
        PAPER + " Table 2 (10 exploratory tasks, 26B-A4B, thinking off): adk 1/10, cached 5/10 - aggregate",
        PAPER + " Table 3 (50 confirmatory tasks, 26B-A4B, thinking off): adk 3/50, cached 10/50; 7 solved only under cached, "
                "0 only under adk, 3 under both; exact McNemar p = 0.016 - aggregate",
        PAPER + " Table 4 (post-hoc ablation, 13 confirmatory tasks, one run per cell): adk 3, adk_task 2, adk_calls 5, cached "
                "10 of 13 - aggregate",
        PAPER + " Table 5 (50 confirmatory tasks, thinking off): E4B adk 5/50, E4B cached 5/50, E2B adk 1/50 - aggregate",
        PAPER + " sections 5 and 8: thinking budget 4,096 on 2 exploratory tasks, both unsolved; post-hoc 1,024-token arm on 20 "
                "confirmatory tasks 'solved 6 against 3 with thinking off' - aggregate",
        "controls: " + PAPER + " 2.2 'Reference patches now pass on 127 of 129 tasks, and the empty patch passes on 2' (first "
        "grader copy: reference patch 58 of 129); paper/preregistration_2026-11-01.md: grader v2 'gold went from 58 to "
        "**97/129 (0.752)**', 'The empty-patch rate got worse under v2:** 5/129 (0.039), up from 2/129.' (new passes "
        "requests_7315, requests_7427, requests_7309, fastapi_14851, fastapi_14077) "
        "- no per-task rows in the repository",
        PAPER + " section 9: 'all traces except one discarded after an HTTP 500' - one attempt the lab dropped",
    ],
    "dropped": {},
    "flags": ["aggregated_only"],
    "reported_numbers": [],
}


def attempts(src):
    return
    yield
