"""Broader search for sources that the six registered queries missed.

The registered source frame (ANALYSIS_PLAN.md section 3) is six repository
searches. This script casts a wider net, after the results were known, to see
whether the conclusions hold on more sources: further repository searches, and
GitHub code search for development-set task identifiers and for strings that
only participants' repositories contain. Sources found here form the
"extended" frame and never enter the registered analysis.

Requires the GitHub CLI (`gh`) to be logged in. Standard library only.

    python scripts/search_extended.py
    -> data/sources/search_results_extended.json   (same shape as search_results.json)
       data/sources/search_extended_log.json        (queries, hits per query, filters)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources"
CREATED_FROM = "2026-08-01"
MAX_KB = 60_000

REPO_QUERIES = [
    "gemma4 swe agent", "gemma 4 swe agent", "swegemma", "gemma4 coding agent", "gemma 4 coding agent",
    "gemma4 agent kaggle", "gemma 4 kaggle agent", "gemma4 kaggle", "gemma4 swe-bench", "gemma 4 software engineering agent",
    "gemma4 scaffold", "gemma4 paper", "gemma 4 paper track", "gemma developer agent", "gemma4 bug fix agent",
    "gemma4 agent", "gemma-4 agent", "kaggle gemma 4 competition", "gemma4 competition", "gemma 4 developer",
    "gemma4 developer", "gemma agent paper", "gemma4 lora agent", "gemma 4 31b agent", "gemma 4 e4b agent",
]
# Development-set task identifiers (spread over the four libraries) and participant-only strings.
CODE_QUERIES = [
    '"fastapi_15589"', '"fastapi_14372"', '"fastapi_14851"', '"fastapi_11194"', '"fastapi_15661"', '"fastapi_14077"',
    '"fastapi_13537"', '"fastapi_15023"', '"fastapi_14794"', '"fastapi_14186"',
    '"rich_4077"', '"rich_3930"', '"rich_3006"', '"rich_3953"', '"rich_3772"', '"rich_3471"', '"rich_3944"',
    '"requests_7205"', '"requests_6644"', '"requests_7315"', '"requests_7433"', '"httpx_3672"',
    '"gemma-4-31b-it-qat-w4a16-ct"', '"gemma-4-e4b-it-qat-w4a16-ct"', '"gemma-4-12b-it-qat-w4a16-ct"',
    '"gemma-4-e2b-it-qat-w4a16-ct"', "swegemma evaluate", "swegemma summary.jsonl", "adk-submission agent.yaml gemma",
]
RELEVANT = re.compile(r"gemma", re.I)
TOPIC = re.compile(r"develop|agent|swe|kaggle|competition|paper|coding|bug|scaffold|lora", re.I)


def gh(args, retries: int = 3):
    for attempt in range(retries):
        p = subprocess.run(["gh"] + args, capture_output=True)
        if p.returncode == 0:
            return json.loads(p.stdout.decode("utf-8", "replace") or "null")
        err = p.stderr.decode("utf-8", "replace")
        if "rate limit" in err.lower() or "secondary" in err.lower() or "403" in err:
            time.sleep(30 * (attempt + 1))
            continue
        return None
    return None


def main() -> int:
    known = {m["repo"].lower() for m in json.loads((SOURCES / "manifest.json").read_text(encoding="utf-8"))}
    hits = {}
    log = {"created_from": CREATED_FROM, "repo_queries": {}, "code_queries": {}}
    for q in REPO_QUERIES:
        res = gh(["search", "repos", q, "--limit", "100", "--created", f">={CREATED_FROM}",
                  "--json", "fullName,description,createdAt,size,license,stargazersCount,updatedAt,isFork"]) or []
        log["repo_queries"][q] = len(res)
        for r in res:
            hits.setdefault(r["fullName"], {"meta": r, "via": []})["via"].append(f"repos:{q}")
        print(f"repos  {q!r}: {len(res)}", flush=True)
        time.sleep(2.2)
    for q in CODE_QUERIES:
        res = gh(["search", "code", q, "--limit", "100", "--json", "repository,path"]) or []
        repos = {}
        for r in res:
            repos.setdefault(r["repository"]["nameWithOwner"], []).append(r["path"])
        log["code_queries"][q] = {"files": len(res), "repositories": len(repos)}
        for name, paths in repos.items():
            h = hits.setdefault(name, {"meta": None, "via": []})
            h["via"].append(f"code:{q}")
            h.setdefault("paths", []).extend(paths[:5])
        print(f"code   {q!r}: {len(res)} files in {len(repos)} repositories", flush=True)
        time.sleep(7.5)

    new = {k: v for k, v in hits.items() if k.lower() not in known}
    log["distinct_repositories"] = len(hits)
    log["already_in_registered_frame"] = len(hits) - len(new)
    out, dropped = [], {"fork": 0, "created_before": 0, "too_large": 0, "no_metadata": 0, "off_topic": 0}
    for name, h in sorted(new.items(), key=lambda kv: kv[0].lower()):
        by_code = any(v.startswith("code:") for v in h["via"])
        meta = h["meta"]
        text = f"{name} {(meta or {}).get('description') or ''}"
        # A repository found only by a name search must look like a competition entry; code hits speak for themselves.
        if not by_code and not (RELEVANT.search(text) and TOPIC.search(text)):
            dropped["off_topic"] += 1
            continue
        api = gh(["api", f"repos/{name}"])
        if not api:
            dropped["no_metadata"] += 1
            continue
        if api.get("fork"):
            dropped["fork"] += 1
            continue
        if (api.get("created_at") or "") < CREATED_FROM:
            dropped["created_before"] += 1
            continue
        if (api.get("size") or 0) > MAX_KB:
            dropped["too_large"] += 1
            continue
        out.append({"fullName": name, "description": api.get("description") or "", "createdAt": api.get("created_at"),
                    "size": api.get("size"), "license": {"key": (api.get("license") or {}).get("key", "")},
                    "stargazersCount": api.get("stargazers_count"), "updatedAt": api.get("updated_at"),
                    "via": sorted(set(h["via"]))[:12], "example_paths": sorted(set(h.get("paths", [])))[:6]})
    log["new_candidates"] = len(out)
    log["dropped"] = dropped
    log["new_with_code_hit"] = sum(any(v.startswith("code:") for v in r["via"]) for r in out)
    (SOURCES / "search_results_extended.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    (SOURCES / "search_extended_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in log.items() if k not in ("repo_queries", "code_queries")}))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
