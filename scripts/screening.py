"""Write the screening table: what happened to each of the candidate sources.

One row per repository returned by the searches, with the stage at which it
left the study or the amount of data it contributed. The registered frame
(the six searches of ANALYSIS_PLAN.md section 3) and the extended frame (the
broader search made after the results, scripts/search_extended.py) are kept
in separate files.

    data/derived/screening.csv, screening_summary.json                    registered frame
    data/derived/screening_extended.csv, screening_extended_summary.json  extended frame, if any

Reads data/sources/manifest.json, the discovery and triage outputs under
data/raw/ (not committed), the adapters' cards and the pooled tables.

    python scripts/screening.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harmonize  # noqa: E402
import srcio  # noqa: E402

RAW = srcio.ROOT / "data" / "raw"
DERIVED = srcio.ROOT / "data" / "derived"
MIN_TEXT_IDS = 10
# The dataset used for external validation has a repository of its own in the extended frame. It is
# kept out of the pool so that the validation stays out of sample.
EXTERNAL_VALIDATION = {"mobius-style/gemma4-scaffold-capacity"}


def usage() -> dict:
    """source -> attempts, runs, systems in the pooled table(s)."""
    out = defaultdict(lambda: {"n_attempts": 0, "runs": set(), "systems": set()})
    path = DERIVED / "attempts_extended.csv"
    if not path.exists():
        path = DERIVED / "attempts.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            u = out[r["source"]]
            u["n_attempts"] += 1
            u["runs"].add(r["run"])
            u["systems"].add(r["system"])
    return out


def summarise(rows) -> dict:
    stages = Counter(r["stage"] for r in rows)
    fetched = [r for r in rows if r["stage"] != "not fetched"]
    with_ids = [r for r in fetched if r["stage"] != "no development-set task identifiers"]
    licensed = [r for r in with_ids if r["stage"] != "no licence"]
    return {
        "candidates": len(rows), "fetched": len(fetched),
        "not_fetched": dict(Counter(r["detail"] for r in rows if r["stage"] == "not fetched")),
        "with_dev_task_ids": len(with_ids), "of_those_licensed": len(licensed),
        "used": stages["used"], "stages": dict(stages),
        "licensed_not_examined": stages["licensed, not examined"],
        "external_validation_set": stages["external validation set"],
        "licensed_and_examined": len(licensed) - stages["licensed, not examined"] - stages["external validation set"],
        "licences_of_used": dict(Counter(r["licence_used"] for r in rows if r["stage"] == "used")),
        "labs_used": len({r["source"].split("/")[0] for r in rows if r["stage"] == "used"}),
        "attempts_used": sum(r["n_attempts"] for r in rows if r["stage"] == "used"),
    }


def main() -> int:
    manifest = json.loads(srcio.MANIFEST.read_text(encoding="utf-8"))
    cards = {m.REPO: getattr(m, "CARD", {}) for m in harmonize.load_adapters()}
    used = usage()
    by_frame = defaultdict(list)
    for m in sorted(manifest, key=lambda m: m["repo"].lower()):
        repo, stem = m["repo"], m["repo"].replace("/", "__")
        row = {"source": repo, "commit": m.get("commit", ""), "github_licence": m.get("license_spdx") or "",
               "ids_in_json_csv": "", "ids_in_text": "", "licence_used": "", "licence_evidence": "",
               "stage": "", "detail": "", "n_attempts": 0, "n_runs": 0, "n_systems": 0}
        frame = m.get("frame", "registered")
        by_frame[frame].append(row)
        if m.get("status") != "fetched":
            row["stage"] = "not fetched"
            row["detail"] = {"no_commit": "empty repository", "too_large": "larger than 60 MB"}.get(m.get("status"), str(m.get("status")))
            continue
        disc_path, tri_path = RAW / "_discovery" / f"{stem}.json", RAW / "_triage" / f"{stem}.json"
        if not disc_path.exists() or not tri_path.exists():
            row["stage"], row["detail"] = "not screened", "discovery or triage output missing"
            continue
        disc = json.loads(disc_path.read_text(encoding="utf-8"))
        tri = json.loads(tri_path.read_text(encoding="utf-8"))
        n_struct = max([max(f["id_keys"].values(), default=0) for f in disc["families"]] +
                       [f["n_name_ids"] for f in disc["families"]] + [0])
        n_text = max([f["n_ids"] for f in tri["text_files_with_task_ids"]] + [0])
        row["ids_in_json_csv"], row["ids_in_text"] = n_struct, n_text
        card = cards.get(repo)
        if card is not None:
            row["licence_used"] = (card.get("licence") or {}).get("spdx", "")
            row["licence_evidence"] = (card.get("licence") or {}).get("evidence", "")
        if n_struct == 0 and n_text < MIN_TEXT_IDS:
            row["stage"] = "no development-set task identifiers"
        elif repo in EXTERNAL_VALIDATION:
            row["stage"] = "external validation set"
            row["detail"] = "the repository of the dataset used for external validation; not pooled"
        elif card is None and frame == "extended" and (m.get("license_spdx") or "NOASSERTION") != "NOASSERTION":
            # Only in the extended frame: every licensed candidate of the registered frame was examined.
            row["stage"] = "licensed, not examined"
            row["detail"] = "not examined before the deadline"
        elif card is None:
            row["stage"] = "no licence"
            row["detail"] = "no licence file and no licence statement in the README"
        elif not card.get("eligible"):
            row["stage"] = "no licence" if "licence" in str(card.get("reason", "")).lower()[:12] else "no eligible records"
            row["detail"] = str(card.get("reason", ""))[:300]
        else:
            u = used.get(repo)
            n = u["n_attempts"] if u else 0
            row.update(n_attempts=n, n_runs=len(u["runs"]) if u else 0, n_systems=len(u["systems"]) if u else 0)
            row["stage"] = "used" if n else "no eligible records"
            if not n:
                row["detail"] = "no run with at least 10 development-set tasks after the common rules"
    for frame, name in (("registered", "screening"), ("extended", "screening_extended")):
        rows = by_frame.get(frame, [])
        if not rows:
            continue
        with (DERIVED / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        summary = summarise(rows)
        (DERIVED / f"{name}_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(frame, json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
