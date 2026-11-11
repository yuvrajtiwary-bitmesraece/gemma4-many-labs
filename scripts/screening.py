"""Write the screening table: what happened to each of the candidate sources.

One row per repository returned by the searches in ANALYSIS_PLAN.md section 3,
with the stage at which it left the study or the amount of data it contributed.

    data/derived/screening.csv
    data/derived/screening_summary.json

Reads data/sources/manifest.json, the discovery and triage outputs under
data/raw/ (not committed), the adapters' cards and data/derived/sources.csv.

    python scripts/screening.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harmonize  # noqa: E402
import srcio  # noqa: E402

RAW = srcio.ROOT / "data" / "raw"
DERIVED = srcio.ROOT / "data" / "derived"
MIN_TEXT_IDS = 10


def main() -> int:
    manifest = json.loads(srcio.MANIFEST.read_text(encoding="utf-8"))
    cards = {m.REPO: getattr(m, "CARD", {}) for m in harmonize.load_adapters()}
    used = {}
    with (DERIVED / "sources.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            used[row["source"]] = row
    rows = []
    for m in sorted(manifest, key=lambda m: m["repo"].lower()):
        repo, stem = m["repo"], m["repo"].replace("/", "__")
        row = {"source": repo, "commit": m.get("commit", ""), "github_licence": m.get("license_spdx") or "",
               "ids_in_json_csv": "", "ids_in_text": "", "licence_used": "", "licence_evidence": "",
               "stage": "", "detail": "", "n_attempts": 0, "n_runs": 0, "n_systems": 0}
        if m.get("status") != "fetched":
            row["stage"] = "not fetched"
            row["detail"] = "empty repository" if m.get("status") == "no_commit" else "larger than 60 MB"
            rows.append(row)
            continue
        disc = json.loads((RAW / "_discovery" / f"{stem}.json").read_text(encoding="utf-8"))
        tri = json.loads((RAW / "_triage" / f"{stem}.json").read_text(encoding="utf-8"))
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
        elif card is None:
            row["stage"] = "no licence"
            row["detail"] = "no licence file and no licence statement in the README"
        elif not card.get("eligible"):
            row["stage"] = "no licence" if "licence" in str(card.get("reason", "")).lower()[:12] else "no eligible records"
            row["detail"] = str(card.get("reason", ""))[:300]
        else:
            u = used.get(repo, {})
            n = int(u.get("n_attempts") or 0)
            row.update(n_attempts=n, n_runs=int(u.get("n_runs") or 0), n_systems=int(u.get("n_systems") or 0))
            row["stage"] = "used" if n else "no eligible records"
            if not n:
                row["detail"] = "no run with at least 10 development-set tasks after the common rules"
        rows.append(row)
    with (DERIVED / "screening.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    stages = Counter(r["stage"] for r in rows)
    fetched = [r for r in rows if r["stage"] != "not fetched"]
    with_ids = [r for r in fetched if r["stage"] != "no development-set task identifiers"]
    licensed = [r for r in with_ids if r["stage"] != "no licence"]
    summary = {
        "candidates": len(rows), "fetched": len(fetched),
        "not_fetched": dict(Counter(r["detail"] for r in rows if r["stage"] == "not fetched")),
        "with_dev_task_ids": len(with_ids), "of_those_licensed": len(licensed),
        "used": stages["used"], "stages": dict(stages),
        "licences_of_used": dict(Counter(r["licence_used"] for r in rows if r["stage"] == "used")),
        "labs_used": len({r["source"].split("/")[0] for r in rows if r["stage"] == "used"}),
    }
    (DERIVED / "screening_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
