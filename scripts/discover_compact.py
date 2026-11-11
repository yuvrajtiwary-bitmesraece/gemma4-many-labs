"""Compact view of the discovery output for sources with many file families.

    python scripts/discover_compact.py owner/repo [owner/repo ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "_discovery"
OUTCOME = re.compile(r"(resolv|success|pass|status|outcome|solved|result|verdict|correct|score|model|variant|seed|rep|arm|config|cond|mode|split|set$)", re.I)


def main(argv) -> int:
    for repo in argv:
        d = json.loads((OUT / (repo.replace("/", "__") + ".json")).read_text(encoding="utf-8"))
        print(f"## {d['repo']}  licence={d['license_spdx']}")
        docs = sorted(d["docs"], key=lambda x: -x["bytes"])[:6]
        print("docs: " + ", ".join(f"{x['path']} ({x['bytes'] // 1024}K)" for x in docs))
        for f in d["families"]:
            idk = max(f["id_keys"].values(), default=0)
            if not idk and not f["n_name_ids"]:
                continue
            lc = {k: v for k, v in f["lowcard"].items() if OUTCOME.search(k)}
            print(f"- [{f['kind']}] {f['examples'][0]}  (x{f['files']} files, {f['n_dirs']} dirs) recs={f['records']} "
                  f"ids={f['id_keys']} name_ids={f['n_name_ids']}")
            print(f"    keys={f['keys'][:22]}")
            if lc:
                print(f"    {json.dumps(lc, ensure_ascii=True)[:520]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
