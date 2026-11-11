"""Triage every fetched source in one pass over its tarball.

For each source this records, without unpacking anything:

- licence evidence: licence-like files (first lines) and README lines that
  mention a licence;
- where development-set task identifiers occur outside JSON/CSV files
  (markdown, notebooks, logs, text), which schema discovery does not read;
- an inventory of file extensions, to spot result files in other formats.

Output: data/raw/_triage/<owner>__<repo>.json and a one-line summary each.

    python scripts/triage.py [owner/repo ...]
"""
from __future__ import annotations

import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources" / "manifest.json"
RAW = ROOT / "data" / "raw"
OUT = RAW / "_triage"

TASK_ID = re.compile(r"(?<![A-Za-z0-9])(fastapi|requests|rich|httpx)[_\-](\d{1,6})(?!\d)", re.I)
LICENCE_FILE = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE|UNLICENSE)[^/]*$", re.I)
README_FILE = re.compile(r"(^|/)README[^/]*$", re.I)
LICENCE_LINE = re.compile(r"licen[cs]e|apache|\bMIT\b|creative commons|CC[- ]BY|CC0|BSD|GPL|public domain|"
                          r"all rights reserved|copyright", re.I)
TEXT_EXT = {".md", ".txt", ".rst", ".ipynb", ".log", ".html", ".htm", ".tex", ".out", ".tsv", ".yaml", ".yml"}
MAX_BYTES = 30 * 1024 * 1024


def triage(entry) -> dict:
    tar_path = RAW / entry["repo"].replace("/", "__") / f"{entry['commit']}.tar.gz"
    ext_count, ext_bytes = Counter(), Counter()
    licence_files, readme_lines, id_files = [], [], []
    n_files = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            n_files += 1
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            base = rel.rsplit("/", 1)[-1]
            ext = ("." + base.rsplit(".", 1)[-1].lower()) if "." in base else ""
            ext_count[ext] += 1
            ext_bytes[ext] += m.size
            is_lic = bool(LICENCE_FILE.search(rel))
            is_readme = bool(README_FILE.search(rel))
            if not (is_lic or is_readme or ext in TEXT_EXT) or m.size > MAX_BYTES:
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            text = fh.read().decode("utf-8", "replace")
            if is_lic:
                head = " ".join(text.split())[:260]
                licence_files.append({"path": rel, "bytes": m.size, "head": head})
            if is_readme:
                for i, line in enumerate(text.splitlines(), 1):
                    if LICENCE_LINE.search(line) and len(readme_lines) < 40:
                        readme_lines.append({"path": rel, "line": i, "text": line.strip()[:240]})
            if ext in TEXT_EXT:
                ids = {f"{a.lower()}_{int(b)}" for a, b in TASK_ID.findall(text)}
                if len(ids) >= 5:
                    id_files.append({"path": rel, "bytes": m.size, "n_ids": len(ids)})
    id_files.sort(key=lambda f: -f["n_ids"])
    return {
        "repo": entry["repo"], "commit": entry["commit"], "license_spdx": entry.get("license_spdx"),
        "license_path": entry.get("license_path"), "n_files": n_files,
        "ext": {k: [ext_count[k], ext_bytes[k]] for k, _ in ext_count.most_common(25)},
        "licence_files": licence_files, "readme_licence_lines": readme_lines,
        "text_files_with_task_ids": id_files[:40], "n_text_files_with_task_ids": len(id_files),
    }


def main(argv) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = set(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        if entry.get("status") != "fetched" or (wanted and entry["repo"] not in wanted):
            continue
        try:
            t = triage(entry)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{entry['repo']}: ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        (OUT / (entry["repo"].replace("/", "__") + ".json")).write_text(json.dumps(t, indent=1), encoding="utf-8")
        best = t["text_files_with_task_ids"][0]["n_ids"] if t["text_files_with_task_ids"] else 0
        print(f"{entry['repo']}: files={t['n_files']} spdx={t['license_spdx']} licence_files={len(t['licence_files'])} "
              f"readme_licence_lines={len(t['readme_licence_lines'])} text_id_files={t['n_text_files_with_task_ids']} "
              f"max_ids={best}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
