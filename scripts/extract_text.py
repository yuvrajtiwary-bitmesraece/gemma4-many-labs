"""Unpack the small text files of fetched sources for reading (never executed).

Only regular files with a text-like extension and a modest size are written,
under data/raw/<owner>__<repo>/x/.  Path components are sanitised, links and
special files are skipped, and nothing can be written outside the target
directory.  Per-task trace files are skipped because they are large and are not
needed to read outcomes.

    python scripts/extract_text.py [owner/repo ...]
"""
from __future__ import annotations

import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources" / "manifest.json"
RAW = ROOT / "data" / "raw"
TEXT_EXT = {".md", ".txt", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".yaml", ".yml",
            ".toml", ".cfg", ".ini", ".py", ".sh", ".tex", ".bib", ""}
MAX_BYTES = 1_500_000
BAD = re.compile(r'[<>:"|?*\x00-\x1f]')
SKIP = re.compile(r"(^|/)(trace_[^/]*\.json|node_modules/|\.git/|__pycache__/)", re.I)


def safe_parts(path: str):
    parts = []
    for part in path.split("/"):
        if part in ("", ".", ".."):
            continue
        part = BAD.sub("_", part).rstrip(" .")
        if not part:
            continue
        parts.append(part[:120])
    return parts


def main(argv) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = set(argv)
    for entry in manifest:
        if entry.get("status") != "fetched" or (wanted and entry["repo"] not in wanted):
            continue
        base = (RAW / entry["repo"].replace("/", "__") / "x").resolve()
        tar_path = RAW / entry["repo"].replace("/", "__") / f"{entry['commit']}.tar.gz"
        n = skipped = 0
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                ext = ("." + rel.rsplit(".", 1)[-1].lower()) if "." in rel.rsplit("/", 1)[-1] else ""
                if ext not in TEXT_EXT or m.size > MAX_BYTES or SKIP.search(rel):
                    skipped += 1
                    continue
                parts = safe_parts(rel)
                if not parts:
                    continue
                dest = base.joinpath(*parts)
                if base not in dest.parents or len(str(dest)) > 250:
                    skipped += 1
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(fh.read())
                n += 1
        print(f"{entry['repo']}: wrote {n}, skipped {skipped}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
