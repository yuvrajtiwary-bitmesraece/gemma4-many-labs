"""Read files of a pinned source straight from its tarball.

Nothing is unpacked to disk and nothing from a source is executed.  A source
is addressed as "owner/repo"; its tarball is the commit recorded in
data/sources/manifest.json.

    src = Source("owner/repo")
    src.ls(r"results/.*\\.jsonl$")           -> [(path, size), ...]
    src.text("README.md")                    -> str
    for path, rows in src.jsonl(regex): ...  -> list of dicts per file
    for path, rows in src.csv(regex): ...    -> list of dicts per file
    for path, obj in src.json(regex): ...    -> parsed JSON per file

Command line:

    python scripts/srcio.py owner/repo --ls [regex]
    python scripts/srcio.py owner/repo --cat path [start [n_chars]]
    python scripts/srcio.py owner/repo --grep regex [path_regex]
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources" / "manifest.json"
RAW = ROOT / "data" / "raw"
csv.field_size_limit(1 << 30)

_manifest_cache = None


def manifest() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = {m["repo"]: m for m in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    return _manifest_cache


class Source:
    """One pinned repository snapshot, read in place."""

    def __init__(self, repo: str):
        self.repo = repo
        self.entry = manifest()[repo]
        self.commit = self.entry["commit"]
        self.tar_path = RAW / repo.replace("/", "__") / f"{self.commit}.tar.gz"
        self._listing = None

    # -- low level ---------------------------------------------------------
    def _members(self):
        with tarfile.open(self.tar_path, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                yield tf, m, rel

    def ls(self, pattern: str | None = None):
        """[(path, size)] of regular files whose path matches the regex."""
        if self._listing is None:
            self._listing = [(rel, m.size) for _, m, rel in self._members()]
        if pattern is None:
            return list(self._listing)
        pat = re.compile(pattern)
        return [(p, s) for p, s in self._listing if pat.search(p)]

    def read_many(self, pattern: str, max_bytes: int = 40 * 1024 * 1024):
        """Yield (path, bytes) for files whose path matches, in archive order."""
        pat = re.compile(pattern)
        for tf, m, rel in self._members():
            if pat.search(rel) and m.size <= max_bytes:
                fh = tf.extractfile(m)
                if fh is not None:
                    yield rel, fh.read()

    def bytes(self, path: str) -> bytes:
        for tf, m, rel in self._members():
            if rel == path:
                return tf.extractfile(m).read()
        raise FileNotFoundError(f"{self.repo}: {path}")

    def text(self, path: str) -> str:
        return self.bytes(path).decode("utf-8", "replace")

    # -- parsed views ------------------------------------------------------
    def jsonl(self, pattern: str):
        """Yield (path, [dict, ...]) for line-delimited JSON files."""
        for path, raw in self.read_many(pattern):
            rows = []
            for line in raw.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
            yield path, rows

    def json(self, pattern: str):
        """Yield (path, parsed object) for JSON files; unparseable files are skipped."""
        for path, raw in self.read_many(pattern):
            try:
                yield path, json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue

    def csv(self, pattern: str, delimiter: str | None = None):
        """Yield (path, [dict, ...]) for delimited text files with a header row."""
        for path, raw in self.read_many(pattern):
            text = raw.decode("utf-8-sig", "replace")
            delim = delimiter
            if delim is None:
                head = text[:4096]
                delim = "\t" if path.lower().endswith(".tsv") or head.count("\t") > head.count(",") else ","
                if head.count(";") > max(head.count(","), head.count("\t")):
                    delim = ";"
            yield path, list(csv.DictReader(io.StringIO(text), delimiter=delim))


def truthy(value) -> bool:
    """Interpret the usual spellings of a boolean outcome."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "resolved", "pass", "passed",
                                          "success", "solved", "ok"}


def _main(argv) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    src = Source(argv[0])
    mode = argv[1]
    if mode == "--ls":
        for path, size in src.ls(argv[2] if len(argv) > 2 else None):
            print(f"{size:>10}  {path}")
        return 0
    if mode == "--cat":
        text = src.text(argv[2])
        start = int(argv[3]) if len(argv) > 3 else 0
        n = int(argv[4]) if len(argv) > 4 else 8000
        print(f"[{argv[2]}: {len(text)} chars, showing {start}..{start + n}]")
        print(text[start:start + n])
        return 0
    if mode == "--grep":
        pat = re.compile(argv[2], re.I)
        path_pat = argv[3] if len(argv) > 3 else r"\.(md|txt|rst|ya?ml|toml|py|cfg)$|(^|/)(README|LICEN[CS]E|NOTICE)[^/]*$"
        shown = 0
        for path, raw in src.read_many(path_pat, max_bytes=3_000_000):
            for i, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                if pat.search(line):
                    print(f"{path}:{i}: {line.strip()[:300]}")
                    shown += 1
                    if shown >= 200:
                        return 0
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
