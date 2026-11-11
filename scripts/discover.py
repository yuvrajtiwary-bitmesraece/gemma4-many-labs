"""Schema discovery: summarise the tabular files in every fetched source.

Tarballs are read in place; nothing is extracted and nothing from a source is
executed.  The output is a per-source report used to write the adapters in
harmonize.py.  It prints structure (keys, how many values look like task ids,
low-cardinality columns) and no pooled statistic.

    python scripts/discover.py [owner/repo ...]
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources" / "manifest.json"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "raw" / "_discovery"

DATA_EXT = (".csv", ".tsv", ".jsonl", ".ndjson", ".json")
MAX_BYTES = 30 * 1024 * 1024
STRICT_ID = re.compile(r"^(fastapi|requests|rich|httpx)_(\d{1,6})$")
LOOSE_ID = re.compile(r"(fastapi|requests|rich|httpx)[_\-](\d{1,6})(?!\d)", re.I)
csv.field_size_limit(1 << 30)


def norm_name(name: str) -> str:
    """Collapse task ids and digit runs so sibling files share one family."""
    name = LOOSE_ID.sub("{task}", name)
    return re.sub(r"\d+", "#", name)


def records_from(kind: str, raw: bytes):
    """Return (records, note). records is a list of dicts, or None."""
    text = raw.decode("utf-8", "replace")
    if kind in (".jsonl", ".ndjson"):
        recs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                recs.append(obj)
        return recs, ""
    if kind in (".csv", ".tsv"):
        head = text[:4096]
        delim = "\t" if kind == ".tsv" or head.count("\t") > head.count(",") else ","
        if head.count(";") > max(head.count(","), head.count("\t")):
            delim = ";"
        try:
            return list(csv.DictReader(io.StringIO(text), delimiter=delim)), ""
        except csv.Error as exc:
            return None, f"csv error: {exc}"
    try:
        obj = json.loads(text)
    except ValueError as exc:
        return None, f"json error: {str(exc)[:60]}"
    if isinstance(obj, list):
        recs = [o for o in obj if isinstance(o, dict)]
        return recs, f"list[{len(obj)}]"
    if isinstance(obj, dict):
        keys = list(obj.keys())
        strict = [k for k in keys if STRICT_ID.match(str(k))]
        if strict and len(strict) >= 0.5 * len(keys):
            recs = []
            for k in strict:
                v = obj[k]
                rec = {"__key__": k}
                if isinstance(v, dict):
                    rec.update(v)
                else:
                    rec["__value__"] = v
                recs.append(rec)
            return recs, f"dict keyed by task id ({len(strict)}/{len(keys)})"
        # Look one level down for lists of dicts or task-keyed dicts.
        notes = []
        best = None
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                notes.append(f"{k}: list[{len(v)}] of dict")
                if best is None or len(v) > len(best[1]):
                    best = (k, [o for o in v if isinstance(o, dict)])
            elif isinstance(v, dict):
                sk = [kk for kk in v if STRICT_ID.match(str(kk))]
                if sk:
                    notes.append(f"{k}: dict keyed by task id ({len(sk)})")
                    recs = []
                    for kk in sk:
                        vv = v[kk]
                        rec = {"__key__": kk}
                        if isinstance(vv, dict):
                            rec.update(vv)
                        else:
                            rec["__value__"] = vv
                        recs.append(rec)
                    if best is None or len(recs) > len(best[1]):
                        best = (k, recs)
                else:
                    notes.append(f"{k}: dict[{len(v)}]")
            else:
                notes.append(f"{k}: {type(v).__name__}")
        note = "dict{" + "; ".join(notes[:25]) + ("; ..." if len(notes) > 25 else "") + "}"
        if best:
            return best[1], note + f" -> using '{best[0]}'"
        return [obj], note
    return None, type(obj).__name__


def summarise(recs):
    keys = Counter()
    id_hits = defaultdict(set)
    values = defaultdict(Counter)
    for rec in recs[:20000]:
        for k, v in rec.items():
            keys[k] += 1
            if isinstance(v, (dict, list)):
                values[k]["<" + type(v).__name__ + ">"] += 1
                continue
            s = str(v)
            if STRICT_ID.match(s):
                id_hits[k].add(s)
            elif LOOSE_ID.search(s) and len(s) < 80:
                id_hits[k + "~loose"].add(LOOSE_ID.search(s).group(0).lower())
            if len(values[k]) <= 12:
                values[k][s[:40]] += 1
    lowcard = {k: dict(c.most_common(8)) for k, c in values.items() if len(c) <= 8 and k in keys}
    return {
        "n": len(recs),
        "keys": [k for k, _ in keys.most_common(45)],
        "id_keys": {k: len(v) for k, v in id_hits.items()},
        "lowcard": lowcard,
    }


def discover(entry) -> dict:
    full = entry["repo"]
    tar_path = RAW / full.replace("/", "__") / f"{entry['commit']}.tar.gz"
    families = {}
    docs = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            path = m.name.split("/", 1)[1] if "/" in m.name else m.name
            low = path.lower()
            base = low.rsplit("/", 1)[-1]
            if base.startswith(("readme", "license", "licence", "notice", "copying")) or low.endswith(".md"):
                docs.append({"path": path, "bytes": m.size})
            if not low.endswith(DATA_EXT):
                continue
            kind = "." + low.rsplit(".", 1)[-1]
            fam_key = norm_name(path.rsplit("/", 1)[-1])
            fam = families.setdefault(fam_key, {
                "kind": kind, "files": 0, "bytes": 0, "dirs": set(), "examples": [],
                "name_ids": set(), "records": 0, "keys": Counter(), "id_keys": defaultdict(int),
                "lowcard": defaultdict(Counter), "notes": Counter(), "parsed": 0,
            })
            fam["files"] += 1
            fam["bytes"] += m.size
            fam["dirs"].add(path.rsplit("/", 1)[0] if "/" in path else ".")
            if len(fam["examples"]) < 3:
                fam["examples"].append(path)
            hit = LOOSE_ID.search(path.rsplit("/", 1)[-1])
            if hit:
                fam["name_ids"].add(hit.group(0).lower().replace("-", "_"))
            # Parse every tabular file, but only a sample of big per-task JSON families.
            if m.size > MAX_BYTES:
                fam["notes"]["skipped: too large"] += 1
                continue
            if kind == ".json" and fam["parsed"] >= 3 and hit:
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            recs, note = records_from(kind, fh.read())
            fam["parsed"] += 1
            if note:
                fam["notes"][note[:300]] += 1
            if not recs:
                continue
            s = summarise(recs)
            fam["records"] += s["n"]
            fam["keys"].update(s["keys"])
            for k, n in s["id_keys"].items():
                fam["id_keys"][k] = max(fam["id_keys"][k], n)
            for k, c in s["lowcard"].items():
                fam["lowcard"][k].update(c)
    out = []
    for key, fam in families.items():
        out.append({
            "family": key, "kind": fam["kind"], "files": fam["files"], "bytes": fam["bytes"],
            "n_dirs": len(fam["dirs"]), "examples": fam["examples"],
            "n_name_ids": len(fam["name_ids"]), "records": fam["records"], "parsed": fam["parsed"],
            "keys": [k for k, _ in fam["keys"].most_common(45)],
            "id_keys": dict(fam["id_keys"]),
            "lowcard": {k: dict(c.most_common(8)) for k, c in fam["lowcard"].items() if len(c) <= 10},
            "notes": dict(fam["notes"].most_common(4)),
        })
    out.sort(key=lambda f: (-(max(f["id_keys"].values(), default=0) + f["n_name_ids"]), -f["records"]))
    return {"repo": full, "commit": entry["commit"], "license_spdx": entry.get("license_spdx"),
            "docs": docs, "families": out}


def render(d) -> str:
    lines = [f"## {d['repo']}  licence={d['license_spdx']}  commit={d['commit'][:10]}"]
    docs = sorted(d["docs"], key=lambda x: -x["bytes"])[:8]
    lines.append("docs: " + ", ".join(f"{x['path']} ({x['bytes'] // 1024}K)" for x in docs))
    for f in d["families"]:
        idk = max(f["id_keys"].values(), default=0)
        if not idk and not f["n_name_ids"]:
            continue
        lines.append(
            f"- [{f['kind']}] {f['family']}  files={f['files']} dirs={f['n_dirs']} "
            f"bytes={f['bytes'] // 1024}K records={f['records']} (parsed {f['parsed']}) "
            f"name_ids={f['n_name_ids']}"
        )
        lines.append(f"    e.g. {f['examples'][0]}")
        if len(f["examples"]) > 1:
            lines.append(f"         {f['examples'][-1]}")
        lines.append(f"    id_keys={f['id_keys']}")
        lines.append(f"    keys={f['keys'][:40]}")
        lc = {k: v for k, v in f["lowcard"].items()}
        if lc:
            lines.append(f"    lowcard={json.dumps(lc, ensure_ascii=True)[:900]}")
        if f["notes"]:
            lines.append(f"    notes={json.dumps(f['notes'], ensure_ascii=True)[:500]}")
    n_other = sum(1 for f in d["families"] if not f["id_keys"] and not f["n_name_ids"])
    lines.append(f"  ({n_other} other data-file families without task ids)")
    return "\n".join(lines)


def main(argv) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = set(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        if entry.get("status") != "fetched":
            continue
        if wanted and entry["repo"] not in wanted:
            continue
        try:
            d = discover(entry)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"## {entry['repo']}  ERROR {type(exc).__name__}: {exc}")
            continue
        stem = entry["repo"].replace("/", "__")
        (OUT / f"{stem}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        text = render(d)
        (OUT / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        has = any(f["id_keys"] or f["n_name_ids"] for f in d["families"])
        print(f"{entry['repo']}: {len(d['families'])} families, task ids: {'yes' if has else 'no'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
