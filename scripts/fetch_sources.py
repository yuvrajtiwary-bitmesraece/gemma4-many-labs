"""Pin and download every candidate source repository.

For each repository in data/sources/search_results.json this script records
the HEAD commit, the licence GitHub detects, and downloads the tarball of that
exact commit into data/raw/ (which is not committed).  Re-running it never
moves a pin: sources already present in data/sources/manifest.json are fetched
at the recorded commit.

Requires the GitHub CLI (`gh`) to be logged in.  Standard library only.

    python scripts/fetch_sources.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "data" / "sources" / "search_results.json"
MANIFEST = ROOT / "data" / "sources" / "manifest.json"
RAW = ROOT / "data" / "raw"

# Repositories larger than this (GitHub's reported size, in KB) are not
# downloaded automatically; they are logged with status "too_large".
MAX_KB = 60_000


def gh_json(path: str):
    p = subprocess.run(["gh", "api", path], capture_output=True)
    if p.returncode != 0:
        return None, p.stderr.decode("utf-8", "replace").strip()[:200]
    return json.loads(p.stdout.decode("utf-8", "replace")), None


def save(manifest) -> None:
    """Write the manifest atomically, so that a concurrent reader never sees half a file."""
    tmp = MANIFEST.with_suffix(".json.part")
    tmp.write_text(json.dumps(sorted(manifest.values(), key=lambda m: m["repo"].lower()), indent=1) + "\n",
                   encoding="utf-8")
    tmp.replace(MANIFEST)


def main() -> int:
    # Without arguments: the registered frame (the six searches of the plan). With a search file as
    # argument (data/sources/search_results_extended.json): the extended frame, added after the results.
    search = Path(sys.argv[1]) if len(sys.argv) > 1 else SEARCH
    frame = "extended" if len(sys.argv) > 1 else "registered"
    repos = json.loads(search.read_text(encoding="utf-8"))
    manifest = {}
    if MANIFEST.exists():
        manifest = {m["repo"]: m for m in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    RAW.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(repos, 1):
        full = r["fullName"]
        entry = manifest.get(full, {"repo": full, "frame": frame})
        entry.setdefault("description", r.get("description") or "")
        entry.setdefault("created_at", r.get("createdAt"))
        entry.setdefault("size_kb", r.get("size"))

        if "commit" not in entry:
            head, err = gh_json(f"repos/{full}/commits/HEAD")
            if head is None:
                entry["status"] = "no_commit"
                entry["error"] = err
                manifest[full] = entry
                print(f"[{i}/{len(repos)}] {full}: no commit ({err})", flush=True)
                continue
            entry["commit"] = head["sha"]
            entry["commit_date"] = head["commit"]["committer"]["date"]
            lic, _ = gh_json(f"repos/{full}/license")
            entry["license_spdx"] = (lic or {}).get("license", {}).get("spdx_id")
            entry["license_path"] = (lic or {}).get("path")

        if (entry.get("size_kb") or 0) > MAX_KB:
            entry["status"] = "too_large"
            manifest[full] = entry
            print(f"[{i}/{len(repos)}] {full}: too large, skipped", flush=True)
            continue

        dest_dir = RAW / full.replace("/", "__")
        dest = dest_dir / f"{entry['commit']}.tar.gz"
        if not dest.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                p = subprocess.run(
                    ["gh", "api", f"repos/{full}/tarball/{entry['commit']}"],
                    stdout=fh, stderr=subprocess.PIPE,
                )
            if p.returncode != 0:
                tmp.unlink(missing_ok=True)
                entry["status"] = "download_failed"
                entry["error"] = p.stderr.decode("utf-8", "replace").strip()[:200]
                manifest[full] = entry
                print(f"[{i}/{len(repos)}] {full}: download failed", flush=True)
                continue
            tmp.replace(dest)
            entry["fetched_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        h = hashlib.sha256()
        with dest.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        entry["tarball_sha256"] = h.hexdigest()
        entry["tarball_bytes"] = dest.stat().st_size
        entry["status"] = "fetched"
        entry.pop("error", None)
        manifest[full] = entry
        print(f"[{i}/{len(repos)}] {full}: {entry['commit'][:10]} "
              f"{entry['tarball_bytes'] // 1024} KB licence={entry['license_spdx']}", flush=True)

        save(manifest)

    save(manifest)
    by = {}
    for m in manifest.values():
        by[m.get("status")] = by.get(m.get("status"), 0) + 1
    print("done:", by)
    return 0


if __name__ == "__main__":
    sys.exit(main())
