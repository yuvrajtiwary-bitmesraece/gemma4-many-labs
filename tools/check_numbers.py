"""List the numbers in a text that do not occur in the generated results.

A check for anyone who writes about these results (the paper's author first of
all): every number in the text should be traceable to results/RESULTS.md.

    python tools/check_numbers.py draft.md
    python tools/check_numbers.py draft.md --against results/RESULTS.md DEVIATIONS.md

A number in the text counts as found if the reference files contain the same
number, or one that rounds to it at the precision used in the text (the text
may say 11% where the results say 10.9%). Percentages and proportions are
matched both ways (80% and 0.80). Numbers that are not found are printed with
their line, to be checked by hand; a number that is found may still be used
for the wrong quantity, so this does not replace reading.

Standard library only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d+(?:\.\d+)?")
SKIP = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}|\b(?:19|20)\d{2}\b|doi:\S+|zenodo\.\d+|https?://\S+|\bfig(?:ure)?\.?\s*\d+|\bS\d+\b|\bL\d{2}\b"
                  r"|\bP\d\b|\bE\d\b|\bX\d\b|\bF\d\b|pp\.\s*\d+\s*-\s*\d+|\b\d+\(\d+\)", re.I)


def numbers(text: str):
    """(value, decimals, raw) for every number in a text, without ids, years, links and labels."""
    text = SKIP.sub(" ", text)
    out = []
    for m in NUMBER.finditer(text):
        raw = m.group(0)
        clean = raw.replace(",", "").lstrip("+")
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append((value, decimals, raw))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("draft")
    ap.add_argument("--against", nargs="+", default=[str(ROOT / "results" / "RESULTS.md")],
                    help="reference files (default: results/RESULTS.md)")
    ap.add_argument("--min", type=float, default=13, help="whole numbers below this are not checked (default 13)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    reference = []
    for path in args.against:
        reference += [v for v, _, _ in numbers(Path(path).read_text(encoding="utf-8"))]
    ref = sorted(set(reference))

    def found(value: float, decimals: int) -> bool:
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        for cand in (value, value / 100, value * 100):
            scale = 1 if cand == value else (0.01 if cand == value / 100 else 100)
            t = tol * scale
            lo, hi = 0, len(ref)
            while lo < hi:                      # first reference number >= cand - t
                mid = (lo + hi) // 2
                if ref[mid] < cand - t:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(ref) and ref[lo] <= cand + t:
                return True
        return False

    missing = checked = 0
    for i, line in enumerate(Path(args.draft).read_text(encoding="utf-8").splitlines(), 1):
        bad = []
        for value, decimals, raw in numbers(line):
            if decimals == 0 and abs(value) < args.min:
                continue
            checked += 1
            if not found(abs(value), decimals):
                bad.append(raw)
        if bad:
            missing += len(bad)
            print(f"line {i}: {', '.join(bad)}\n    {line.strip()[:200]}")
    print(f"\n{checked} numbers checked, {missing} not found in {', '.join(Path(p).name for p in args.against)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
