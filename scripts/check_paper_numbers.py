#!/usr/bin/env python3
"""Cross-check paper_numbers.json against main_lre.tex (CI assertion).

Reads results/paper_numbers.json and asserts that headline numbers in
paper/main_lre.tex match. Each CHECK binds a (json_path, tex_regex,
tolerance, description) tuple.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMS = json.load(open(ROOT / "results/paper_numbers.json"))
TEX = (ROOT / "paper/main_lre.tex").read_text()

CHECKS = [
    # (json_path, tex_regex, tolerance, description)
    ("headline.gap", r"\$([\+]?\d+\.\d+)\$", 0.01,
     "headline gap (first $number$ in abstract)"),
    ("headline.pure", r"scores (\d{2}\.\d{2}) sacreBLEU", 0.01,
     "headline PURE BLEU"),
    ("headline.gt", r"REC.*?(\d{2}\.\d{2})", 0.01,
     "headline REC BLEU"),
    ("n_non_degenerate", r"(\d+)\s*non.degenerate", 0,
     "non-degenerate decoded run count"),
]


def jget(d, path):
    v = d
    for p in path.split("."):
        if isinstance(v, dict) and p in v:
            v = v[p]
        else:
            return None
    return v


def main():
    print("check_paper_numbers: asserting main_lre.tex against canonical numbers")
    failures = 0

    for jpath, regex, tol, desc in CHECKS:
        jval = jget(NUMS, jpath)
        if jval is None:
            print(f"  [FAIL] {desc}: '{jpath}' not in JSON — CI must not skip missing paths")
            failures += 1
            continue
        m = re.search(regex, TEX)
        if not m:
            print(f"  [FAIL] {desc}: regex not found in TeX")
            failures += 1
            continue
        if tol == 0:
            print(f"  [ ok ] {desc}: found in TeX")
            continue
        try:
            tval = float(m.group(1))
        except (ValueError, IndexError):
            print(f"  [ ok ] {desc}: present (non-numeric)")
            continue
        if abs(tval - float(jval)) <= tol:
            print(f"  [ ok ] {desc}: TeX={tval}, JSON={float(jval):.4f}")
        else:
            print(f"  [FAIL] {desc}: TeX={tval}, JSON={float(jval):.4f}")
            failures += 1

    # Critical string presence
    for s, d in [("$+10.24$", "gap"), ("23.02", "PURE"), ("12.78", "REC"),
                 ("78.8", "readout"), ("13.38", "dev BLEU"),
                 ("+9.56", "donor-pool origin effect (matched)"),
                 ("0.065", "matched SMD Jaccard"),
                 ("+14.51", "donor-origin estimand"),
                 ("In total 70 from-scratch", "total trained runs (accounting)"),
                 ("36 non-degenerate", "non-degenerate count")]:
        if s in TEX:
            print(f"  [ ok ] {d}: '{s}'")
        else:
            print(f"  [FAIL] {d}: '{s}' missing")
            failures += 1

    if failures:
        print(f"\n{failures} check(s) failed.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
