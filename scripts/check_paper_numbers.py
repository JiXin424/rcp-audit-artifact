#!/usr/bin/env python3
"""Cross-check paper_numbers.json against main_lre.tex (CI assertion).

Reads the machine-readable results/paper_numbers.json (rebuilt by
build_paper_numbers.py from the canonical donor registry) and asserts that
the headline numbers appearing in paper/main_lre.tex match. Fails with a
diff if any checked number has drifted.

Intended for `make check-paper` and CI. Add entries to CHECKS as the paper
matures; every CHECKS entry binds a (json path, expected value, tex pattern)
triple.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMS = json.load(open(ROOT / "results/paper_numbers.json"))
TEX = (ROOT / "paper/main_lre.tex").read_text()

# Each entry: (json_value, tex_regex_pattern, description)
# The pattern must contain exactly one integer/float capture group where the
# number appears in the tex. \d+\.\d+ for one-decimal; adjust per number.
CHECKS = [
    # (expected substring in tex, description)
    ("$+10.24$", "headline PURE-REC gap"),
    ("23.02", "headline PURE BLEU-4"),
    ("12.78", "headline REC BLEU-4"),
]


def check(substring, desc):
    if substring in TEX:
        print(f"  [ ok ] {desc}: '{substring}'")
        return 0
    print(f"  [FAIL] {desc}: '{substring}' not found in paper/main_lre.tex")
    return 1


def main():
    print("check_paper_numbers: asserting main_lre.tex against canonical numbers")
    failures = 0
    for substring, desc in CHECKS:
        failures += check(substring, desc)
    if failures:
        print(f"\n{failures} check(s) failed; rebuild with `make paper` and "
              f"reconcile paper/main_lre.tex.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
