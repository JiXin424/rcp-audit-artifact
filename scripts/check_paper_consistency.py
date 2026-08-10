#!/usr/bin/env python3
"""Paper--artifact consistency test (reviewer M1 + M4).

Reads artifact/claim_manifest.json and asserts:
  (1) every source_file and audit_target_file referenced in the manifest exists;
  (2) results/accounting_table.json is internally consistent (family sums match
      headline counts; decoded <= runs; non-degenerate <= decoded);
  (3) headline counts in main_lre.tex match the accounting registry;
  (4) selected headline numbers in main_lre.tex match their JSON sources.

This is the single command an auditor runs to verify the paper's numbers trace
to machine-readable sources. Exit code 0 = all checks passed.

Run: python3 scripts/check_paper_consistency.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifact/claim_manifest.json"
ACCOUNTING = ROOT / "results/accounting_table.json"
PAPER = ROOT / "main_lre.tex"

errors = []
warnings = []
passed = []


def check_file(rel):
    p = ROOT / rel
    if p.exists():
        passed.append(f"file exists: {rel}")
        return True
    errors.append(f"MISSING file referenced in manifest: {rel}")
    return False


def main():
    # (1) Manifest file existence
    manifest = json.load(open(MANIFEST))
    for f in manifest.get("audit_target_files", []):
        check_file(f.rstrip("/"))
    for claim in manifest.get("claims", []):
        sf = claim.get("source_file")
        if sf:
            check_file(sf.rstrip("/"))

    # (2) Accounting internal consistency
    acc = json.load(open(ACCOUNTING))
    hc = acc["headline_counts"]
    fam = acc["family_detail"]
    sum_runs = sum(f["runs"] for f in fam.values())
    sum_decoded = sum(f["decoded"] for f in fam.values())
    sum_nondeg = sum(f["non_degenerate"] for f in fam.values())

    if sum_runs == hc["total_trained_runs"]:
        passed.append(f"accounting: family runs sum {sum_runs} == headline {hc['total_trained_runs']}")
    else:
        errors.append(f"accounting: family runs sum {sum_runs} != headline {hc['total_trained_runs']}")
    if sum_decoded == hc["total_decoded_gap_panel"]:
        passed.append(f"accounting: decoded sum {sum_decoded} == headline {hc['total_decoded_gap_panel']}")
    else:
        errors.append(f"accounting: decoded sum {sum_decoded} != headline {hc['total_decoded_gap_panel']}")
    if sum_nondeg == hc["total_non_degenerate"]:
        passed.append(f"accounting: non-deg sum {sum_nondeg} == headline {hc['total_non_degenerate']}")
    else:
        errors.append(f"accounting: non-deg sum {sum_nondeg} != headline {hc['total_non_degenerate']}")

    # Sanity inequalities
    if hc["total_decoded_gap_panel"] > hc["total_trained_runs"]:
        errors.append("accounting: decoded > trained (impossible)")
    if hc["total_non_degenerate"] > hc["total_decoded_gap_panel"]:
        errors.append("accounting: non-degenerate > decoded (impossible)")
    if hc["total_unique_binaries"] > hc["total_decoded_gap_panel"]:
        errors.append("accounting: unique > decoded (impossible)")

    # (3) Paper headline counts match accounting
    tex = open(PAPER).read()
    expected_counts = {
        "70": hc["total_trained_runs"],
        "41": hc["total_decoded_gap_panel"],
        "40": hc["total_unique_binaries"],
        "36": hc["total_non_degenerate"],
    }
    # Check the bold total row appears: \textbf{70} & \textbf{41} ...
    total_row = (f"\\textbf{{{hc['total_trained_runs']}}} & "
                 f"\\textbf{{{hc['total_decoded_gap_panel']}}} & "
                 f"\\textbf{{{hc['total_unique_binaries']}}} & "
                 f"\\textbf{{{hc['total_non_degenerate']}}}")
    if total_row in tex:
        passed.append(f"tex: accounting total row matches registry ({hc['total_trained_runs']}/"
                      f"{hc['total_decoded_gap_panel']}/{hc['total_unique_binaries']}/"
                      f"{hc['total_non_degenerate']})")
    else:
        errors.append(f"tex: accounting total row NOT found as {total_row}")

    # (4) Cross-check selected headline numbers against JSON sources
    cross_checks = [
        ("results/gap_43_canonical_beam3.json", ["released", "gap"], 10.243,
         "+10.24", 0.1),
        ("results/matched_donor_pool.json",
         ["matched_subsets", "caliper_0.05", "origin_effect"], 9.56,
         "+9.56", 0.1),
        ("results/donor_pool_resampling.json",
         ["resampled_640_subpool_distribution", "origin_effect_mean"], 8.38,
         "+8.38", 0.1),
        ("results/robustness_diagnostics.json",
         ["part1_spearman_readout_competence", "excl_released", "rho"], 0.907,
         "0.907", 0.005),
    ]
    for src, key_path, expected_json, tex_token, tol in cross_checks:
        if not (ROOT / src).exists():
            errors.append(f"cross-check source missing: {src}")
            continue
        d = json.load(open(ROOT / src))
        for k in (key_path if isinstance(key_path, list) else [key_path]):
            d = d[k]
        if abs(d - expected_json) > tol:
            errors.append(f"cross-check {src}: JSON value {d} != expected {expected_json}")
        else:
            passed.append(f"cross-check {src}: {key_path} = {d}")
        if tex_token.replace("+", "") in tex or tex_token in tex:
            passed.append(f"tex: contains '{tex_token}'")
        else:
            errors.append(f"tex: does NOT contain '{tex_token}'")

    # Probe multiplicity: 6/12 explored
    pm = json.load(open(ROOT / "results/probe_multiplicity.json"))
    # The manifest says 6 explored matrix cells + 2 random baselines
    explored_matrix = sum(1 for c in pm["complete_matrix"] if c.get("explored"))
    n_baseline = len(pm.get("baseline_probes", []))
    if "of the 12 retrieval--construction cells, 6 were explored" in tex.lower().replace("--", "-"):
        passed.append("tex: probe multiplicity '6 explored of 12' statement present")
    else:
        # try alternate phrasing
        if re.search(r"6 were explored", tex):
            passed.append("tex: probe multiplicity '6 were explored' present")
        else:
            errors.append("tex: probe multiplicity '6 explored' statement NOT found")

    # Report
    print("=" * 60)
    print("PAPER--ARTIFACT CONSISTENCY TEST")
    print("=" * 60)
    print(f"\nPassed checks: {len(passed)}")
    for p in passed:
        print(f"  OK  {p}")
    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for w in warnings:
            print(f"  WARN  {w}")
    if errors:
        print(f"\nFAILED checks: {len(errors)}")
        for e in errors:
            print(f"  FAIL  {e}")
        print("\n=== RESULT: FAIL ===")
        sys.exit(1)
    else:
        print("\n=== RESULT: ALL PASSED ===")
        sys.exit(0)


if __name__ == "__main__":
    main()
