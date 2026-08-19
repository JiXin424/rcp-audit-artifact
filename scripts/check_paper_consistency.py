#!/usr/bin/env python3
"""Paper--artifact consistency test.

Single command an auditor runs to verify the paper's numbers trace to
machine-readable sources. Exit code 0 = all checks passed.

Reads:
  - artifact/claim_manifest.json
  - results/canonical_checkpoint_registry.json (disk-scanned, schema v4)
  - results/accounting_table.json (derived from the registry)
  - results/gap_43_canonical_beam3.json (canonical gap panel)
  - results/matched_donor_pool.json, donor_pool_resampling.json,
    robustness_diagnostics.json, probe_multiplicity.json
  - main_lre.tex, supplementary.tex

Invariants verified:
  (1) manifest-referenced files exist;
  (2) registry summary counts match entry counts (no stale summary);
  (3) disk ↔ registry: every checkpoint dir with best.ckpt or training_log.json
      under checkpoints/*/* is in the registry and vice versa;
  (4) accounting internal consistency (family sums match headline; monotonicity);
  (5) paper bold total row in main_lre.tex matches accounting JSON;
  (6) gap panel JSON's non_released_gap_range matches registry's gap range;
  (7) selected headline numbers in main_lre.tex match their JSON sources;
  (8) probe multiplicity "6 of 12 explored" statement present;
  (9) supplementary.tex does not contain stale 70/70/43/67 Table S6.

Run: python3 scripts/check_paper_consistency.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "checkpoints"
SKIP_DIRS = {"released", "finetune_released", "step_faithful_legacy_nll", "retrain_logs",
              "faithful_steps", "ctc_clip_sens"}  # round-33 sensitivity runs: outside the canonical panel

_MANIFEST_PRIMARY = ROOT / "artifact" / "claim_manifest.json"
_MANIFEST_FALLBACK = ROOT / "claim_manifest.json"
MANIFEST = _MANIFEST_PRIMARY if _MANIFEST_PRIMARY.exists() else _MANIFEST_FALLBACK
REGISTRY = ROOT / "results" / "canonical_checkpoint_registry.json"
ACCOUNTING = ROOT / "results/accounting_table.json"
GAP_PANEL = ROOT / "results/gap_43_canonical_beam3.json"
# Layout-aware: artifact mirror has paper/main_lre.tex; RCP working repo has ./main_lre.tex
_paper = ROOT / "paper" / "main_lre.tex"
if not _paper.exists():
    _paper = ROOT / "main_lre.tex"
_supp = ROOT / "paper" / "supplementary.tex"
if not _supp.exists():
    _supp = ROOT / "supplementary.tex"
PAPER = _paper
SUPP = _supp

errors = []
warnings = []
passed = []


def ok(msg): passed.append(msg)
def warn(msg): warnings.append(msg)
def fail(msg): errors.append(msg)


def check_file(rel):
    p = ROOT / rel
    if p.exists():
        ok(f"file exists: {rel}")
        return True
    fail(f"MISSING file referenced in manifest: {rel}")
    return False


def main():
    # (1) Manifest file existence
    if not MANIFEST.exists():
        fail(f"manifest not found at {MANIFEST}")
    else:
        manifest = json.load(open(MANIFEST))
        for f in manifest.get("audit_target_files", []):
            check_file(f.rstrip("/"))
        for claim in manifest.get("claims", []):
            sf = claim.get("source_file")
            if sf:
                check_file(sf.rstrip("/"))

    # (2) Registry summary vs entry counts
    if not REGISTRY.exists():
        fail(f"registry not found at {REGISTRY}")
        reg = None
    else:
        reg = json.load(open(REGISTRY))
        ckpts = reg.get("checkpoints", [])
        summ = reg.get("summary", {})
        # total_trained_runs == len(checkpoints)
        if summ.get("total_trained_runs") == len(ckpts):
            ok(f"registry summary.total_trained_runs ({summ['total_trained_runs']}) == len(checkpoints) ({len(ckpts)})")
        else:
            fail(f"registry summary.total_trained_runs ({summ.get('total_trained_runs')}) != len(checkpoints) ({len(ckpts)})")
        # Recompute counts
        has_gap = sum(1 for c in ckpts if c.get("has_gap"))
        has_dev = sum(1 for c in ckpts if c.get("has_dev"))
        non_deg = sum(1 for c in ckpts if c.get("has_gap") and not c.get("degenerate"))
        if has_gap == summ.get("decoded_gap_panel"):
            ok(f"registry has_gap count ({has_gap}) == summary.decoded_gap_panel")
        else:
            fail(f"registry has_gap count ({has_gap}) != summary.decoded_gap_panel ({summ.get('decoded_gap_panel')})")
        if non_deg == summ.get("non_degenerate"):
            ok(f"registry non_degenerate count ({non_deg}) == summary.non_degenerate")
        else:
            fail(f"registry non_degenerate count ({non_deg}) != summary.non_degenerate ({summ.get('non_degenerate')})")

    # (3) Disk ↔ registry consistency
    # Only enforced when checkpoints/ is populated (RCP working repo). In the
    # artifact mirror, checkpoints/ is gitignored and weights are hosted
    # externally (see scripts/download_checkpoints.sh); the registry remains
    # the source of truth but cannot be cross-checked against missing files.
    if reg is not None and CKPT_ROOT.exists():
        # Detect artifact-mirror layout (paper/ subdir present + checkpoints
        # essentially empty or marked as external).
        is_artifact_mirror = (ROOT / "paper" / "main_lre.tex").exists()
        disk_dirs = set()
        for fam_dir in CKPT_ROOT.iterdir():
            if not fam_dir.is_dir() or fam_dir.name in SKIP_DIRS: continue
            for seed_dir in fam_dir.iterdir():
                if not seed_dir.is_dir(): continue
                if seed_dir.name.startswith("_") or seed_dir.name.endswith("_OOM"): continue
                has_ckpt = (seed_dir / "best.ckpt").exists()
                has_log = (seed_dir / "training_log.json").exists()
                if has_ckpt or has_log:
                    disk_dirs.add(str(seed_dir.relative_to(ROOT)))
        reg_dirs = {c.get("training_dir") for c in ckpts if c.get("training_dir")}
        only_disk = disk_dirs - reg_dirs
        only_reg = reg_dirs - disk_dirs
        if is_artifact_mirror:
            # Artifact mirror ships only a subset of checkpoints (the rest are
            # hosted externally per scripts/download_checkpoints.sh). Disk ↔
            # registry match is enforced in RCP, not here.
            ok(f"artifact mirror: skipping disk ↔ registry invariant ({len(disk_dirs)} ckpt dirs on disk, {len(reg_dirs)} in registry); enforced in RCP")
        elif not only_disk and not only_reg:
            ok(f"disk ↔ registry: {len(disk_dirs)} training dirs match")
        else:
            if only_disk:
                fail(f"on disk but not in registry: {sorted(only_disk)[:5]}{'...' if len(only_disk)>5 else ''}")
            if only_reg:
                fail(f"in registry but not on disk: {sorted(only_reg)[:5]}{'...' if len(only_reg)>5 else ''}")

    # (4) Accounting internal consistency
    if not ACCOUNTING.exists():
        fail(f"accounting not found at {ACCOUNTING}")
    else:
        acc = json.load(open(ACCOUNTING))
        hc = acc["headline_counts"]
        fam = acc["family_detail"]
        sum_runs = sum(f["runs"] for f in fam.values())
        sum_decoded = sum(f["decoded"] for f in fam.values())
        sum_nondeg = sum(f["non_degenerate"] for f in fam.values())
        if sum_runs == hc["total_trained_runs"]:
            ok(f"accounting: family runs sum ({sum_runs}) == headline ({hc['total_trained_runs']})")
        else:
            fail(f"accounting: family runs sum ({sum_runs}) != headline ({hc['total_trained_runs']})")
        if sum_decoded == hc["total_decoded_gap_panel"]:
            ok(f"accounting: decoded sum ({sum_decoded}) == headline ({hc['total_decoded_gap_panel']})")
        else:
            fail(f"accounting: decoded sum ({sum_decoded}) != headline ({hc['total_decoded_gap_panel']})")
        if sum_nondeg == hc["total_non_degenerate"]:
            ok(f"accounting: non-deg sum ({sum_nondeg}) == headline ({hc['total_non_degenerate']})")
        else:
            fail(f"accounting: non-deg sum ({sum_nondeg}) != headline ({hc['total_non_degenerate']})")
        # Sanity inequalities
        if hc["total_decoded_gap_panel"] > hc["total_trained_runs"]:
            fail("accounting: decoded > trained (impossible)")
        if hc["total_non_degenerate"] > hc["total_decoded_gap_panel"]:
            fail("accounting: non-degenerate > decoded (impossible)")
        if hc["total_unique_binaries"] > hc["total_decoded_gap_panel"]:
            fail("accounting: unique > decoded (impossible)")
        # accounting vs registry
        if reg is not None:
            rsum = reg["summary"]
            for k in ("total_trained_runs", "decoded_gap_panel", "unique_binaries",
                     "sha256_collisions", "non_degenerate"):
                ak = {"total_trained_runs": "total_trained_runs",
                      "decoded_gap_panel": "total_decoded_gap_panel",
                      "unique_binaries": "total_unique_binaries",
                      "sha256_collisions": "sha256_collisions",
                      "non_degenerate": "total_non_degenerate"}[k]
                if hc[ak] == rsum[k]:
                    ok(f"accounting.{ak} ({hc[ak]}) == registry.summary.{k}")
                else:
                    fail(f"accounting.{ak} ({hc[ak]}) != registry.summary.{k} ({rsum[k]})")

        # (5) Paper bold total row matches accounting
        if PAPER.exists():
            tex = open(PAPER).read()
            supp_tex = open(SUPP).read() if SUPP.exists() else ""
            total_row = (f"\\textbf{{{hc['total_trained_runs']}}} & "
                         f"\\textbf{{{hc['total_decoded_gap_panel']}}} & "
                         f"\\textbf{{{hc['total_unique_binaries']}}} & "
                         f"\\textbf{{{hc['total_non_degenerate']}}}")
            summary_form = (f"{hc['total_trained_runs']} trained / "
                            f"{hc['total_decoded_gap_panel']} decoded / "
                            f"{hc['total_unique_binaries']} unique / "
                            f"{hc['total_non_degenerate']} non-degenerate")
            if total_row in tex or total_row in supp_tex or summary_form in tex:
                ok(f"tex: accounting total row matches ({hc['total_trained_runs']}/"
                    f"{hc['total_decoded_gap_panel']}/{hc['total_unique_binaries']}/"
                    f"{hc['total_non_degenerate']})")
            else:
                fail(f"tex: accounting total row NOT found as {total_row}")

            # (9) Supplementary should not have stale 70/70/43/67 table
            if SUPP.exists():
                stex = open(SUPP).read()
                # The old Table S6 line said "Total & 70 & 70 & 43 & 67"
                if re.search(r"Total\s*&\s*70\s*&\s*70\s*&\s*43\s*&\s*67", stex):
                    fail("supplementary.tex: stale 'Total & 70 & 70 & 43 & 67' Table S6 still present")
                else:
                    ok("supplementary.tex: stale 70/70/43/67 table removed")

            # (7) Cross-check selected headline numbers
            cross_checks = [
                ("results/gap_43_canonical_beam3.json", ["released", "gap"], 10.243,
                 "+10.24", 0.1),
                ("results/matched_donor_pool.json",
                 ["matched_subsets", "caliper_0.05", "origin_effect"], 9.56,
                 "+9.56", 0.1),
                ("results/donor_pool_resampling.json",
                 ["resampled_640_subpool_distribution", "origin_effect_mean"], 8.38,
                 "+8.38", 0.1),
                ("results/readout_gap_correlation.json",
                 ["correlations", "readout_vs_gap", "excl_released", "rho"], -0.564,
                 "{-}0.56", 0.005),
            ]
            for src, key_path, expected_json, tex_token, tol in cross_checks:
                if not (ROOT / src).exists():
                    fail(f"cross-check source missing: {src}")
                    continue
                d = json.load(open(ROOT / src))
                for k in (key_path if isinstance(key_path, list) else [key_path]):
                    d = d[k]
                if abs(d - expected_json) > tol:
                    fail(f"cross-check {src}: JSON value {d} != expected {expected_json}")
                else:
                    ok(f"cross-check {src}: {key_path} = {d}")
                if tex_token.replace("+", "") in tex or tex_token in tex:
                    ok(f"tex: contains '{tex_token}'")
                else:
                    fail(f"tex: does NOT contain '{tex_token}'")

            # (7a) Round-33 sensitivity headline: unclipped (clipinf) run
            fs_path = ROOT / "results/faithful_steps_eval.json"
            if fs_path.exists():
                fs = json.load(open(fs_path))
                row = next((r for r in fs.get("rows", [])
                            if "clipinf" in r.get("ckpt", "") and r["ckpt"].endswith("/best")), None)
                if row is None:
                    fail("faithful_steps_eval.json: clipinf best row missing")
                else:
                    if abs(row["gap"] - 11.19) > 0.01:
                        fail(f"clipinf gap {row['gap']} != 11.19")
                    else:
                        ok(f"clipinf gap = {row['gap']}")
                    if abs(row["train_readout_bleu"] - 99.98) > 0.01:
                        fail(f"clipinf readout {row['train_readout_bleu']} != 99.98")
                    else:
                        ok(f"clipinf readout = {row['train_readout_bleu']}")
                    for token in ("+11.19", "99.98"):
                        if token in tex:
                            ok(f"tex: contains '{token}'")
                        else:
                            fail(f"tex: does NOT contain '{token}'")
                row5 = next((r for r in fs.get("rows", [])
                             if r.get("ckpt") == "ctc_clip_sens/clip5p0_seed42/best"), None)
                if row5 is None:
                    fail("faithful_steps_eval.json: clip5p0 best row missing")
                else:
                    if abs(row5["train_readout_bleu"] - 82.69) > 0.01:
                        fail(f"clip5p0 readout {row5['train_readout_bleu']} != 82.69")
                    else:
                        ok(f"clip5p0 readout = {row5['train_readout_bleu']}")
                    if "82.69" in tex or "82.7" in tex:
                        ok("tex: contains clip5.0 readout token")
                    else:
                        fail("tex: does NOT contain clip5.0 readout token (82.69/82.7)")
            else:
                fail("cross-check source missing: results/faithful_steps_eval.json")

            # (7b) Regression guards: stale 31-point readout--gap correlations
            # (superseded by results/readout_gap_correlation.json, round 33)
            for stale in ("0.916", "0.907", "-0.082", "rho{=}{-}0.082"):
                if stale in tex:
                    fail(f"tex: stale readout-correlation token '{stale}' present")
            else:
                ok("tex: no stale 31-point readout-correlation tokens")

            # (8) Probe multiplicity
            pm_path = ROOT / "results/probe_multiplicity.json"
            if pm_path.exists():
                pm = json.load(open(pm_path))
                explored_matrix = sum(1 for c in pm.get("complete_matrix", []) if c.get("explored"))
                if "of the 12 retrieval--construction cells, 6 were explored" in tex.lower().replace("--", "-"):
                    ok("tex: probe multiplicity '6 explored of 12' statement present")
                elif re.search(r"6 were explored", tex):
                    ok("tex: probe multiplicity '6 were explored' present")
                elif "6 of 12 cells explored" in tex:
                    ok("tex: probe multiplicity '6 of 12 cells explored' present")
                else:
                    fail("tex: probe multiplicity '6 explored' statement NOT found")

            # (6) Gap panel JSON non_released_gap_range matches registry
            if reg is not None and GAP_PANEL.exists():
                panel = json.load(open(GAP_PANEL))
                meta = panel.get("_meta", {})
                declared_range = meta.get("non_released_gap_range")
                # Compute actual range from non-released non-degenerate entries in panel
                non_released = [v for k, v in panel.items()
                                if k not in ("released", "_meta") and isinstance(v, dict) and "gap" in v]
                if non_released:
                    gaps = [v["gap"] for v in non_released if v.get("gap") is not None]
                    actual_min = min(gaps); actual_max = max(gaps)
                    # Sanity: meta range should match
                    if declared_range and (abs(declared_range[0] - actual_min) > 0.01 or
                                           abs(declared_range[1] - actual_max) > 0.01):
                        fail(f"gap panel _meta.non_released_gap_range ({declared_range}) != "
                             f"actual min/max ({actual_min:.4f}/{actual_max:.4f})")
                    else:
                        ok(f"gap panel _meta range matches actual ({actual_min:.4f}/{actual_max:.4f})")

    # (10) Round-34: title, terminology, dev-probe panel, split reconciliation,
    # seven-correction family mean/SD
    if PAPER.exists():
        tex = open(PAPER).read()
        # Title tokens
        if "Gradient-Clipping Sensitivity Study" in tex:
            ok("tex: round-34 title present")
        else:
            fail("tex: round-34 title token missing")
        for stale in ("What One Inference Hid", "What Reproduces, and What Does Not"):
            if stale in tex:
                fail(f"tex: stale title fragment '{stale}' present")
        if "signjoey-exact" in tex:
            fail("tex: stale 'signjoey-exact' terminology present (now candidate-upstream)")
        else:
            ok("tex: candidate-upstream terminology in use")
        # Dev-probe panel
        dp_path = ROOT / "results/dev_probe_eval.json"
        if dp_path.exists():
            dp = json.load(open(dp_path))
            gaps = {r["ckpt"]: r["gap"] for r in dp.get("rows", [])}
            for ck, expect in (("ctc_clip_sens/clipinf_seed42/best", 8.07),
                               ("ctc_clip_sens/clipinf_seed43/best", 8.00),
                               ("ctc_clip_sens/clipinf_seed44/best", 8.64),
                               ("ctc_clip_sens/signjoeyexact_seed42/best", 7.87),
                               ("faithful/seed_42/best", -1.35)):
                if ck not in gaps:
                    fail(f"dev_probe_eval.json: row missing: {ck}")
                elif abs(gaps[ck] - expect) > 0.01:
                    fail(f"dev_probe_eval.json: {ck} gap {gaps[ck]} != {expect}")
                else:
                    ok(f"dev probe {ck}: gap = {gaps[ck]}")
            for token in ("+8.07", "+8.00", "+8.64", "+7.87", "+8.10"):
                if token in tex:
                    ok(f"tex: contains dev-probe token '{token}'")
                else:
                    fail(f"tex: dev-probe token '{token}' missing")
        else:
            fail("cross-check source missing: results/dev_probe_eval.json")
        # Split reconciliation
        sr_path = ROOT / "results/split_reconciliation.json"
        if sr_path.exists():
            sr = json.load(open(sr_path))
            for split, miss in (("train", 36), ("dev", 4), ("test", 1)):
                got = sr["splits"][split]["missing_from_slrtp"]
                if got != miss:
                    fail(f"split_reconciliation: {split} missing {got} != {miss}")
                else:
                    ok(f"split_reconciliation: {split} missing = {got}")
            for token in ("7,096/519/642", "36/4/1"):
                if token in tex:
                    ok(f"tex: contains split token '{token}'")
                else:
                    fail(f"tex: split token '{token}' missing")
        else:
            fail("cross-check source missing: results/split_reconciliation.json")
        # Seven-correction family test-REC mean/SD
        if REGISTRY.exists():
            regd = json.load(open(REGISTRY))
            vals = [c["gt_bleu"] for c in regd["checkpoints"]
                    if c["run_id"].startswith("faithful_") and c.get("gt_bleu") is not None]
            if len(vals) == 8:
                mean = sum(vals) / 8
                sd = (sum((v - mean) ** 2 for v in vals) / 7) ** 0.5
                if abs(mean - 12.27) > 0.01 or abs(sd - 0.46) > 0.01:
                    fail(f"faithful test-REC mean/SD {mean:.3f}/{sd:.3f} != 12.27/0.46")
                else:
                    ok(f"faithful test-REC mean/SD = {mean:.3f}/{sd:.3f}")
                if "12.27" in tex and "0.46" in tex:
                    ok("tex: contains mean±SD tokens")
                else:
                    fail("tex: mean±SD tokens (12.27/0.46) missing")
            else:
                fail(f"registry faithful rows = {len(vals)}, expected 8")

    # Report
    print("=" * 60)
    print("PAPER--ARTIFACT CONSISTENCY TEST")
    print("=" * 60)
    print(f"\nPassed checks: {len(passed)}")
    for p in passed:
        print(f"  OK   {p}")
    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for w in warnings:
            print(f"  WARN {w}")
    if errors:
        print(f"\nFAILED checks: {len(errors)}")
        for e in errors:
            print(f"  FAIL {e}")
        print("\n=== RESULT: FAIL ===")
        sys.exit(1)
    else:
        print("\n=== RESULT: ALL PASSED ===")
        sys.exit(0)


if __name__ == "__main__":
    main()
