#!/usr/bin/env python3
"""Regression test: paper numbers in main_lre.tex match the canonical JSON sources.

Assertions cover every number the LRE round-2 reviewers flagged (reviewer
questions 4-6): the exposure table (UNSEEN 8.51 / -4.27 / +14.51), the
matched-subset residual gap CI [-0.13, 1.99], the 43-entry canonical gap panel
with 0 positive gaps, the per-alpha distillation ladder gaps, and that the
stale values (8.86 / -3.92 / +14.16 / 10.81) do not reappear in the paper text.

Run: pytest tests/test_paper_numbers.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "results/gap_43_canonical_beam3.json"
PAPER = ROOT / "results/paper_numbers.json"
MATCHED = ROOT / "results/canonical_matched_subset.json"
# Paper sources live at the repo root locally, under paper/ in the artifact
MAIN_TEX = ROOT / "main_lre.tex"
SUPP_TEX = ROOT / "supplementary.tex"
if not MAIN_TEX.exists():
    MAIN_TEX = ROOT / "paper" / "main_lre.tex"
if not SUPP_TEX.exists():
    SUPP_TEX = ROOT / "paper" / "supplementary.tex"

TOL = 1e-6


def test_headline_matches_canonical():
    d = json.load(open(PAPER))
    h = d["headline"]
    assert abs(h["gt"] - 12.777) < 1e-3
    assert abs(h["pure"] - 23.020) < 1e-3
    assert abs(h["gap"] - 10.2433) < 1e-3
    ci = h["bootstrap_ci"]
    assert 8.88 <= ci[0] <= 8.90 and 11.60 <= ci[1] <= 11.64


def test_donor_origin_matches_exposure_section():
    d = json.load(open(PAPER))["donor_origin"]
    assert abs(d["seen_gap"] - 10.2433) < 1e-3
    assert abs(d["unseen_gap"] - (-4.27033)) < 1e-3
    assert abs(d["estimand"] - 14.51365) < 1e-3


def test_tex_exposure_numbers_match_json():
    tex = MAIN_TEX.read_text()
    d = json.load(open(PAPER))["donor_origin"]
    assert f"{d['unseen_gap']:.2f}" in tex.replace("$-4.27$", "-4.27")
    # exact tokens present in the exposure section
    assert "8.51" in tex
    assert "-4.27" in tex
    assert "14.51" in tex
    # stale values must not reappear (18.86 is the canonical PT-composed value)
    assert not re.search(r"(?<![0-9])8\.86", tex)
    assert not re.search(r"14\.16", tex)


def test_stale_distill_dev_value_not_in_tex():
    tex = MAIN_TEX.read_text()
    assert "10.81" not in tex


def test_alpha1_row_degenerate():
    tex = MAIN_TEX.read_text()
    assert re.search(r"1\.0 \(full KL\).*degenerate", tex, re.S)


def test_canonical_panel_counts_and_signs():
    d = json.load(open(PANEL))
    keys = [k for k in d if k != "_meta"]
    non_rel = [k for k in keys if k != "released"]
    assert len(non_rel) == 43, f"expected 43 non-released entries, got {len(non_rel)}"
    gaps = [d[k]["gap"] for k in non_rel]
    assert all(g <= 0 for g in gaps)
    assert max(gaps) < 1e-9
    assert min(gaps) <= -2.01 and min(gaps) >= -2.02


def test_distill_ladder_gaps_match_table10():
    d = json.load(open(PANEL))
    expect = {
        "a0.0": (-1.71, -0.80),
        "a0.25": (-2.01, -0.77),
        "a0.5": (-1.38, -0.77),
        "a0.75": (-1.21, -0.81),
        "a1.0": (0.0, 0.0),
    }
    for a, (lo, hi) in expect.items():
        gaps = [d[f"distill_{a}_{s}"]["gap"] for s in ("101", "202", "303")]
        assert (round(min(gaps), 2), round(max(gaps), 2)) == (lo, hi), (a, gaps)


def test_matched_subset_residual_gap_ci():
    d = json.load(open(MATCHED))
    assert abs(d["human_gap"] - 0.94975) < 1e-3
    ci = d["human_gap_bootstrap_ci"]
    assert ci[0] <= -0.13 <= ci[1] + 1e-3 or abs(ci[0] + 0.13) < 0.01
    assert -0.2 <= ci[0] <= -0.1, ci
    assert 1.9 <= ci[1] <= 2.1, ci
    # CI includes zero (main-text claim: residual indistinguishable from 0)
    assert ci[0] < 0 < ci[1]
    # attenuation CI still the published one
    assert abs(d["bootstrap_ci"][0] - 7.442) < 0.01
    assert abs(d["bootstrap_ci"][1] - 10.862) < 0.01


def test_residual_ci_stated_in_tex():
    tex = MAIN_TEX.read_text()
    assert "[-0.13, 1.99]" in tex


def test_sup_b_dev_column_marked_legacy():
    tex = SUPP_TEX.read_text()
    assert "dev BLEU-4$^{\\dagger}$" in tex
    assert "legacy protocol" in tex
