#!/usr/bin/env python3
"""Experiment C: per-item gap decomposition [mechanism story].

Reviewer R2-1 asks us to strengthen the mechanism narrative. We regress the
per-item sentence-level PURE--REC gap (released evaluator, canonical donor
registry) on interpretable item-level features to quantify how much of the
gap variance is attributable to lexical coupling, template density, signer,
text length, and donor reuse.

Features (per test item i):
  - jaccard_donor_ref   : token-set Jaccard between donor text and reference
                          (the lexical-coupling core; donor was selected by
                          Jaccard with the source sentence, which is the
                          speech-transcribed twin of the reference)
  - levenshtein_norm    : 1 - char-Levenshtein(donor, ref) / max(len)
  - exact_match_flag    : 1 if reference text exactly matches a training text
  - template_density    : # training texts sharing first-60-char prefix
  - high_overlap_train  : # training texts with Jaccard(ref, train_text) >= 0.5
  - text_length         : # whitespace tokens in reference
  - speaker_id          : signer index (9 levels; one-hot in regression)
  - donor_reuse_count   : # test queries whose donor == this item's donor
  - confidence_1_flag   : 1 if item is in Czehmann confidence=1 subset

Outcome y_i = sent_bleu(pure_hyp_i, ref_i) - sent_bleu(rec_hyp_i, ref_i)
        (sacrebleu sentence BLEU-4, 13a, exp smooth)

Output: results/per_item_gap_decomposition.json + figure.
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
import torch

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results/gap_43_canonical_beam3_items"
REG = ROOT / "results/gap_43_canonical_beam3_items/donor_registry.jsonl"
CSV_HC = ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
TEST_PT = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt"
TRAIN_PT = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt"
OUT_JSON = ROOT / "results/per_item_gap_decomposition.json"
OUT_FIG = ROOT / "generated_figures/per_item_gap_decomposition.pdf"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def normalize_text(s):
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def token_set(s):
    return set(normalize_text(s).split())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def char_lev_norm(a, b):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    # cheap Levenshtein via sacrebleu? use python-Levenshtein-free dynamic prog
    la, lb = len(a), len(b)
    if abs(la - lb) > 60:
        return 0.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return 1.0 - prev[lb] / m


def sent_bleu(hyp, ref):
    return BLEU.sentence_score(hyp, [ref]).score


def main():
    # Load
    gt_items = {x["id"]: x for x in json.load(open(ITEMS / "released_gt.json"))}
    pure_items = {x["id"]: x for x in json.load(open(ITEMS / "released_pure.json"))}
    registry = {}
    for line in open(REG):
        r = json.loads(line)
        registry[r["query_id"]] = r

    raw_test = torch.load(str(TEST_PT), map_location="cpu", weights_only=False)
    raw_train = torch.load(str(TRAIN_PT), map_location="cpu", weights_only=False)
    if isinstance(raw_train, dict):
        raw_train = list(raw_train.values())
    if isinstance(raw_test, dict):
        raw_test_dict = raw_test
    else:
        raw_test_dict = {it["name"]: it for it in raw_test}
    train_by_id = {it["name"]: it for it in raw_train}
    train_texts = [normalize_text(it["text"]) for it in raw_train]
    train_token_sets = [token_set(it["text"]) for it in raw_train]
    train_prefix60 = Counter(t[:60] for t in train_texts)

    # confidence=1 set
    hc_ids = set()
    for line in open(CSV_HC).read().splitlines()[1:]:
        parts = line.split("|")
        if len(parts) >= 3:
            try:
                if float(parts[2]) == 1.0:
                    hc_ids.add(parts[0])
            except ValueError:
                pass

    speakers = sorted({v.get("speaker", "unk") for v in raw_test_dict.values()})
    spk_idx = {s: i for i, s in enumerate(speakers)}

    donor_use = Counter(r["donor_id"] for r in registry.values())

    rows = []
    for qid in sorted(gt_items):
        if qid not in pure_items or qid not in registry:
            continue
        ref = gt_items[qid]["reference"]
        rec_hyp = gt_items[qid]["hypothesis"]
        pure_hyp = pure_items[qid]["hypothesis"]
        donor = train_by_id[registry[qid]["donor_id"]]
        donor_text = donor["text"]
        ref_norm = normalize_text(ref)
        ref_toks = token_set(ref)
        donor_toks = token_set(donor_text)
        # features
        feats = {
            "id": qid,
            "y_gap": sent_bleu(pure_hyp, ref) - sent_bleu(rec_hyp, ref),
            "rec_sent_bleu": sent_bleu(rec_hyp, ref),
            "pure_sent_bleu": sent_bleu(pure_hyp, ref),
            "jaccard_donor_ref": jaccard(donor_toks, ref_toks),
            "levenshtein_norm": char_lev_norm(donor_text, ref),
            "exact_match_flag": float(any(ref_norm == t for t in train_texts)),
            "template_density": float(train_prefix60.get(ref_norm[:60], 0)),
            "high_overlap_train": float(sum(1 for ts in train_token_sets
                                             if jaccard(ts, ref_toks) >= 0.5)),
            "text_length": float(len(ref.split())),
            "speaker": raw_test_dict[qid].get("speaker", "unk"),
            "speaker_id": float(spk_idx.get(raw_test_dict[qid].get("speaker", "unk"), -1)),
            "donor_reuse_count": float(donor_use[registry[qid]["donor_id"]]),
            "confidence_1_flag": float(qid in hc_ids),
            "retrieval_jaccard": float(registry[qid].get("jaccard", 0.0)),
        }
        rows.append(feats)

    n = len(rows)
    print(f"built {n} rows")

    # ---- OLS regression ----
    cont_feats = ["jaccard_donor_ref", "levenshtein_norm", "exact_match_flag",
                  "template_density", "high_overlap_train", "text_length",
                  "donor_reuse_count", "confidence_1_flag"]
    # standardize continuous features for comparable effect sizes
    def standardize(name):
        v = np.array([r[name] for r in rows], dtype=float)
        if v.std() > 0:
            return (v - v.mean()) / v.std()
        return v * 0.0

    speaker_levels = sorted({r["speaker"] for r in rows})
    spk_dummies = {s: np.array([float(r["speaker"] == s) for r in rows])
                   for s in speaker_levels[:-1]}  # drop last as reference

    X_cols = cont_feats + list(spk_dummies.keys())
    X = np.column_stack([standardize(f) for f in cont_feats]
                        + [spk_dummies[s] for s in speaker_levels[:-1]])
    X = np.column_stack([np.ones(n), X])
    y = np.array([r["y_gap"] for r in rows])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # leave-one-feature-out R2 drop (continuous only, plus speaker block)
    loo = {}
    for j, fname in enumerate(cont_feats):
        keep = [k for k in range(len(cont_feats)) if k != j]
        X_lo = np.column_stack(
            [np.ones(n)]
            + [standardize(cont_feats[k]) for k in keep]
            + [spk_dummies[s] for s in speaker_levels[:-1]]
        )
        b_lo, *_ = np.linalg.lstsq(X_lo, y, rcond=None)
        r_lo = y - X_lo @ b_lo
        ss_lo = float((r_lo ** 2).sum())
        loo[fname] = r2 - (1 - ss_lo / ss_tot)

    # speaker block contribution
    X_no_spk = np.column_stack([np.ones(n)] + [standardize(f) for f in cont_feats])
    b_ns, *_ = np.linalg.lstsq(X_no_spk, y, rcond=None)
    r_ns = y - X_no_spk @ b_ns
    loo["__speaker_block__"] = r2 - (1 - float((r_ns ** 2).sum()) / ss_tot)

    # exact-match stratified gap means
    em = [r["y_gap"] for r in rows if r["exact_match_flag"] > 0.5]
    nem = [r["y_gap"] for r in rows if r["exact_match_flag"] <= 0.5]

    result = {
        "schema": "per-item-gap-decomposition-C-v1",
        "n": n,
        "y_mean": float(y.mean()), "y_std": float(y.std()),
        "y_min": float(y.min()), "y_max": float(y.max()),
        "r2_full": float(r2),
        "coefficients_standardized": {
            cont_feats[j]: float(beta[1 + j]) for j in range(len(cont_feats))},
        "speaker_coeffs": {s: float(beta[1 + len(cont_feats) + i])
                           for i, s in enumerate(speaker_levels[:-1])},
        "leave_one_out_r2_drop": {k: float(v) for k, v in loo.items()},
        "exact_match_gap_mean": float(np.mean(em)) if em else None,
        "non_exact_match_gap_mean": float(np.mean(nem)) if nem else None,
        "n_exact_match": len(em),
        "per_item": rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))

    print(f"\ny (per-item gap): mean={result['y_mean']:.2f} "
          f"sd={result['y_std']:.2f} range=[{result['y_min']:.2f},"
          f"{result['y_max']:.2f}]")
    print(f"R^2 (full model) = {r2:.3f}")
    print("\nStandardized coefficients (effect on gap in BLEU points per SD):")
    for f in cont_feats:
        c = result["coefficients_standardized"][f]
        d = result["leave_one_out_r2_drop"][f]
        print(f"  {f:24s} beta*={c:+7.2f}  R2_drop={d:+.3f}")
    print(f"  {'__speaker_block__':24s}               R2_drop="
          f"{result['leave_one_out_r2_drop']['__speaker_block__']:+.3f}")
    print(f"\nexact-match items (n={len(em)}): mean gap={np.mean(em):.2f}")
    print(f"non-exact-match (n={len(nem)}): mean gap={np.mean(nem):.2f}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    order = sorted(cont_feats,
                   key=lambda f: -result["leave_one_out_r2_drop"][f])
    drops = [result["leave_one_out_r2_drop"][f] for f in order]
    ax1.barh(range(len(order)), drops, color="#2166ac")
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(order)
    ax1.invert_yaxis()
    ax1.axvline(0, color="k", lw=0.6)
    ax1.set_xlabel("$R^2$ drop when feature is removed")
    ax1.set_title(f"(a) Per-feature explanatory power  (full $R^2$={r2:.2f})")

    x = np.array([r["jaccard_donor_ref"] for r in rows])
    yv = np.array([r["y_gap"] for r in rows])
    em_f = np.array([r["exact_match_flag"] > 0.5 for r in rows])
    ax2.scatter(x[~em_f], yv[~em_f], s=10, alpha=0.4, color="#4d9221",
                label=f"non-exact-match (n={(~em_f).sum()})")
    ax2.scatter(x[em_f], yv[em_f], s=22, alpha=0.7, color="#b2182b",
                label=f"exact-match train text (n={em_f.sum()})")
    # binned mean line
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(x, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
    centers = (bins[:-1] + bins[1:]) / 2
    means = []
    for i in range(len(centers)):
        sel = bin_idx == i
        means.append(yv[sel].mean() if sel.sum() > 5 else np.nan)
    ax2.plot(centers, means, "k-o", ms=4, lw=1.2, label="bin mean")
    ax2.axhline(0, color="k", lw=0.6, ls="--")
    ax2.set_xlabel("donor--reference token-set Jaccard")
    ax2.set_ylabel("per-item gap (PURE $-$ REC, sentence BLEU)")
    ax2.set_title("(b) Gap vs lexical donor--reference overlap")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    print(f"\nsaved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
