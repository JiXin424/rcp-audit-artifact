#!/usr/bin/env python3
"""Retrieval-query decoupling control (round 33, reviewer point 4).

Question: is the released evaluator's positive probe response dependent on
selecting donors with the SCORED reference text specifically, or on any
lexically competent text-based selection?

Design (no GPU; released-evaluator hypotheses for all 7,060 training-pool
poses are cached in results/full_readout/backTranslation_PHIX_model.json):

  2x2 on the matched 461-item confidence=1 Czehmann subset:
    query text  in {original reference, human back-translation}
    scoring ref in {original reference, human back-translation}
  The query=original row reproduces the published matched-subset cells
  (REC 14.94 / PURE 24.98 vs REC 7.35 / PURE 8.30); the query=human row is
  new. Retrieval = same source-Jaccard argmax with the canonical
  exact-normalized-text exclusion (donors whose normalized text equals the
  query's ORIGINAL reference text are excluded, matching the canonical
  registry rule).

  Controls on the full 641 queries: top-k-random-within-Jaccard (k=5,20) and
  uniform-random donors, scored against original references; plus
  donor-selection diagnostics (achieved Jaccard to original and human query
  texts, donor overlap between the two retrievers).

Output: results/retrieval_query_decoupling.json
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT / "results/full_readout/backTranslation_PHIX_model.json"
ITEMS_DIR = ROOT / "results/gap_43_canonical_beam3_items"
CSV_HC = ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
OUT = ROOT / "results/retrieval_query_decoupling.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s.strip().lower())


def toks(s: str) -> set:
    return set(norm_text(s).split())


def jac(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def load_human_refs():
    """Czehmann subset CSV -> {id: (back-translation, confidence)}."""
    out = {}
    for line in CSV_HC.read_text().splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        try:
            conf = float(parts[2])
        except ValueError:
            conf = 0.0
        out[parts[0]] = (parts[1], conf)
    return out


def main():
    ro = json.load(open(READOUT))
    train_items = ro["splits"]["train"]["per_item"]
    rec = json.load(open(ITEMS_DIR / "released_gt.json"))
    pure = json.load(open(ITEMS_DIR / "released_pure.json"))
    rec_map = {it["id"]: it for it in rec}
    pure_map = {it["id"]: it for it in pure}

    query_ids = [it["id"] for it in rec]
    refs_orig = {it["id"]: it["reference"] for it in rec}
    human = load_human_refs()
    ids461 = [q for q in query_ids
              if q in human and human[q][1] == 1.0]
    refs_human = {q: human[q][0] for q in ids461}
    print(f"Queries: {len(query_ids)} | matched conf=1 subset: {len(ids461)}")

    train_ids = [t["id"] for t in train_items]
    train_texts = [t["ref"] for t in train_items]
    train_hyps = {t["id"]: t["hyp"] for t in train_items}
    train_tok = [toks(t) for t in train_texts]
    train_norm = [norm_text(t) for t in train_texts]
    n_tr = len(train_items)

    # Precompute Jaccard rows for the two query texts on the 461 subset.
    _row_cache = {}

    def jac_row(q_text):
        if q_text not in _row_cache:
            qt = toks(q_text)
            _row_cache[q_text] = np.array([jac(qt, tt) for tt in train_tok],
                                          dtype=np.float32)
        return _row_cache[q_text]

    def select_donors(qids, qtext_of, exclude_norm_of):
        """Argmax Jaccard with canonical exact-normalized-text exclusion."""
        donors = {}
        for q in qids:
            row = jac_row(qtext_of(q))
            excluded = exclude_norm_of(q)
            mask = np.array([tn != excluded for tn in train_norm])
            row = np.where(mask, row, -1.0)
            donors[q] = int(row.argmax())
        return donors

    # --- 2x2 on the 461 subset ---
    don_orig = select_donors(
        ids461, lambda q: refs_orig[q], lambda q: norm_text(refs_orig[q]))
    don_hum = select_donors(
        ids461, lambda q: refs_human[q], lambda q: norm_text(refs_orig[q]))

    def score(ids, hyps_of, ref_of):
        h = [hyps_of(q) for q in ids]
        r = [ref_of(q) for q in ids]
        return BLEU.corpus_score(h, [r]).score

    cells = {}
    cells["REC_query_na_score_orig"] = score(
        ids461, lambda q: rec_map[q]["hypothesis"], lambda q: refs_orig[q])
    cells["REC_query_na_score_human"] = score(
        ids461, lambda q: rec_map[q]["hypothesis"], lambda q: refs_human[q])
    # query=original: canonical donors (published cells) -- recompute for internal consistency
    cells["PURE_query_orig_score_orig"] = score(
        ids461, lambda q: pure_map[q]["hypothesis"], lambda q: refs_orig[q])
    cells["PURE_query_orig_score_human"] = score(
        ids461, lambda q: pure_map[q]["hypothesis"], lambda q: refs_human[q])
    # query=human: NEW
    cells["PURE_query_human_score_orig"] = score(
        ids461, lambda q: train_hyps[train_ids[don_hum[q]]], lambda q: refs_orig[q])
    cells["PURE_query_human_score_human"] = score(
        ids461, lambda q: train_hyps[train_ids[don_hum[q]]], lambda q: refs_human[q])

    grid = {
        "subset_n": len(ids461),
        "score_orig": {
            "REC": cells["REC_query_na_score_orig"],
            "PURE_query_orig": cells["PURE_query_orig_score_orig"],
            "PURE_query_human": cells["PURE_query_human_score_orig"],
            "gap_query_orig": cells["PURE_query_orig_score_orig"] - cells["REC_query_na_score_orig"],
            "gap_query_human": cells["PURE_query_human_score_orig"] - cells["REC_query_na_score_orig"],
        },
        "score_human": {
            "REC": cells["REC_query_na_score_human"],
            "PURE_query_orig": cells["PURE_query_orig_score_human"],
            "PURE_query_human": cells["PURE_query_human_score_human"],
            "gap_query_orig": cells["PURE_query_orig_score_human"] - cells["REC_query_na_score_human"],
            "gap_query_human": cells["PURE_query_human_score_human"] - cells["REC_query_na_score_human"],
        },
    }
    print("\n2x2 grid (461-item subset):")
    for frame, g in [("score_orig", grid["score_orig"]), ("score_human", grid["score_human"])]:
        print(f"  {frame}: REC {g['REC']:.2f} | Q-orig PURE {g['PURE_query_orig']:.2f} "
              f"(gap {g['gap_query_orig']:+.2f}) | Q-human PURE {g['PURE_query_human']:.2f} "
              f"(gap {g['gap_query_human']:+.2f})")

    # --- donor-selection diagnostics ---
    same_donor = np.mean([don_orig[q] == don_hum[q] for q in ids461])
    jac_to_orig_orig = np.mean([jac_row(refs_orig[q])[don_orig[q]] for q in ids461])
    jac_to_orig_hum = np.mean([jac_row(refs_orig[q])[don_hum[q]] for q in ids461])
    jac_to_hum_hum = np.mean([jac_row(refs_human[q])[don_hum[q]] for q in ids461])
    jac_to_hum_orig = np.mean([jac_row(refs_human[q])[don_orig[q]] for q in ids461])
    # donor text vs SCORING original reference, per selector
    donor_text_jac_orig = {
        "query_orig": float(np.mean([jac(toks(refs_orig[q]), train_tok[don_orig[q]]) for q in ids461])),
        "query_human": float(np.mean([jac(toks(refs_orig[q]), train_tok[don_hum[q]]) for q in ids461])),
    }
    diag = {
        "frac_same_donor_both_queries": float(same_donor),
        "mean_jac_query_orig_to_donor_of_query_orig_selector": float(jac_to_orig_orig),
        "mean_jac_query_orig_to_donor_of_query_human_selector": float(jac_to_orig_hum),
        "mean_jac_query_human_to_donor_of_query_human_selector": float(jac_to_hum_hum),
        "mean_jac_query_human_to_donor_of_query_orig_selector": float(jac_to_hum_orig),
        "mean_jac_scoring_orig_ref_to_donor": donor_text_jac_orig,
    }
    print("\nDiagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v if isinstance(v, float) else v}")

    # --- paired bootstrap for the two new gaps (descriptive) ---
    rng = np.random.RandomState(2026)
    N = len(ids461)
    hyps = {
        "rec": {q: rec_map[q]["hypothesis"] for q in ids461},
        "q_orig": {q: pure_map[q]["hypothesis"] for q in ids461},
        "q_hum": {q: train_hyps[train_ids[don_hum[q]]] for q in ids461},
    }
    boots = {k: [] for k in ["gap_q_orig_s_orig", "gap_q_hum_s_orig",
                             "gap_q_orig_s_human", "gap_q_hum_s_human",
                             "contrast_q_hum_minus_q_orig_s_orig"]}
    for b in range(10000):
        idx = rng.randint(0, N, N)
        qs = [ids461[i] for i in idx]
        ro_ = [refs_orig[q] for q in qs]
        rh_ = [refs_human[q] for q in qs]
        rec_h = [hyps["rec"][q] for q in qs]
        qo_h = [hyps["q_orig"][q] for q in qs]
        qh_h = [hyps["q_hum"][q] for q in qs]
        g_oo = BLEU.corpus_score(qo_h, [ro_]).score - BLEU.corpus_score(rec_h, [ro_]).score
        g_ho = BLEU.corpus_score(qh_h, [ro_]).score - BLEU.corpus_score(rec_h, [ro_]).score
        g_oh = BLEU.corpus_score(qo_h, [rh_]).score - BLEU.corpus_score(rec_h, [rh_]).score
        g_hh = BLEU.corpus_score(qh_h, [rh_]).score - BLEU.corpus_score(rec_h, [rh_]).score
        boots["gap_q_orig_s_orig"].append(g_oo)
        boots["gap_q_hum_s_orig"].append(g_ho)
        boots["gap_q_orig_s_human"].append(g_oh)
        boots["gap_q_hum_s_human"].append(g_hh)
        boots["contrast_q_hum_minus_q_orig_s_orig"].append(g_ho - g_oo)
    boot_ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
               for k, v in boots.items()}

    # --- full-641 controls: top-k random within Jaccard; uniform random ---
    print("\nFull-641 controls...", flush=True)
    rows_orig = {}
    for q in query_ids:
        rows_orig[q] = jac_row(refs_orig[q])
    refs_full = [refs_orig[q] for q in query_ids]
    rec_full = [rec_map[q]["hypothesis"] for q in query_ids]
    rec_bleu_full = BLEU.corpus_score(rec_full, [refs_full]).score
    pure_full = [pure_map[q]["hypothesis"] for q in query_ids]
    pure_bleu_full = BLEU.corpus_score(pure_full, [refs_full]).score

    def topk_random_gaps(k, n_draws=10):
        gaps = []
        for d in range(n_draws):
            r2 = np.random.RandomState(1000 + d)
            hsel = []
            for q in query_ids:
                row = rows_orig[q].copy()
                row[[j for j in range(n_tr) if train_norm[j] == norm_text(refs_orig[q])]] = -1.0
                top = np.argpartition(-row, k)[:k]
                hsel.append(train_hyps[train_ids[int(r2.choice(top))]])
            pb = BLEU.corpus_score(hsel, [refs_full]).score
            gaps.append(pb - rec_bleu_full)
        return {"k": k, "gaps": gaps, "mean": float(np.mean(gaps)),
                "min": float(np.min(gaps)), "max": float(np.max(gaps))}

    def random_gaps(n_draws=10):
        gaps = []
        for d in range(n_draws):
            r2 = np.random.RandomState(2000 + d)
            hsel = [train_hyps[train_ids[int(r2.randint(n_tr))]] for q in query_ids]
            pb = BLEU.corpus_score(hsel, [refs_full]).score
            gaps.append(pb - rec_bleu_full)
        return {"gaps": gaps, "mean": float(np.mean(gaps)),
                "min": float(np.min(gaps)), "max": float(np.max(gaps))}

    topk5 = topk_random_gaps(5)
    topk20 = topk_random_gaps(20)
    rnd = random_gaps()
    print(f"  top-5 random: mean gap {topk5['mean']:+.2f} | top-20: {topk20['mean']:+.2f} "
          f"| uniform random: {rnd['mean']:+.2f}")

    out = {
        "schema": "retrieval-query-decoupling-v1",
        "generated_by": "scripts/e_retrieval_query_decoupling.py",
        "design": ("2x2 retrieval-query x scoring-frame grid on the 461-item "
                   "confidence=1 Czehmann subset, rescored from the cached "
                   "released-evaluator training-pool decodes (no GPU). "
                   "Retrieval: source-Jaccard argmax with canonical "
                   "exact-normalized-text exclusion (excluded text = the "
                   "query's ORIGINAL reference, matching the canonical "
                   "registry rule in both arms)."),
        "grid_2x2": grid,
        "diagnostics": diag,
        "bootstrap_ci95_descriptive": boot_ci,
        "full_641_controls": {
            "rec_bleu": rec_bleu_full,
            "pure_query_orig_bleu": pure_bleu_full,
            "gap_query_orig": pure_bleu_full - rec_bleu_full,
            "topk_random_within_jaccard_k5": topk5,
            "topk_random_within_jaccard_k20": topk20,
            "uniform_random": rnd,
        },
        "interpretation": (
            "Scoring frame and selection query are partially separable: see "
            "grid. If the query=human gap against ORIGINAL references remains "
            "large, the positive response does not require selecting donors "
            "with the scored reference specifically -- any lexically "
            "competent text retrieval suffices; the coupling is to lexical "
            "similarity in general, not to the exact scored text. If it "
            "collapses, coupling to the exact scored reference is implicated. "
            "Either outcome weakens the earlier 'entirely dependent' phrasing "
            "in a specific, data-supported direction."
        ),
    }
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
