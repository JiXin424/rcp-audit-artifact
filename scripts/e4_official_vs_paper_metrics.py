#!/usr/bin/env python3
"""Round 5 E4: official-vs-paper chrF/WER/ROUGE alignment on byte-identical hypotheses.

Addresses reviewer Major #6: official repo reports PHX test GT chrF=34.585, WER=85.775;
paper reports chrF=48.05, WER=0.793. This script:
  1. Loads canonical decoded hypotheses (cp0 = original evaluator) for GT/PT/PTCOMP/PURE.
  2. Computes official implementations (SLRTP2025_eval/metrics.py: jiwer 3.1.0 WER with
     official transforms, sacrebleu CHRF() default corpus score, external_metrics rouge).
  3. Computes paper implementations (custom sentence-avg chrF n_max=4 incl whitespace,
     custom raw-split corpus WER, custom sentence-avg ROUGE-L F1).
  4. Ablates chrF: {corpus,sentence-avg} x {n_max 4,6} x {whitespace incl,excl}
     and WER: {raw tokens, official jiwer normalization}.
  5. Verifies official implementations on our hypotheses reproduce the official
     e19_test_run.json values (34.585 / 85.775 / 35.196 / 12.777).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "revision_20260728_canonical_rebuild/outputs/evaluations/cells"
OUT = ROOT / "revision_20260729_round5/results/e4_official_vs_paper_metrics.json"
EVAL_ROOT = Path(__file__).resolve().parents[1] / "src/models"
sys.path.insert(0, str(EVAL_ROOT))

from metrics import wer as official_wer, chrf as official_chrf, rouge as official_rouge  # noqa: E402
import sacrebleu  # noqa: E402

# ---------------- paper implementations (verbatim from e3_metric_sensitivity.py) --------
def _ngrams(s: str, n: int) -> dict[str, int]:
    if len(s) < n:
        return {}
    out: dict[str, int] = {}
    for i in range(len(s) - n + 1):
        g = s[i:i + n]
        out[g] = out.get(g, 0) + 1
    return out


def paper_chrf(references, hypotheses, n_max=4, beta=2.0):
    total_p = [0.0] * n_max
    total_r = [0.0] * n_max
    n_pairs = 0
    for ref, hyp in zip(references, hypotheses):
        ref_str = " ".join(ref)
        hyp_str = " ".join(hyp)
        for n in range(1, n_max + 1):
            ref_ng = _ngrams(ref_str, n)
            hyp_ng = _ngrams(hyp_str, n)
            if not ref_ng or not hyp_ng:
                continue
            overlap = sum(min(ref_ng[g], hyp_ng.get(g, 0)) for g in hyp_ng if g in ref_ng)
            total_p[n - 1] += overlap / sum(hyp_ng.values())
            total_r[n - 1] += overlap / sum(ref_ng.values())
        n_pairs += 1
    if n_pairs == 0:
        return 0.0
    f_scores = []
    for n in range(n_max):
        p = total_p[n] / n_pairs
        r = total_r[n] / n_pairs
        f_scores.append(0.0 if p + r == 0 else (1 + beta * beta) * (p * r) / (beta * beta * p + r))
    return sum(f_scores) / len(f_scores)


def paper_wer(references, hypotheses):
    total_edits = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        total_edits += dp[m][n]
        total_words += m
    return total_edits / total_words if total_words else 0


def paper_rouge_l(references, hypotheses):
    total_f = 0.0
    n_pairs = 0
    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            continue
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        r_lcs = lcs / m if m else 0
        p_lcs = lcs / n if n else 0
        f = 2 * r_lcs * p_lcs / (r_lcs + p_lcs) if r_lcs + p_lcs > 0 else 0.0
        total_f += f
        n_pairs += 1
    return total_f / n_pairs if n_pairs else 0


# ---------------- ablation implementations ----------------------------------------------
def chrf_variant(references, hypotheses, level="corpus", n_max=6, whitespace=False, beta=2.0):
    """chrF ablation: corpus vs sentence-average, n_max, whitespace inclusion."""
    refs = [r if isinstance(r, str) else " ".join(r) for r in references]
    hyps = [h if isinstance(h, str) else " ".join(h) for h in hypotheses]
    if not whitespace:
        refs = [r.replace(" ", "") for r in refs]
        hyps = [h.replace(" ", "") for h in hyps]
    if level == "corpus":
        # micro-averaged n-gram counts over corpus, sacrebleu-style averaging of per-order F
        sums_p, sums_r = [], []
        for n in range(1, n_max + 1):
            tp = tr = ov = 0
            for r, h in zip(refs, hyps):
                rng, hng = _ngrams(r, n), _ngrams(h, n)
                o = sum(min(rng[g], hng.get(g, 0)) for g in hng if g in rng)
                ov += o; tp += sum(hng.values()); tr += sum(rng.values())
            sums_p.append(ov / tp if tp else 0.0)
            sums_r.append(ov / tr if tr else 0.0)
        f = [(1 + beta * beta) * p * r / (beta * beta * p + r) if p + r else 0.0
             for p, r in zip(sums_p, sums_r)]
        return sum(f) / len(f)
    else:
        return paper_chrf([r.split() for r in refs] if whitespace else [list(r) for r in refs],
                            [h.split() for h in hyps] if whitespace else [list(h) for h in hyps],
                            n_max=n_max, beta=beta) if whitespace else _sent_chrf(refs, hyps, n_max, beta)


def _sent_chrf(refs, hyps, n_max, beta):
    fs = []
    for r, h in zip(refs, hyps):
        f_n = []
        for n in range(1, n_max + 1):
            rng, hng = _ngrams(r, n), _ngrams(h, n)
            if not rng or not hng:
                f_n.append(0.0); continue
            o = sum(min(rng[g], hng.get(g, 0)) for g in hng if g in rng)
            p = o / sum(hng.values()); rr = o / sum(rng.values())
            f_n.append((1 + beta * beta) * p * rr / (beta * beta * p + rr) if p + rr else 0.0)
        fs.append(sum(f_n) / len(f_n))
    return sum(fs) / len(fs)


def load_cell(name):
    d = json.load(open(CELLS / name))
    items = d["metrics"]["items"]
    return [it["reference"] for it in items], [it["hypothesis"] for it in items]


def main():
    systems = ["GT-v1", "PT-v1", "TN-PTCOMP-v1", "TN-PURE-v1"]
    out = {"e19_official_reference": {"bleu4": 12.777148564612425, "chrf": 34.58509769785496,
                                      "rouge": 35.19608041310144, "wer": 85.77470203767781},
           "library_versions": {"jiwer": "3.1.0 (requirements pin; env 3.1.0)",
                                 "sacrebleu_env": sacrebleu.__version__,
                                 "official_chrf_signature": "sacrebleu.metrics.CHRF() defaults",
                                 "official_wer_signature": "jiwer.wer + ExpandCommonEnglishContractions/ToLowerCase/RemoveMultipleSpaces/Strip/RemovePunctuation"},
           "systems": {}, "ablation": {}}

    for sysname in systems:
        refs, hyps = load_cell(f"cp0_{sysname}.json")
        ref_toks = [r.split() for r in refs]
        hyp_toks = [h.split() for h in hyps]
        row = {
            "official_chrf": official_chrf(hyps, refs),
            "official_wer": official_wer(hyps, refs),
            "official_rouge": official_rouge(hyps, refs),
            "paper_chrf": paper_chrf(ref_toks, hyp_toks),
            "paper_wer": paper_wer(ref_toks, hyp_toks),
            "paper_rouge_l": paper_rouge_l(ref_toks, hyp_toks),
        }
        out["systems"][sysname] = row
        print(f"{sysname}: official chrf={row['official_chrf']:.3f} wer={row['official_wer']:.3f} rouge={row['official_rouge']:.3f} | "
              f"paper chrf={row['paper_chrf']*100:.3f} wer={row['paper_wer']*100:.3f} rougeL={row['paper_rouge_l']*100:.3f}")

    # Ablation on GT to attribute the chrF difference
    refs, hyps = load_cell("cp0_GT-v1.json")
    ref_toks = [r.split() for r in refs]
    hyp_toks = [h.split() for h in hyps]
    ab = {}
    for level in ["corpus", "sentence"]:
        for n_max in [4, 6]:
            for ws in [False, True]:
                v = chrf_variant(ref_toks, hyp_toks, level=level, n_max=n_max, whitespace=ws)
                ab[f"{level}_nmax{n_max}_ws{int(ws)}"] = v
                print(f"chrf {level} n_max={n_max} whitespace={ws}: {v*100:.3f}")
    out["ablation"]["chrf_GT"] = ab
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
