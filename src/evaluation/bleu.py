#!/usr/bin/env python3
"""sacreBLEU wrapper for the audit paper's canonical signature.

Paper signature: BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1
"""
from __future__ import annotations
from typing import List, Dict
import sacrebleu


def corpus_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Compute the canonical paper-signature corpus BLEU-4.

    Args:
        hypotheses: list of decoded hypothesis strings (already detokenized).
        references: list of reference strings (same length as hypotheses).

    Returns:
        Dict with keys: bleu, bleu_1, bleu_2, bleu_3, bleu_4, brevity_penalty,
        hyp_len, ref_len, signature.
    """
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize="13a",
        smooth_method="exp",
        force=True,
        lowercase=False,
        use_effective_order=False,
    )
    score = bleu.score
    # Decompose into n-gram precisions (sacrebleu 2.x returns 0-100 percentages)
    p1, p2, p3, p4 = bleu.precisions
    # Hard-coded signature (paper convention)
    signature = "BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:" + sacrebleu.__version__
    return {
        "bleu": score,
        "bleu_1": float(p1),
        "bleu_2": float(p2),
        "bleu_3": float(p3),
        "bleu_4": float(p4),
        "brevity_penalty": bleu.bp,
        "hyp_len": bleu.sys_len,
        "ref_len": bleu.ref_len,
        "signature": signature,
    }


def corpus_chrf(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """chrF (corpus-level, sacreBLEU defaults)."""
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"chrf": chrf.score, "signature": "chrF2|" + sacrebleu.__version__}


def corpus_wer(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Official jiwer 3.1.0 normalized WER (matches SLRTP2025 official scorer).

    Normalization: lowercase, remove punctuation. Same as the released
    official scorer (see main_mmsys.tex §3.5 metric implementations).
    """
    import re
    import jiwer

    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    h_norm = [norm(h) for h in hypotheses]
    r_norm = [norm(r) for r in references]
    wer = jiwer.wer(r_norm, h_norm)
    return {"wer": wer * 100, "wer_normalized": wer}
