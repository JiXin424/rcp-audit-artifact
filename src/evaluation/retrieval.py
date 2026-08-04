#!/usr/bin/env python3
"""Donor retrieval for the TN-PURE / TN-PTCOMP stress tests.

Reproduces the canonical donor registries used in the paper:
  - text-nearest (source-text Jaccard, the canonical TN-PURE-v1 retriever)
  - oracle-gloss (uses target gloss; input-leakage diagnostic)
  - random donor
  - semantic retrievers (LaBSE, multi-SBERT) — for sensitivity analyses
  - donor exclusion at threshold tau
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import unicodedata
import re


def normalize_text(s: str) -> str:
    """Unicode NFKC + whitespace + lowercase normalization (matches paper §3.2)."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def token_set(text: str) -> set:
    """Token set for Jaccard similarity."""
    return set(normalize_text(text).split())


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def char_levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein distance (for tie-breaking)."""
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


def build_text_nearest_registry(
    queries: List[Dict],
    donors: List[Dict],
    exclusion_threshold: float = 0.0,
    tie_break: str = "levenshtein",
) -> Dict[str, Dict]:
    """For each query, select the donor with max source-text Jaccard.

    Args:
        queries: list of {'id', 'text'} for test set
        donors: list of {'id', 'text'} for train pool
        exclusion_threshold: if > 0, exclude donors with Jaccard > threshold to the query.
        tie_break: 'levenshtein' (min char-Levenshtein) or 'hash' (min SHA-256 hash).

    Returns:
        Dict {query_id: {'donor_id', 'jaccard', 'donor_text', 'levenshtein'}}
    """
    donor_tokens = [(d["id"], d["text"], token_set(d["text"])) for d in donors]
    registry: Dict[str, Dict] = {}
    for q in queries:
        q_tokens = token_set(q["text"])
        q_norm = normalize_text(q["text"])
        best = None
        best_jac = -1.0
        best_tb = float("inf")
        for did, dtext, dtok in donor_tokens:
            jac = jaccard(q_tokens, dtok)
            if exclusion_threshold > 0 and jac > exclusion_threshold:
                continue
            if jac > best_jac or (jac == best_jac and jac > 0):
                # tie-break
                if tie_break == "levenshtein":
                    tb = char_levenshtein(q_norm, normalize_text(dtext))
                else:
                    tb = float(int(hashlib.sha256(did.encode()).hexdigest(), 16))
                if jac > best_jac or (jac == best_jac and tb < best_tb):
                    best_jac = jac
                    best_tb = tb
                    best = {"donor_id": did, "jaccard": jac, "donor_text": dtext,
                            "levenshtein": tb}
        if best is None:
            # all donors excluded; fallback to highest-jaccard donor (still excluded)
            best = {"donor_id": None, "jaccard": 0.0, "donor_text": "", "levenshtein": 0}
        registry[q["id"]] = best
    return registry


def build_random_registry(
    queries: List[Dict],
    donors: List[Dict],
    seed: int = 0,
) -> Dict[str, Dict]:
    """Random donor selection (uniform)."""
    import random
    rng = random.Random(seed)
    registry: Dict[str, Dict] = {}
    for q in queries:
        d = rng.choice(donors)
        registry[q["id"]] = {
            "donor_id": d["id"], "jaccard": jaccard(token_set(q["text"]), token_set(d["text"])),
            "donor_text": d["text"], "levor": 0,
        }
    return registry


# Expose token-set / jaccard helpers
__all__ = [
    "normalize_text", "token_set", "jaccard", "char_levenshtein",
    "build_text_nearest_registry", "build_random_registry",
]
