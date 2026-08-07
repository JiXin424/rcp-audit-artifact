#!/usr/bin/env python3
"""Round 5 E3a: multivariate balance for the Jaccard-matched donor contrast (D).

For the 622-query common support, compares the SEEN-PURE-MATCHED-v1 (train-pool) donor
and the UNSEEN-PURE-v1 (test-pool) donor per query on:
  source-text Jaccard, LaBSE cosine, donor duration, donor word count,
  donor template density (gloss bigram-Jaccard to nearest train neighbor),
  same-signer indicator.
Reports mean/median/SD per group, standardized mean differences (paired),
and distributional overlap (shared-support coverage at fixed tolerances).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R4 = ROOT / "revision_20260728_round4/results/r4_seen_matched_registry.jsonl"
E21 = ROOT / "revision_20260728_round3/results/e2_1_exposure_registry.jsonl"
OUT = ROOT / "revision_20260729_round5/results/e3_balance.json"
LABSE = Path("/ssd/model/models/hub/models--sentence-transformers--LaBSE/snapshots/836121a0533e5664b21c7aacc5d22951f2b8b25b")
DATA = Path(__file__).resolve().parents[1] / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"

sys.path.insert(0, str(ROOT / "revision_20260728_major"))
from src import evaluate_checkpoints as ev  # noqa: E402


def norm_tokens(s: str) -> list[str]:
    return " ".join(s.casefold().split()).split()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def gloss_bigrams(g: str) -> set:
    t = norm_tokens(g)
    return {(t[i], t[i + 1]) for i in range(len(t) - 1)}


def main():
    seen_matched = {json.loads(l)["query_id"]: json.loads(l) for l in R4.open()
                    if json.loads(l)["system"] == "SEEN-PURE-MATCHED-v1"}
    unseen = {json.loads(l)["query_id"]: json.loads(l) for l in E21.open()
              if json.loads(l)["system"] == "UNSEEN-PURE-v1"}
    qids = sorted(set(seen_matched) & set(unseen))
    assert len(qids) == 622, len(qids)

    train = ev.safe_torch_load(ev.TRAIN_PT, ev.PINNED[str(ev.TRAIN_PT)], "train")
    test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")

    # Template density per donor: max gloss-bigram Jaccard to any OTHER train sequence.
    train_ids = sorted(train)
    train_bg = {k: gloss_bigrams(train[k]["gloss"]) for k in train_ids}

    def donor_density(donor_bg: set, self_id: str) -> float:
        best = 0.0
        for k in train_ids:
            if k == self_id:
                continue
            v = jaccard(donor_bg, train_bg[k])
            if v > best:
                best = v
        return best

    # Collect texts for LaBSE
    rows = []
    for q in qids:
        sd = seen_matched[q]["donor_id"]
        ud = unseen[q]["donor_id"]
        rows.append({"query": q, "seen_donor": sd, "unseen_donor": ud,
                     "seen_jaccard": seen_matched[q]["jaccard"],
                     "unseen_jaccard": unseen[q]["jaccard"]})

    texts = {}
    for r in rows:
        texts[r["query"]] = test[r["query"]]["text"]
        texts[r["seen_donor"]] = train[r["seen_donor"]]["text"]
        texts[r["unseen_donor"]] = test[r["unseen_donor"]]["text"]

    # LaBSE encode (CPU ok)
    import torch as th
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import load_file
    tok = AutoTokenizer.from_pretrained(LABSE, local_files_only=True)
    model = AutoModel.from_pretrained(LABSE, local_files_only=True).eval()
    dense_cfg = json.loads((LABSE / "2_Dense/config.json").read_text())
    state = load_file(str(LABSE / "2_Dense/model.safetensors"), device="cpu")
    dense = th.nn.Linear(dense_cfg["in_features"], dense_cfg["out_features"])
    dense.load_state_dict({"weight": state["linear.weight"], "bias": state["linear.bias"]})
    dense = dense.eval()

    keys = sorted(texts)
    embs = {}
    with th.no_grad():
        for i in range(0, len(keys), 64):
            chunk = [texts[k] for k in keys[i:i + 64]]
            inp = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            hidden = model(**inp).last_hidden_state
            mask = inp["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            emb = th.nn.functional.normalize(th.tanh(dense(pooled)), p=2, dim=1)
            for k, e in zip(keys[i:i + 64], emb):
                embs[k] = e.numpy()
            if i % 640 == 0:
                print(f"encoded {i}/{len(keys)}", flush=True)

    def cov(qid, donor_id, pool):
        src = train if pool == "train" else test
        d = src[donor_id]
        q = test[qid]
        return {
            "jaccard": jaccard(set(norm_tokens(q["text"])), set(norm_tokens(d["text"]))),
            "labse": float(np.dot(embs[qid], embs[donor_id])),
            "duration_s": d["poses_3d"].shape[0] / 25.0,
            "words": len(d["text"].split()),
            "density": donor_density(gloss_bigrams(d["gloss"]), donor_id if pool == "train" else "__test__"),
            "same_signer": bool(d.get("speaker") == q.get("speaker")),
        }

    out_rows = []
    for r in rows:
        cs = cov(r["query"], r["seen_donor"], "train")
        cu = cov(r["query"], r["unseen_donor"], "test")
        out_rows.append({"query": r["query"], "seen": cs, "unseen": cu})

    # Balance statistics
    stats = {}
    for key in ["jaccard", "labse", "duration_s", "words", "density", "same_signer"]:
        s = np.array([r["seen"][key] for r in out_rows], dtype=float)
        u = np.array([r["unseen"][key] for r in out_rows], dtype=float)
        diff = s - u
        sd_pooled = np.sqrt((s.var(ddof=1) + u.var(ddof=1)) / 2)
        stats[key] = {
            "seen_mean": float(s.mean()), "seen_median": float(np.median(s)), "seen_sd": float(s.std(ddof=1)),
            "unseen_mean": float(u.mean()), "unseen_median": float(np.median(u)), "unseen_sd": float(u.std(ddof=1)),
            "paired_diff_mean": float(diff.mean()), "paired_diff_sd": float(diff.std(ddof=1)),
            "smd_unpooled": float(diff.mean() / sd_pooled) if sd_pooled > 0 else 0.0,
            "smd_paired": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0,
        }
    overlap = {
        "abs_jaccard_diff_le_0p05": float(np.mean([abs(r["seen"]["jaccard"] - r["unseen"]["jaccard"]) <= 0.05 for r in out_rows])),
        "abs_jaccard_diff_le_0p10": float(np.mean([abs(r["seen"]["jaccard"] - r["unseen"]["jaccard"]) <= 0.10 for r in out_rows])),
        "abs_duration_diff_le_1s": float(np.mean([abs(r["seen"]["duration_s"] - r["unseen"]["duration_s"]) <= 1.0 for r in out_rows])),
        "abs_words_diff_le_2": float(np.mean([abs(r["seen"]["words"] - r["unseen"]["words"]) <= 2 for r in out_rows])),
        "both_same_signer": float(np.mean([r["seen"]["same_signer"] and r["unseen"]["same_signer"] for r in out_rows])),
        "seen_same_signer": float(np.mean([r["seen"]["same_signer"] for r in out_rows])),
        "unseen_same_signer": float(np.mean([r["unseen"]["same_signer"] for r in out_rows])),
    }
    OUT.write_text(json.dumps({"n": len(out_rows), "stats": stats, "overlap": overlap, "rows": out_rows}, indent=1))
    print(json.dumps({"stats": stats, "overlap": overlap}, indent=1))


if __name__ == "__main__":
    main()
