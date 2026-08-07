#!/usr/bin/env python3
"""E-F': propensity-weighted pool-origin estimate + caliper-support-effect curve.

The 605-factorial strata retain a similarity SMD of ~0.4 between seen (train-pool) and
unseen (test-pool) donors. Here we balance the 622-query matched comparison
(SEEN-PURE-MATCHED vs UNSEEN-PURE under the original evaluator) with inverse
propensity weighting: logistic propensity P(train-pool origin | LaBSE, Jaccard,
duration, word count), stabilized IPW, weighted corpus BLEU from per-item sufficient
statistics. Bootstrap (1000 replicates, propensity refit each time) for the weighted
pool-origin contrast. Caliper-support-effect curve: repeat with propensity-model
trimming (overlap region only, propensity in [0.1, 0.9] / [0.2, 0.8] / [0.3, 0.7]).
"""
import json, math, re
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "revision_20260729_round5"
EVAL_JSON = ROOT / "revision_20260728_round4/results/r5_common_support_eval.json"
BAL = R5 / "results/e3_balance.json"
OUT = R5 / "results/e12f_ipw.json"
B = 1000


def tokenize13a(text):
    text = text.replace("<skipped>", "").replace("-\n", "")
    text = " ".join(re.findall(r"\S+", text))
    text = re.sub(r"([\{-\~\[-\^\`\{-\}])", r" \1 ", text)
    text = re.sub(r"([\.\!\;\:\?\,])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower().split(" ") if text.strip() else []


def suff(hyp, ref):
    c = np.zeros(4); t = np.zeros(4)
    for n in range(1, 5):
        rng = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hng = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        c[n - 1] = sum(min(v, rng.get(g, 0)) for g, v in hng.items())
        t[n - 1] = max(len(hyp) - n + 1, 0)
    return c, t, len(hyp), len(ref)


def wbleu(S, w, idx):
    c = np.zeros(4); t = np.zeros(4); sl = 0.0; rl = 0.0
    for i in idx:
        cc, tt, s_, r_ = S[i]
        c += w[i] * cc; t += w[i] * tt; sl += w[i] * s_; rl += w[i] * r_
    if sl == 0 or t.sum() == 0:
        return 0.0
    bp = 1.0 if sl >= rl else math.exp(1 - rl / sl)
    p = [(0.5 / t[n] if t[n] and c[n] == 0 else (c[n] / t[n] if t[n] else 0.0)) for n in range(4)]
    if min(p) <= 0:
        return 0.0
    return float(bp * math.exp(sum(math.log(x) for x in p) / 4) * 100)


def fit_propensity(X, y):
    """Simple logistic regression via IRLS (no sklearn dependency assumptions)."""
    X1 = np.hstack([np.ones((len(X), 1)), (X - X.mean(0)) / X.std(0)])
    beta = np.zeros(X1.shape[1])
    for _ in range(50):
        z = X1 @ beta
        p_ = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        W = np.clip(p_ * (1 - p_), 1e-6, None)
        g = X1.T @ (y - p_)
        H = (X1.T * W) @ X1
        step = np.linalg.solve(H + 1e-4 * np.eye(len(beta)), g)
        beta += step
        if np.max(np.abs(step)) < 1e-6:
            break
    z = X1 @ beta
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ipw_contrast(idx, X, y, S_seen, S_unseen, trim=None):
    prop = fit_propensity(X[idx], y[idx])
    if trim:
        keep = (prop >= trim[0]) & (prop <= trim[1])
        idx = idx[keep[idx]]
        if len(idx) < 50:
            return None, len(idx)
        prop = fit_propensity(X[idx], y[idx])
    # stabilized IPW for ATE on the pooled (query, origin) pairs
    p1 = y[idx].mean()
    w = np.where(y[idx] == 1, p1 / np.clip(prop, 0.05, 0.95), (1 - p1) / np.clip(1 - prop, 0.05, 0.95))
    seen_rows = np.where(y[idx] == 1)[0]
    unseen_rows = np.where(y[idx] == 0)[0]
    seen_bleu = wbleu([S_seen[i] for i in idx], w, seen_rows)
    unseen_bleu = wbleu([S_unseen[i] for i in idx], w, unseen_rows)
    return seen_bleu - unseen_bleu, len(idx)


def main():
    d = json.loads(EVAL_JSON.read_text())
    ids = d["common_support_ids"]
    by_es = {(r["evaluator"], r["system"]): r["items"] for r in d["rows"]}
    seen_items = by_es[("original", "SEEN-PURE-MATCHED-v1")]
    unseen_items = by_es[("original", "UNSEEN-PURE-v1")]
    bal = json.load(open(BAL))
    cov = {r["query"]: r for r in bal["rows"]}

    X = np.array([[cov[q]["seen"]["labse"], cov[q]["seen"]["jaccard"],
                   cov[q]["seen"]["duration_s"], cov[q]["seen"]["words"]] for q in ids])
    y = np.ones(len(ids))
    X2 = np.array([[cov[q]["unseen"]["labse"], cov[q]["unseen"]["jaccard"],
                    cov[q]["unseen"]["duration_s"], cov[q]["unseen"]["words"]] for q in ids])
    # pooled (query, donor) observations: origin label 1 = train pool
    Xpool = np.vstack([X, X2])
    ypool = np.concatenate([y, np.zeros(len(ids))])
    S = []
    for si, ui in zip(seen_items, unseen_items):
        S.append((suff(tokenize13a(si["hypothesis"]), tokenize13a(si["reference"])),
                  suff(tokenize13a(ui["hypothesis"]), tokenize13a(ui["reference"]))))
    S_seen = [a[0] for a in S] + [a[0] for a in S]
    S_unseen = [a[1] for a in S] + [a[1] for a in S]
    # note: for unseen rows, S_seen/S_unseen indexing maps: seen donor decode = seen_items hyp,
    # unseen donor decode = unseen_items hyp. For IPW we weight each (query,origin) pair by
    # propensity of origin; the outcome for origin=1 is seen_items BLEU stats, for origin=0
    # is unseen_items BLEU stats. Stack: first 622 = origin 1, next 622 = origin 0.
    Y_stats = [a[0] for a in S] + [a[1] for a in S]

    def contrast(idx, trim=None):
        prop = fit_propensity(Xpool[idx], ypool[idx])
        if trim:
            keep = (prop >= trim[0]) & (prop <= trim[1])
            idx = idx[keep[idx]]
            if len(idx) < 50:
                return None, len(idx)
            prop = fit_propensity(Xpool[idx], ypool[idx])
        p1 = ypool[idx].mean()
        w = np.where(ypool[idx] == 1, p1 / np.clip(prop, 0.05, 0.95),
                     (1 - p1) / np.clip(1 - prop, 0.05, 0.95))
        seen_rows = np.where(ypool[idx] == 1)[0]
        unseen_rows = np.where(ypool[idx] == 0)[0]
        b1 = wbleu([Y_stats[i] for i in idx], w, seen_rows)
        b0 = wbleu([Y_stats[i] for i in idx], w, unseen_rows)
        return b1 - b0, len(idx)

    n_all = len(ypool)
    idx_all = np.arange(n_all)
    out = {"unweighted": {}, "ipw": {}, "trims": {}}
    # unweighted reference
    def unw(idx):
        seen_rows = np.where(ypool[idx] == 1)[0]
        unseen_rows = np.where(ypool[idx] == 0)[0]
        w1 = np.ones(len(idx))
        b1 = wbleu([Y_stats[i] for i in idx], w1, seen_rows)
        b0 = wbleu([Y_stats[i] for i in idx], w1, unseen_rows)
        return b1 - b0
    rng = np.random.default_rng(42)
    boots = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n_all, n_all)
        boots[b] = unw(idx)
    out["unweighted"] = {"point": unw(idx_all),
                         "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    pt, _ = contrast(idx_all)
    boots = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n_all, n_all)
        v, _ = contrast(idx)
        boots[b] = v if v is not None else np.nan
    out["ipw"] = {"point": pt, "ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]}
    for trim in [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]:
        pt_t, n_t = contrast(idx_all, trim=trim)
        boots = []
        for b in range(B):
            idx = rng.integers(0, n_all, n_all)
            v, _ = contrast(idx, trim=trim)
            if v is not None:
                boots.append(v)
        boots = np.array(boots)
        out["trims"][str(trim)] = {"n_retained": n_t, "point": pt_t,
                                   "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        print(trim, n_t, round(pt_t, 2), out["trims"][str(trim)]["ci"], flush=True)
    # balance after weighting
    prop = fit_propensity(Xpool, ypool)
    p1 = ypool.mean()
    w = np.where(ypool == 1, p1 / np.clip(prop, 0.05, 0.95), (1 - p1) / np.clip(1 - prop, 0.05, 0.95))
    bal_w = {}
    for j, name in enumerate(["labse", "jaccard", "duration", "words"]):
        x = Xpool[:, j]
        s1 = ypool == 1
        m1 = np.average(x[s1], weights=w[s1]); m0 = np.average(x[~s1], weights=w[~s1])
        v1 = np.average((x[s1] - m1) ** 2, weights=w[s1]); v0 = np.average((x[~s1] - m0) ** 2, weights=w[~s1])
        bal_w[name] = float((m1 - m0) / np.sqrt((v1 + v0) / 2))
    out["balance_after_ipw_smd"] = bal_w
    print("balance after IPW:", {k: round(v, 3) for k, v in bal_w.items()})
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
