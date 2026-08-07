#!/usr/bin/env python3
"""E-C': system-set ranking stability across evaluators.

Decodes the remaining system x seed cells and computes, for the 9-system fixed set
(GT, PT, TN-PURE, TN-PTCOMP, LaBSE, SBERT, oracle-gloss-composed, UNSEEN-PURE, random),
Kendall tau-b, Spearman rho, and pairwise flip rates between each evaluator pair,
plus per-evaluator rankings. PT and UNSEEN-PURE are the "does-not-copy-training-poses"
controls; TN-PURE/TN-PTCOMP copy training-pool poses by construction.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

MAJOR = Path(__file__).resolve().parents[1] / "results"
FULL = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(MAJOR)); sys.path.insert(0, str(FULL))
from src import evaluate_checkpoints as ev  # noqa: E402
import src.evaluate_checkpoints as evf  # noqa: E402

R5 = Path(__file__).resolve().parents[1] / "results"
CELLS = Path(__file__).resolve().parents[1] / "data/cells"
OUT = R5 / "results/e12c_rank_stability.json"
DEC_DIR = R5 / "results/e12c_cells"
DEC_DIR.mkdir(exist_ok=True)

SEEDS = [101, 202, 303, 404, 505, 606]
SEED_CKPTS = {s: MAJOR / f"checkpoints/v2_seed_{s}/best.ckpt" for s in SEEDS}
SYS_POSES = {
    "LaBSE": (str(Path(__file__).resolve().parents[1] / "results/labse_public_seq/slrtp_labse_donor_copy_test.pt"), True),
    "SBERT": (str(Path(__file__).resolve().parents[1] / "results/msbert_public_seq/slrtp_msbert_donor_copy_test.pt"), True),
    "ORACLE-COMP": (str(Path(__file__).resolve().parents[1] / "results/slrtp_rcp4_upper_facehands_upperdyn_a0p88_test.pt"), False),
    "UNSEEN-PURE": (str(Path(__file__).resolve().parents[1] / "data/cells") + "/UNSEEN-PURE-v1.pt", False),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def decode_missing():
    torch.cuda.set_device(0)
    device = "cuda:0"
    test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")
    ids = sorted(test)
    texts = {x: test[x]["text"] for x in ids}
    # random donor: build frozen random selection (seed 0) from train pool
    import random
    train = ev.safe_torch_load(ev.TRAIN_PT, ev.PINNED[str(ev.TRAIN_PT)], "train")
    train_ids = sorted(train)
    rng = random.Random(0)
    rand_poses = {}
    for q in ids:
        cand = [t for t in train_ids if t != q]
        rand_poses[q] = train[rng.choice(cand)]["poses_3d"][::2]
    _, txt, model = ev._vocab_and_model(device)
    for seed in SEEDS:
        ck = SEED_CKPTS[seed]
        payload = ev.safe_torch_load(ck, sha256_file(ck), "checkpoint")
        model.load_state_dict(payload["model"]); model.to(device).eval()
        out = {}
        for sysname, (path, subsample) in SYS_POSES.items():
            fp = DEC_DIR / f"seed{seed}_{sysname}.json"
            if fp.exists():
                m = json.load(open(fp))
            else:
                data = torch.load(path, map_location="cpu", weights_only=True)
                poses = {}
                for k in ids:
                    t = data[k] if isinstance(data[k], torch.Tensor) else data[k]["poses_3d"]
                    poses[k] = t[::2] if subsample else t
                m = ev.evaluate_pose_set(model, txt, ids, poses, texts, device, 48, False)
                fp.write_text(json.dumps({"bleu": m["decoded_bleu"]}))
            out[sysname] = m["decoded_bleu"]
        fp = DEC_DIR / f"seed{seed}_RAND.json"
        if fp.exists():
            m = json.load(open(fp))
        else:
            m = ev.evaluate_pose_set(model, txt, ids, rand_poses, texts, device, 48, False)
            fp.write_text(json.dumps({"bleu": m["decoded_bleu"]}))
        out["RAND"] = m["decoded_bleu"]
        print(seed, {k: round(v * 100, 2) for k, v in out.items()}, flush=True)


def analyze():
    from scipy import stats as sst
    SYSTEMS = ["GT", "PT", "TN-PURE", "TN-PTCOMP", "LaBSE", "SBERT", "ORACLE-COMP", "UNSEEN-PURE", "RAND"]
    evals = ["original"] + [f"seed_{s}" for s in SEEDS]
    # scores: canonical cells for GT/PT/PURE/PTCOMP; decoded cells for the rest
    scores = {e: {} for e in evals}
    for i, e in enumerate(evals):
        for cp_sys, name in [("GT-v1", "GT"), ("PT-v1", "PT"), ("TN-PURE-v1", "TN-PURE"), ("TN-PTCOMP-v1", "TN-PTCOMP")]:
            d = json.load(open(CELLS / f"cp{i}_{cp_sys}.json"))
            scores[e][name] = d["metrics"]["decoded_bleu"] * 100
    scores["original"]["LaBSE"] = 19.15
    scores["original"]["SBERT"] = 16.50
    scores["original"]["ORACLE-COMP"] = 11.78
    scores["original"]["UNSEEN-PURE"] = 8.86
    scores["original"]["RAND"] = 0.90
    for seed in SEEDS:
        e = f"seed_{seed}"
        for sysname in ["LaBSE", "SBERT", "ORACLE-COMP", "UNSEEN-PURE", "RAND"]:
            m = json.load(open(DEC_DIR / f"seed{seed}_{sysname}.json"))
            scores[e][sysname] = m["bleu"] * 100

    rankings = {e: sorted(SYSTEMS, key=lambda x: -scores[e][x]) for e in evals}
    pairs = []
    for i in range(len(evals)):
        for j in range(i + 1, len(evals)):
            a, b = evals[i], evals[j]
            xa = [scores[a][s] for s in SYSTEMS]
            xb = [scores[b][s] for s in SYSTEMS]
            tau, _ = sst.kendalltau(xa, xb)
            rho, _ = sst.spearmanr(xa, xb)
            # pairwise flip rate: fraction of system pairs with different order
            n_pairs = flips = 0
            for ii in range(len(SYSTEMS)):
                for jj in range(ii + 1, len(SYSTEMS)):
                    s1, s2 = SYSTEMS[ii], SYSTEMS[jj]
                    d1 = scores[a][s1] - scores[a][s2]
                    d2 = scores[b][s1] - scores[b][s2]
                    if d1 == 0 or d2 == 0:
                        continue
                    n_pairs += 1
                    if d1 * d2 < 0:
                        flips += 1
            pairs.append({"a": a, "b": b, "kendall_tau": float(tau), "spearman_rho": float(rho),
                          "pairwise_flip_rate": flips / n_pairs if n_pairs else 0.0, "n_pairs": n_pairs})
    orig_pairs = [p for p in pairs if p["a"] == "original"]
    seed_pairs = [p for p in pairs if p["a"] != "original" and p["b"] != "original"]
    out = {"systems": SYSTEMS, "scores": scores, "rankings": rankings, "pairs": pairs,
           "original_vs_seed_mean": {"tau": float(np.mean([p["kendall_tau"] for p in orig_pairs])),
                                     "rho": float(np.mean([p["spearman_rho"] for p in orig_pairs])),
                                     "flip": float(np.mean([p["pairwise_flip_rate"] for p in orig_pairs]))},
           "seed_vs_seed_mean": {"tau": float(np.mean([p["kendall_tau"] for p in seed_pairs])),
                                 "rho": float(np.mean([p["spearman_rho"] for p in seed_pairs])),
                                 "flip": float(np.mean([p["pairwise_flip_rate"] for p in seed_pairs]))}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"rankings": rankings, "original_vs_seed_mean": out["original_vs_seed_mean"],
                      "seed_vs_seed_mean": out["seed_vs_seed_mean"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-only", action="store_true")
    args = ap.parse_args()
    decode_missing()
    if not args.decode_only:
        analyze()
