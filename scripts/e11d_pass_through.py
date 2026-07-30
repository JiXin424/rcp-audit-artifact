#!/usr/bin/env python3
"""Round 6 E-D: donor-content pass-through rate as a function of evaluator competence.

For every available checkpoint, decode TN-PURE-v1 and compute per-item sentence BLEU of
the decoded text against (a) the donor's own transcript and (b) the query reference.
Pass-through ratio = mean sentence-BLEU(hyp, donor transcript) / mean sentence-BLEU(hyp, query).
Together with each checkpoint's dev BLEU, this yields a dose-response curve:
does reading-out donor training content increase with competence, and is the released
evaluator on or off that curve?

Checkpoints: 6 primary cells (cached), 8 extension seeds, rescue2 wd0_seed202,
ladder fractions (when trained), config-faithful runs (when trained).
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import sacrebleu
import torch

MAJOR = Path("/ssd/xkb4/RCP/revision_20260728_major")
FULL = Path("/ssd/xkb4/RCP/revision_20260728_full_revision")
sys.path.insert(0, str(MAJOR)); sys.path.insert(0, str(FULL))
from src import evaluate_checkpoints as ev  # noqa: E402
import src.evaluate_checkpoints as evf  # noqa: E402

R5 = Path("/ssd/xkb4/RCP/revision_20260729_round5")
CANON = Path("/ssd/xkb4/RCP/revision_20260728_canonical_rebuild")
CELLS = CANON / "outputs/evaluations/cells"
REG = CANON / "registry/query_donor_registry.jsonl"
OUT = R5 / "results/e11d_pass_through.json"
SBLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp", effective_order=False, force=True)

EXT_SEEDS = [707, 808, 909, 1001, 1102, 1203, 1304, 1405]
EXT_CKPTS = {s: FULL / f"checkpoints/seed_{s}/best.ckpt" for s in EXT_SEEDS}
EXTRA = {
    "wd0_seed202": R5 / "checkpoints/rescue2_wd0_seed202/best.ckpt",
    "bs512_seed101": R5 / "checkpoints/rescue2_bs512_seed101/best.ckpt",
    "ladder_f0.125": R5 / "checkpoints/ladder_f0.125_seed101/best.ckpt",
    "ladder_f0.25": R5 / "checkpoints/ladder_f0.25_seed101/best.ckpt",
    "ladder_f0.5": R5 / "checkpoints/ladder_f0.5_seed101/best.ckpt",
    "ladder_f0.75": R5 / "checkpoints/ladder_f0.75_seed101/best.ckpt",
    "cfaith101": R5 / "checkpoints/cfaith_seed101/best.ckpt",
    "cfaith404": R5 / "checkpoints/cfaith_seed404/best.ckpt",
    "cfaith202": R5 / "checkpoints/cfaith_seed202/best.ckpt",
    "cfaith303": R5 / "checkpoints/cfaith_seed303/best.ckpt",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sent_bleu(hyp, ref):
    return SBLEU.sentence_score(hyp, [ref]).score / 100.0


def pass_rate(items, donor_text):
    a, b = [], []
    for it in items:
        dt = donor_text.get(it["id"])
        if dt is None:
            continue
        a.append(sent_bleu(it["hypothesis"], dt))
        b.append(sent_bleu(it["hypothesis"], it["reference"]))
    return float(np.mean(a)), float(np.mean(b)), float(np.mean(a) / max(np.mean(b), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    donor_text = {json.loads(l)["query_id"]: json.loads(l)["donor_text"] for l in open(REG)}
    out = json.load(open(OUT)) if OUT.exists() else {}

    # 7 primary cells from cache
    for i, name in enumerate(["original", "seed_101", "seed_202", "seed_303", "seed_404", "seed_505", "seed_606"]):
        if name in out:
            continue
        items = json.load(open(CELLS / f"cp{i}_TN-PURE-v1.json"))["metrics"]["items"]
        a, b, r = pass_rate(items, donor_text)
        out[name] = {"bleu_donor": a, "bleu_query": b, "ratio": r, "source": "cache"}
        print(name, round(a, 3), round(b, 3), round(r, 2), flush=True)

    # extension seeds + extra rescue checkpoints: decode
    todo = {**{f"seed_{s}": EXT_CKPTS[s] for s in EXT_SEEDS if f"seed_{s}" not in out},
            **{k: v for k, v in EXTRA.items() if k not in out}}
    if todo:
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
        test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")
        ids = sorted(test)
        texts = {x: test[x]["text"] for x in ids}
        pure = torch.load(CANON / "outputs/poses/TN-PURE-v1.pt", map_location="cpu", weights_only=True)
        pure = {k: (v if isinstance(v, torch.Tensor) else v["poses_3d"]) for k, v in pure.items()}
        _, txt, model = ev._vocab_and_model(device)
        for name, ck in todo.items():
            payload = ev.safe_torch_load(ck, sha256_file(ck), "checkpoint")
            model.load_state_dict(payload["model"]); model.to(device).eval()
            m = ev.evaluate_pose_set(model, txt, ids, pure, texts, device, 32, True)
            a, b, r = pass_rate(m["items"], donor_text)
            out[name] = {"bleu_donor": a, "bleu_query": b, "ratio": r, "source": "decoded"}
            print(name, round(a, 3), round(b, 3), round(r, 2), flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
