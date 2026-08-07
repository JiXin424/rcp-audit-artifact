#!/usr/bin/env python3
"""E-F: memorization-signature comparison on the seen 525-sequence template holdout.

The released evaluator scores GT on the holdout it was trained on at 77.4 BLEU (vs 12.78
on unseen PHX-public) -- a large memorization signature. Reconstructions were ALSO trained
on those same 525 sequences (they are part of the 7,060 released train pool). This decodes
the holdout GT under reconstruction/rescue checkpoints to test whether training-pool
memorization is present in the recipe-conditioned family at all, and at what strength --
a direct, positive test of "memorization is common, the reversal is special" vs
"the released evaluator memorizes qualitatively differently".
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

FULL = Path(__file__).resolve().parents[1] / "results"
MAJOR = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(FULL)); sys.path.insert(0, str(FULL / "src"))
import evaluate_checkpoints as ec  # noqa: E402

R5 = Path(__file__).resolve().parents[1] / "results"
OUT = R5 / "results/e11f_holdout_memorization.json"
SEED = "rcp-slrtp-holdout-20260722-v1"
FRACTION = 0.075


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def holdout_ids(train):
    import hashlib as hl
    keys = list(train.keys())
    glosses = [train[k].get("gloss", "") for k in keys]
    template_to_seqs = {}
    for i, g in enumerate(glosses):
        template_to_seqs.setdefault(g, []).append(i)
    templates = sorted(template_to_seqs.keys(),
                       key=lambda t: int(hl.sha256(f"{SEED}:{t}".encode()).hexdigest(), 16))
    n = int(len(templates) * FRACTION)
    hold = set()
    for t in templates[:n]:
        hold.update(template_to_seqs[t])
    return [keys[i] for i in sorted(hold)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpts", nargs="+", required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    train = torch.load(ec.TRAIN_PT, map_location="cpu", weights_only=True)
    ids = holdout_ids(train)
    assert len(ids) == 525, len(ids)
    texts = {k: train[k]["text"] for k in ids}
    poses = {k: train[k]["poses_3d"][::2] for k in ids}
    _, txt, model = ec._vocab_and_model(device)
    out = json.load(open(OUT)) if OUT.exists() else {}
    for pair in args.ckpts:
        name, ck = pair.split(":", 1)
        if name in out:
            continue
        if name == "original":
            import src.evaluate_checkpoints as evf  # noqa: E402
            state = evf.load_original_checkpoint(str(ec.MODEL_ROOT / "best.ckpt"),
                                                 ec.PINNED[str(ec.MODEL_ROOT / "best.ckpt")])
            model.load_state_dict(state["state_dict"])
        else:
            payload = ec.safe_torch_load(ck, sha256_file(ck), "checkpoint")
            model.load_state_dict(payload["model"])
        model.to(device).eval()
        m = ec.evaluate_pose_set(model, txt, ids, poses, texts, device, 32, False)
        out[name] = {"holdout_GT_bleu": m["decoded_bleu"] * 100, "n": len(ids)}
        print(name, round(m["decoded_bleu"] * 100, 2), flush=True)
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
