#!/usr/bin/env python3
"""Round 6 E-C: pose-perturbation battery -- what does the released evaluator read?

Applies controlled degradations to the TN-PURE-v1 donor poses and GT poses and measures
decoded BLEU under the original evaluator and the best rescue reconstruction (wd0_seed202):

  identity         : no perturbation (reference)
  reversal         : reverse frame order (destroys linguistic content, keeps marginals)
  window_shuffle   : shuffle frames within 1-s (12-frame) windows (destroys fine temporal order)
  spatial_jitter   : Gaussian noise sigma = 5% of per-joint-channel std (destroys fine spatial detail)
  hand_mask        : zero out hand joints 8-49 (tests hand-channel dependence)
  face_mask        : zero out face joints 50-177 (tests face-channel dependence)

If the original evaluator's high PURE score survives perturbations that destroy
content (reversal/shuffle), it is reading memorized static patterns rather than
linguistic motion; if it collapses like GT does, it reads content unusually well.
Comparison against the rescue checkpoint on identical inputs controls for competence.
"""
from __future__ import annotations
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
CANON = Path(__file__).resolve().parents[1] / "data/cells"
OUT = R5 / "results/e11c_perturbation.json"

HAND = list(range(8, 50))
FACE = list(range(50, 178))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def perturb(name, poses, seed=0):
    g = np.random.default_rng(seed)
    out = {}
    for k, p in poses.items():
        q = p.clone()
        if name == "identity":
            pass
        elif name == "reversal":
            q = torch.flip(q, dims=[0])
        elif name == "window_shuffle":
            idx = torch.arange(q.shape[0])
            w = 12
            perm = []
            for s in range(0, q.shape[0], w):
                seg = idx[s:s + w]
                perm.append(seg[torch.randperm(len(seg), generator=torch.Generator().manual_seed(seed + s))])
            q = q[torch.cat(perm)]
        elif name == "spatial_jitter":
            std = q.std(dim=0, keepdim=True).clamp_min(1e-4)
            q = q + torch.from_numpy(g.normal(0, 0.05, q.shape)).float() * std
        elif name == "hand_mask":
            q[:, HAND, :] = 0
        elif name == "face_mask":
            q[:, FACE, :] = 0
        out[k] = q
    return out


PERTS = ["identity", "reversal", "window_shuffle", "spatial_jitter", "hand_mask", "face_mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--evaluator", required=True, choices=["original", "wd0_seed202"])
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")
    ids = sorted(test)
    texts = {x: test[x]["text"] for x in ids}
    _, txt, model = ev._vocab_and_model(device)
    if args.evaluator == "original":
        state = evf.load_original_checkpoint(str(ev.MODEL_ROOT / "best.ckpt"),
                                             ev.PINNED[str(ev.MODEL_ROOT / "best.ckpt")])
    else:
        ck = R5 / "checkpoints/rescue2_wd0_seed202/best.ckpt"
        state = {"state_dict": ev.safe_torch_load(ck, sha256_file(ck), "checkpoint")["model"]}
    model.load_state_dict(state["state_dict"]); model.to(device).eval()

    out = json.load(open(OUT)) if OUT.exists() else {}
    out.setdefault(args.evaluator, {})
    for sysname in ["GT-v1", "TN-PURE-v1"]:
        base = torch.load(CANON / f"{sysname}.pt", map_location="cpu", weights_only=True)
        base = {k: (v if isinstance(v, torch.Tensor) else v["poses_3d"]) for k, v in base.items()}
        for pert in PERTS:
            if pert in out[args.evaluator].get(sysname, {}):
                continue
            poses = perturb(pert, base, seed=hash((pert, sysname)) % 1000)
            m = ev.evaluate_pose_set(model, txt, ids, poses, texts, device, 32, False)
            out[args.evaluator].setdefault(sysname, {})[pert] = {
                "bleu": m["decoded_bleu"] * 100, "nll": m["teacher_forced_nll_per_token"]}
            print(f"{args.evaluator} {sysname} {pert}: bleu={m['decoded_bleu']*100:.2f}", flush=True)
            OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out[args.evaluator], indent=1))


if __name__ == "__main__":
    main()
