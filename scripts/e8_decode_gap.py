#!/usr/bin/env python3
"""Decode GT-v1 and TN-PURE-v1 canonical cells under a rescue2 checkpoint (competence-gap extension)."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

FULL = Path("/ssd/xkb4/RCP/revision_20260728_full_revision")
sys.path.insert(0, str(FULL)); sys.path.insert(0, str(FULL / "src"))
import evaluate_checkpoints as ec  # noqa: E402

R5 = Path("/ssd/xkb4/RCP/revision_20260729_round5")
CANON_POSE = Path("/ssd/xkb4/RCP/revision_20260728_canonical_rebuild/outputs/poses")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    test = torch.load(ec.TEST_PT, map_location="cpu", weights_only=True)
    ids = sorted(test)
    texts = {x: test[x]["text"] for x in ids}
    _, txt, model = ec._vocab_and_model(device)
    payload = ec.safe_torch_load(args.ckpt, sha256_file(args.ckpt), "checkpoint")
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    out = {}
    for sysname in ["GT-v1", "TN-PURE-v1"]:
        poses = torch.load(CANON_POSE / f"{sysname}.pt", map_location="cpu", weights_only=True)
        poses = {k: (v if isinstance(v, torch.Tensor) else v["poses_3d"]) for k, v in poses.items()}
        assert sorted(poses) == ids
        m = ec.evaluate_pose_set(model, txt, ids, poses, texts, device, 32, False)
        out[sysname] = {"decoded_bleu": m["decoded_bleu"], "wer": m["wer"],
                        "nll": m["teacher_forced_nll_per_token"]}
        print(args.name, sysname, round(m["decoded_bleu"] * 100, 2), flush=True)
    out["gap_pure_minus_gt"] = out["TN-PURE-v1"]["decoded_bleu"] - out["GT-v1"]["decoded_bleu"]
    (R5 / f"results/e8_gap_{args.name}.json").write_text(json.dumps(out, indent=1))
    print(args.name, "gap:", round(out["gap_pure_minus_gt"] * 100, 2))


if __name__ == "__main__":
    main()
