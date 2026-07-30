#!/usr/bin/env python3
"""E-B eval: dev competence + GT/PURE test gaps for ladder checkpoints (and any config-faithful finished)."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

FULL = Path("/ssd/xkb4/RCP/revision_20260728_full_revision")
sys.path.insert(0, str(FULL)); sys.path.insert(0, str(FULL / "src"))
import evaluate_checkpoints as ec  # noqa: E402

R5 = Path("/ssd/xkb4/RCP/revision_20260729_round5")
CANON = Path("/ssd/xkb4/RCP/revision_20260728_canonical_rebuild")
OUT = R5 / "results/e11b_ladder_gaps.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpts", nargs="+", required=True, help="name:path pairs")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    dev = torch.load(ec.DEV_PT, map_location="cpu", weights_only=True)
    dev_ids = sorted(dev)
    dev_texts = {x: dev[x]["text"] for x in dev_ids}
    dev_poses = {x: ec.frame_rate_transform(dev[x]["poses_3d"], 25, False) for x in dev_ids}
    test = torch.load(ec.TEST_PT, map_location="cpu", weights_only=True)
    test_ids = sorted(test)
    test_texts = {x: test[x]["text"] for x in test_ids}
    gt = torch.load(CANON / "outputs/poses/GT-v1.pt", map_location="cpu", weights_only=True)
    gt = {k: (v if isinstance(v, torch.Tensor) else v["poses_3d"]) for k, v in gt.items()}
    pure = torch.load(CANON / "outputs/poses/TN-PURE-v1.pt", map_location="cpu", weights_only=True)
    pure = {k: (v if isinstance(v, torch.Tensor) else v["poses_3d"]) for k, v in pure.items()}
    _, txt, model = ec._vocab_and_model(device)

    out = json.load(open(OUT)) if OUT.exists() else {}
    for pair in args.ckpts:
        name, ck = pair.split(":", 1)
        if name in out:
            continue
        payload = ec.safe_torch_load(ck, sha256_file(ck), "checkpoint")
        model.load_state_dict(payload["model"]); model.to(device).eval()
        md = ec.evaluate_pose_set(model, txt, dev_ids, dev_poses, dev_texts, device, 32, False)
        mg = ec.evaluate_pose_set(model, txt, test_ids, gt, test_texts, device, 32, False)
        mp = ec.evaluate_pose_set(model, txt, test_ids, pure, test_texts, device, 32, True)
        out[name] = {"dev_bleu": md["decoded_bleu"], "dev_wer": md["wer"],
                     "test_GT": mg["decoded_bleu"], "test_PURE": mp["decoded_bleu"],
                     "gap": mp["decoded_bleu"] - mg["decoded_bleu"]}
        print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
