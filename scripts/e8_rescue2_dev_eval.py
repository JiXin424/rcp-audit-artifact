#!/usr/bin/env python3
"""E8: dev-set competence evaluation for rescue2 variant checkpoints.

Decodes the 515-sequence dev split (12.5 fps) with each rescue2 best.ckpt and reports
dev BLEU-4 / WER / NLL plus distance to the competence gate (|dBLEU|<=0.01, |dWER|<=0.03
fractional, vs original 0.1338 / 0.7749).
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

FULL = Path("/ssd/xkb4/RCP/revision_20260728_full_revision")
sys.path.insert(0, str(FULL))
sys.path.insert(0, str(FULL / "src"))
import evaluate_checkpoints as ec  # noqa: E402

R5 = Path("/ssd/xkb4/RCP/revision_20260729_round5")
OUT = R5 / "results/e8_rescue2_dev_eval.json"
ORIG = {"dev_bleu4": 0.13378651856913776, "dev_wer": 0.7748698273499589}
GATE = {"bleu": 0.01, "wer": 0.03}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    dev = torch.load(ec.DEV_PT, map_location="cpu", weights_only=True)
    dev_ids = sorted(dev)
    texts = {x: dev[x]["text"] for x in dev_ids}
    poses = {x: ec.frame_rate_transform(dev[x]["poses_3d"], 25, False) for x in dev_ids}
    _, txt, model = ec._vocab_and_model(device)

    out = json.load(open(OUT)) if OUT.exists() else {}
    for ckpt in sorted(R5.glob("checkpoints/rescue2_*/best.ckpt")):
        name = ckpt.parent.name
        if args.variants and not any(v in name for v in args.variants):
            continue
        if name in out:
            continue
        payload = ec.safe_torch_load(ckpt, sha256_file(ckpt), "checkpoint")
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        m = ec.evaluate_pose_set(model, txt, dev_ids, poses, texts, device, 32, False)
        bleu = m["decoded_bleu"]; wer = m["wer"]
        out[name] = {"dev_bleu4": bleu, "dev_wer": wer,
                     "dev_nll": m["teacher_forced_nll_per_token"],
                     "abs_dbleu": abs(bleu - ORIG["dev_bleu4"]),
                     "abs_dwer": abs(wer - ORIG["dev_wer"]),
                     "gate_pass": bool(abs(bleu - ORIG["dev_bleu4"]) <= GATE["bleu"]
                                       and abs(wer - ORIG["dev_wer"]) <= GATE["wer"])}
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()}, flush=True)
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
