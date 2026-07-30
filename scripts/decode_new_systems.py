#!/usr/bin/env python3
"""Round 5 decode: evaluate new pose systems under a given evaluator at 12.5 fps.

Usage: python decode_new_systems.py --gpu 0 --evaluator original --systems SEEN-RAND640-MATCHED-v1 SEEN-MATCHED-T005 SEEN-MATCHED-T020 SEEN-MATCHED-SIGNER
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

MAJOR = Path("/ssd/xkb4/RCP/revision_20260728_major")
FULL = Path("/ssd/xkb4/RCP/revision_20260728_full_revision")
sys.path.insert(0, str(MAJOR))
sys.path.insert(0, str(FULL))
import torch  # noqa: E402
from src import evaluate_checkpoints as ev  # noqa: E402
import src.evaluate_checkpoints as evf  # noqa: E402

R5 = Path("/ssd/xkb4/RCP/revision_20260729_round5")
OUT_DIR = R5 / "results/decoded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_CKPTS = {101: MAJOR / "checkpoints/v2_seed_101/best.ckpt",
              202: MAJOR / "checkpoints/v2_seed_202/best.ckpt",
              303: MAJOR / "checkpoints/v2_seed_303/best.ckpt",
              404: MAJOR / "checkpoints/v2_seed_404/best.ckpt",
              505: MAJOR / "checkpoints/v2_seed_505/best.ckpt",
              606: MAJOR / "checkpoints/v2_seed_606/best.ckpt"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--evaluator", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"

    test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")
    _, txt, model = ev._vocab_and_model(device)
    if args.evaluator == "original":
        state = evf.load_original_checkpoint(str(ev.MODEL_ROOT / "best.ckpt"),
                                             ev.PINNED[str(ev.MODEL_ROOT / "best.ckpt")])
        model.load_state_dict(state["state_dict"])
    else:
        seed = int(args.evaluator.split("_")[1])
        payload = ev.safe_torch_load(SEED_CKPTS[seed], sha256_file(SEED_CKPTS[seed]), "checkpoint")
        model.load_state_dict(payload["model"])
    model.to(device).eval()

    for sysname in args.systems:
        out_path = OUT_DIR / f"{sysname}__{args.evaluator}.json"
        if out_path.exists():
            print(f"skip {sysname} under {args.evaluator}")
            continue
        poses_raw = torch.load(R5 / "outputs" / f"{sysname}.pt", map_location="cpu", weights_only=True)
        ids = sorted(poses_raw)
        poses = {k: poses_raw[k][::2] for k in ids}
        texts = {k: test[k]["text"] for k in ids}
        t0 = time.time()
        m = ev.evaluate_pose_set(model, txt, ids, poses, texts, device, 32, True)
        out_path.write_text(json.dumps({
            "system": sysname, "evaluator": args.evaluator, "fps": 12.5, "n": len(ids),
            "decoded_bleu": m["decoded_bleu"], "wer": m["wer"],
            "teacher_forced_nll_per_token": m["teacher_forced_nll_per_token"],
            "items": m["items"], "runtime_seconds": time.time() - t0}, ensure_ascii=False))
        print(f"{sysname} under {args.evaluator}: bleu={m['decoded_bleu']*100:.2f} nll={m['teacher_forced_nll_per_token']:.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
