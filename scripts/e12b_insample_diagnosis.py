#!/usr/bin/env python3
"""E-B': in-sample decode diagnosis -- why can a model with train loss 0.20 not
freely decode the samples it was trained on?

For the original evaluator and the most overfit reconstruction (wd0_seed202, train loss 0.20):
  - free-decode BLEU + exact-match rate on the full 7,060-item train pool (in-sample)
  - teacher-forced NLL and top-1 token accuracy on train / dev / test
This separates "memorized in the teacher-forced sense but exposure-biased at free decode"
(normal seq2seq behavior) from a preprocessing/tokenizer/EOS path inconsistency
(which would show as pathological in-sample free-decode failures).
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
OUT = R5 / "results/e12b_insample_diagnosis.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def summarize(m, label):
    items = m["items"]
    exact = float(np.mean([it["hypothesis"].strip() == it["reference"].strip() for it in items]))
    nll = sum(it["nll_sum"] for it in items) / max(sum(it["token_count"] for it in items), 1)
    top1 = sum(it["top1_correct_count"] for it in items) / max(sum(it["token_count"] for it in items), 1)
    return {"label": label, "n": len(items), "free_bleu": m["decoded_bleu"] * 100,
            "exact_match_rate": exact, "teacher_forced_nll": nll, "top1_token_acc": top1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--evaluator", required=True, choices=["original", "wd0_seed202"])
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
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
    for split, pt_path in [("train", ev.TRAIN_PT), ("dev", ev.DEV_PT), ("test", ev.TEST_PT)]:
        if split in out[args.evaluator]:
            continue
        data = ev.safe_torch_load(pt_path, ev.PINNED[str(pt_path)], split)
        ids = sorted(data)
        texts = {x: data[x]["text"] for x in ids}
        poses = {x: data[x]["poses_3d"][::2] for x in ids}
        m = ev.evaluate_pose_set(model, txt, ids, poses, texts, device, 48, True)
        out[args.evaluator][split] = summarize(m, split)
        print(args.evaluator, split, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[args.evaluator][split].items()}, flush=True)
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
