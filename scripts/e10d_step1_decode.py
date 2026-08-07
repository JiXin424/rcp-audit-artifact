#!/usr/bin/env python3
"""E10d step 1: decode path C (double subsample) under the original evaluator, save hyps."""
import json, sys
from pathlib import Path
MAJOR = Path(__file__).resolve().parents[1] / "results"
FULL = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(MAJOR)); sys.path.insert(0, str(FULL))
import torch
from src import evaluate_checkpoints as ev
import src.evaluate_checkpoints as evf
R5 = Path(__file__).resolve().parents[1] / "results"

def main():
    test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")
    ids = sorted(test)
    texts = {x: test[x]["text"] for x in ids}
    torch.cuda.set_device(0)
    _, txt, model = ev._vocab_and_model("cuda:0")
    state = evf.load_original_checkpoint(str(ev.MODEL_ROOT / "best.ckpt"),
                                         ev.PINNED[str(ev.MODEL_ROOT / "best.ckpt")])
    model.load_state_dict(state["state_dict"]); model.to("cuda:0").eval()
    poses_c = {k: test[k]["poses_3d"][::2][::2] for k in ids}
    m = ev.evaluate_pose_set(model, txt, ids, poses_c, texts, "cuda:0", 32, True)
    (R5 / "results/e10d_pathC_decode.json").write_text(json.dumps(
        {"decoded_bleu": m["decoded_bleu"],
         "items": [{"id": it["id"], "reference": it["reference"], "hypothesis": it["hypothesis"]} for it in m["items"]]},
        ensure_ascii=False))
    print("path C bleu:", m["decoded_bleu"])

if __name__ == "__main__":
    main()
