#!/usr/bin/env python3
"""E10d: recompute the three-path FPS sensitivity WER under the OFFICIAL jiwer protocol.

Path C (double subsample, 6.25 fps) is re-decoded under the original evaluator to obtain
its hypotheses; official jiwer 3.1.0 WER (with the official normalization) is computed
for all three paths, replacing the legacy custom-protocol WER (0.7926/0.8026) in the table.
"""
import json, sys
from pathlib import Path

MAJOR = Path(__file__).resolve().parents[1] / "results"
FULL = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(MAJOR)); sys.path.insert(0, str(FULL))
import torch  # noqa: E402
from src import evaluate_checkpoints as ev  # noqa: E402
import src.evaluate_checkpoints as evf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import wer as official_wer  # noqa: E402

R5 = Path(__file__).resolve().parents[1] / "results"
OUT = R5 / "results/e10d_three_path_official_wer.json"
CELLS = Path(__file__).resolve().parents[1] / "data/cells"


def main():
    test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")
    ids = sorted(test)
    texts = {x: test[x]["text"] for x in ids}

    # Path A/B: canonical 12.5 fps decode (already stored)
    gt_items = json.load(open(CELLS / "cp0_GT-v1.json"))["metrics"]["items"]
    refs = [it["reference"] for it in gt_items]
    hyps_ab = [it["hypothesis"] for it in gt_items]
    wer_ab = official_wer(hyps_ab, refs)

    # Path C: double subsample re-decode
    torch.cuda.set_device(0)
    _, txt, model = ev._vocab_and_model("cuda:0")
    state = evf.load_original_checkpoint(str(ev.MODEL_ROOT / "best.ckpt"),
                                         ev.PINNED[str(ev.MODEL_ROOT / "best.ckpt")])
    model.load_state_dict(state["state_dict"]); model.to("cuda:0").eval()
    poses_c = {k: test[k]["poses_3d"][::2][::2] for k in ids}
    m = ev.evaluate_pose_set(model, txt, ids, poses_c, texts, "cuda:0", 32, True)
    hyps_c = [it["hypothesis"] for it in m["items"]]
    wer_c = official_wer(hyps_c, refs)
    bleu_c = m["decoded_bleu"]

    out = {
        "protocol": "official jiwer 3.1.0 with official normalization, 0-100 scale",
        "path_A_canonical": {"bleu4": 12.777, "wer_official": wer_ab},
        "path_B_noop": {"bleu4": 12.777, "wer_official": wer_ab},
        "path_C_double_subsample": {"bleu4": bleu_c * 100, "wer_official": wer_c},
        "legacy_custom_wer": {"A": 79.26, "C": 80.26},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
