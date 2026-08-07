#!/usr/bin/env python3
"""Round 5 E1: re-score the frozen 605-query exposure factorial at the canonical 12.5 fps.

The original task7 scoring (exposure_factorial.py run_score) fed RAW 25-fps poses to
evaluate_pose_set (no [::2] external subsample). The model itself does not subsample
(model.subsample attribute is set but unused in the released code). All canonical cells
apply pose[::2] externally. This script re-scores all six conditions under the original
evaluator and the six matched-best reconstruction seeds with the [::2] transform applied,
then computes the full 2x2 factorial report: cell means, seen/unseen main effect,
high/low similarity main effect, interaction, with CIs and Holm correction.

Usage: python e1_factorial_rescore.py --gpu 0 --evaluators original seed_101 ...
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

MAJOR = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(MAJOR))
sys.path.insert(0, str(MAJOR.parent / "revision_20260728_round3"))

import torch  # noqa: E402
from src import evaluate_checkpoints as ev  # noqa: E402

FULL_REVISION = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(FULL_REVISION))
import src.evaluate_checkpoints as ev_full  # noqa: E402

MANIFEST = MAJOR / "results/task7_canonical_v2/selection_manifest.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "results/e1_factorial_cells_12p5fps")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA = ev.DATA_ROOT / "data"
CONDITIONS = ["seen_high", "seen_low", "unseen_high", "unseen_low", "control_own_gt", "control_seen_matched"]


def seed_inventory():
    summary = json.loads((MAJOR / "results/task5_training_summary_v2.json").read_text())
    inv = {}
    for r in summary["runs"]:
        s = r["selected_checkpoint"]
        inv[f"seed_{r['seed']}"] = {"path": str(MAJOR / s["path"]), "sha256": s["sha256"], "arch": "matched"}
    inv["original"] = {"path": str(ev.MODEL_ROOT / "best.ckpt"),
                       "sha256": ev.PINNED[str(ev.MODEL_ROOT / "best.ckpt")], "arch": "original"}
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--evaluators", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    kept = [r for r in manifest["rows"] if r["included"]]
    ids = [r["target_id"] for r in kept]
    assert len(ids) == 605

    train = ev.safe_torch_load(ev.TRAIN_PT, ev.PINNED[str(ev.TRAIN_PT)], "train")
    test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")
    texts = {x: test[x]["text"] for x in ids}

    # Build 12.5 fps pose sets per condition (external [::2], matching canonical protocol).
    pose_sets = {}
    for cond in CONDITIONS:
        poses = {}
        for r in kept:
            if cond in ("seen_high", "seen_low", "unseen_high", "unseen_low"):
                donor = r["cells"][cond]["donor_id"]
            elif cond == "control_own_gt":
                donor = r["target_id"]
            else:
                donor = r["controls"]["seen_matched"]["donor_id"]
            p = (train if donor in train else test)[donor]["poses_3d"]
            poses[r["target_id"]] = p[::2]
        pose_sets[cond] = poses

    inv = seed_inventory()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    _, txt, model = ev._vocab_and_model(device)

    for name in args.evaluators:
        spec = inv[name]
        out_path = OUT_DIR / f"cells_{name}.json"
        if out_path.exists():
            print(f"skip {name} (exists)")
            continue
        if spec["arch"] == "original":
            state = ev_full.load_original_checkpoint(spec["path"], spec["sha256"])
        else:
            state = ev.checkpoint_state(spec["path"], spec["sha256"], spec["arch"])
        model.load_state_dict(state["state_dict"], strict=True)
        model.to(device).eval()
        t0 = time.time()
        results = {}
        for cond in CONDITIONS:
            m = ev.evaluate_pose_set(model, txt, ids, pose_sets[cond], texts, device, args.batch_size, True)
            results[cond] = m
            print(f"{name} {cond}: bleu={m['decoded_bleu']:.4f} nll={m['teacher_forced_nll_per_token']:.4f}", flush=True)
        out_path.write_text(json.dumps({
            "schema": "e1-factorial-12p5fps-v1",
            "protocol_correction": "poses subsampled externally [::2] to 12.5 fps; task7 original scoring used raw 25 fps",
            "evaluator": name, "n": len(ids), "fps": 12.5,
            "conditions": {c: {"decoded_bleu": results[c]["decoded_bleu"],
                                "teacher_forced_nll_per_token": results[c]["teacher_forced_nll_per_token"],
                                "items": results[c]["items"]} for c in CONDITIONS},
            "runtime_seconds": time.time() - t0}, ensure_ascii=False))
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
