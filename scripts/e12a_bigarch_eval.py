#!/usr/bin/env python3
"""Evaluate big-arch checkpoints: dev competence + GT/PURE test gap (+ optional holdout memorization).

Builds the model from each variant's own model_config.yaml (architecture differs from
the legacy 3L/256 recipe), loads best.ckpt, and evaluates.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
import yaml

MAJOR = Path("/ssd/xkb4/RCP/revision_20260728_major")
ROUND3 = Path("/ssd/xkb4/RCP/revision_20260728_round3")
FULL = Path("/ssd/xkb4/RCP/revision_20260728_full_revision")
sys.path.insert(0, str(ROUND3 / "src"))
sys.path.insert(0, str(FULL))
sys.path.insert(0, str(FULL / "src"))
import train_matched as tm  # noqa: E402
import evaluate_checkpoints as ec  # noqa: E402

R5 = Path("/ssd/xkb4/RCP/revision_20260729_round5")
OUT = R5 / "results/e12a_bigarch_eval.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpts", nargs="+", required=True, help="name:ckpt_dir")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    dev = torch.load(ec.DEV_PT, map_location="cpu", weights_only=True)
    dev_ids = sorted(dev)
    dev_texts = {x: dev[x]["text"] for x in dev_ids}
    dev_poses = {x: dev[x]["poses_3d"][::2] for x in dev_ids}
    test = torch.load(ec.TEST_PT, map_location="cpu", weights_only=True)
    test_ids = sorted(test)
    test_texts = {x: test[x]["text"] for x in test_ids}
    CANON = Path("/ssd/xkb4/RCP/revision_20260728_canonical_rebuild")
    gt = torch.load(CANON / "outputs/poses/GT-v1.pt", map_location="cpu", weights_only=True)
    gt = {k: (v if isinstance(v, torch.Tensor) else v["poses_3d"]) for k, v in gt.items()}
    pure = torch.load(CANON / "outputs/poses/TN-PURE-v1.pt", map_location="cpu", weights_only=True)
    pure = {k: (v if isinstance(v, torch.Tensor) else v["poses_3d"]) for k, v in pure.items()}
    train = torch.load(ec.TRAIN_PT, map_location="cpu", weights_only=True)
    # 525 holdout ids (same selection as e11f)
    import hashlib as hl
    SEED = "rcp-slrtp-holdout-20260722-v1"
    keys = list(train.keys())
    glosses = [train[k].get("gloss", "") for k in keys]
    t2s = {}
    for i, g in enumerate(glosses):
        t2s.setdefault(g, []).append(i)
    templates = sorted(t2s.keys(), key=lambda t: int(hl.sha256(f"{SEED}:{t}".encode()).hexdigest(), 16))
    hold = set()
    for t in templates[: int(len(templates) * 0.075)]:
        hold.update(t2s[t])
    hold_ids = [keys[i] for i in sorted(hold)]
    hold_texts = {k: train[k]["text"] for k in hold_ids}
    hold_poses = {k: train[k]["poses_3d"][::2] for k in hold_ids}

    gls = tm.Vocabulary(tm.GLS_VOCAB, [tm.SPECIAL["sil"], tm.SPECIAL["unk"], tm.SPECIAL["pad"]], 1)
    txt = tm.Vocabulary(tm.TXT_VOCAB, [tm.SPECIAL["unk"], tm.SPECIAL["pad"], tm.SPECIAL["bos"], tm.SPECIAL["eos"]], 0)

    out = json.load(open(OUT)) if OUT.exists() else {}
    for pair in args.ckpts:
        name, ckdir = pair.split(":", 1)
        if name in out:
            continue
        ckdir = Path(ckdir)
        cfg = yaml.safe_load((ckdir / "model_config.yaml").read_text())
        model = tm.build_model(cfg, gls, txt, device)
        ck = ckdir / "best.ckpt"
        payload = ec.safe_torch_load(ck, sha256_file(ck), "checkpoint")
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        md = ec.evaluate_pose_set(model, txt, dev_ids, dev_poses, dev_texts, device, 32, False)
        mg = ec.evaluate_pose_set(model, txt, test_ids, gt, test_texts, device, 32, False)
        mp = ec.evaluate_pose_set(model, txt, test_ids, pure, test_texts, device, 32, True)
        mh = ec.evaluate_pose_set(model, txt, hold_ids, hold_poses, hold_texts, device, 32, False)
        out[name] = {"dev_bleu": md["decoded_bleu"], "dev_wer": md["wer"],
                     "test_GT": mg["decoded_bleu"], "test_PURE": mp["decoded_bleu"],
                     "gap": mp["decoded_bleu"] - mg["decoded_bleu"],
                     "holdout_GT": mh["decoded_bleu"],
                     "pure_items": mp["items"]}
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items() if k != "pure_items"}, flush=True)
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "pure_items"} for k, v in out.items()}
        (R5 / "results/e12a_bigarch_eval.json").write_text(json.dumps(slim, indent=1))
        (R5 / f"results/e12a_{name}_pure_items.json").write_text(json.dumps(out[name]["pure_items"]))


if __name__ == "__main__":
    main()
