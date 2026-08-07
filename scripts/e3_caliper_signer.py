#!/usr/bin/env python3
"""E3b/c: decode UNSEEN-PURE full 641 under original (items) and compute caliper/signer D estimates."""
import json, math, re, sys, time
from pathlib import Path
from collections import Counter
import numpy as np

MAJOR = Path(__file__).resolve().parents[1] / "results"
FULL = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(MAJOR)); sys.path.insert(0, str(FULL))
import torch  # noqa: E402
from src import evaluate_checkpoints as ev  # noqa: E402
import src.evaluate_checkpoints as evf  # noqa: E402

R5 = Path(__file__).resolve().parents[1] / "results"
OUT = R5 / "results/e3_caliper_signer_d.json"
UNSEEN_ITEMS = R5 / "results/decoded/UNSEEN-PURE-v1__original.json"


def tokenize13a(text):
    text = text.replace("<skipped>", "").replace("-\n", "")
    text = " ".join(re.findall(r"\S+", text))
    text = re.sub(r"([\{-\~\[-\^\`\{-\}])", r" \1 ", text)
    text = re.sub(r"([\.\!\;\:\?\,])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower().split(" ") if text.strip() else []


def suff(hyp, ref):
    c = np.zeros(4); t = np.zeros(4)
    for n in range(1, 5):
        rng = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hng = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        c[n - 1] = sum(min(v, rng.get(g, 0)) for g, v in hng.items())
        t[n - 1] = max(len(hyp) - n + 1, 0)
    return c, t, len(hyp), len(ref)


def bleu_corpus(items, idx):
    C = np.zeros(4); T = np.zeros(4); SL = 0; RL = 0
    for i in idx:
        c, t, sl, rl = items[i]
        C += c; T += t; SL += sl; RL += rl
    if SL == 0:
        return 0.0
    bp = 1.0 if SL >= RL else math.exp(1 - RL / SL)
    p = [(0.5 / T[n] if T[n] and C[n] == 0 else (C[n] / T[n] if T[n] else 0.0)) for n in range(4)]
    if min(p) <= 0:
        return 0.0
    return float(bp * math.exp(sum(math.log(x) for x in p) / 4) * 100)


def prep(items_list):
    return [suff(tokenize13a(it["hypothesis"]), tokenize13a(it["reference"])) for it in items_list]


def main():
    # decode UNSEEN-PURE full 641 if needed
    if not UNSEEN_ITEMS.exists():
        torch.cuda.set_device(0)
        test = ev.safe_torch_load(ev.TEST_PT, ev.PINNED[str(ev.TEST_PT)], "test")
        _, txt, model = ev._vocab_and_model("cuda:0")
        state = evf.load_original_checkpoint(str(ev.MODEL_ROOT / "best.ckpt"), ev.PINNED[str(ev.MODEL_ROOT / "best.ckpt")])
        model.load_state_dict(state["state_dict"]); model.to("cuda:0").eval()
        raw = torch.load(str(Path(__file__).resolve().parents[1] / "data/cells") + "/UNSEEN-PURE-v1.pt",
                         map_location="cpu", weights_only=True)
        ids = sorted(raw)
        # NOTE: this pose file is already materialized at 12.5 fps (verified: 30,027 total
        # frames = median donor duration 3.75 s x 12.5 fps). No [::2] here.
        poses = {k: raw[k]["poses_3d"] for k in ids}
        texts = {k: test[k]["text"] for k in ids}
        m = ev.evaluate_pose_set(model, txt, ids, poses, texts, "cuda:0", 32, True)
        UNSEEN_ITEMS.write_text(json.dumps({"system": "UNSEEN-PURE-v1", "evaluator": "original",
                                            "decoded_bleu": m["decoded_bleu"], "items": m["items"]}, ensure_ascii=False))
        print("decoded UNSEEN-PURE:", m["decoded_bleu"])

    unseen = json.load(open(UNSEEN_ITEMS))["items"]
    unseen_by_id = {it["id"]: it for it in unseen}

    variants = {"SEEN-MATCHED-T005": None, "SEEN-MATCHED-T020": None, "SEEN-MATCHED-SIGNER": None,
                "SEEN-PURE-MATCHED-v1 (canonical, 622)": None}
    # canonical matched items from r5 common support
    common = json.loads(Path(__file__).resolve().parents[1] / "results/r5_common_support_eval.json").read_text())
    by_es = {(r["evaluator"], r["system"]): r["items"] for r in common["rows"]}

    out = {}
    B = 10_000
    rng = np.random.default_rng(42)
    for name in ["SEEN-MATCHED-T005", "SEEN-MATCHED-T020", "SEEN-MATCHED-SIGNER"]:
        dec = json.load(open(R5 / f"results/decoded/{name}__original.json"))
        ids = [it["id"] for it in dec["items"]]
        seen_items = prep(dec["items"])
        unseen_items = prep([unseen_by_id[i] for i in ids])
        n = len(ids)
        pt = bleu_corpus(seen_items, range(n)) - bleu_corpus(unseen_items, range(n))
        boots = np.zeros(B)
        for b in range(B):
            idx = rng.integers(0, n, n)
            boots[b] = bleu_corpus(seen_items, idx) - bleu_corpus(unseen_items, idx)
        out[name] = {"n_support": n, "seen_bleu": bleu_corpus(seen_items, range(n)),
                     "unseen_bleu": bleu_corpus(unseen_items, range(n)),
                     "D": float(pt), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        print(name, out[name])

    # canonical D on 622 for reference
    ids622 = common["common_support_ids"]
    seen_items = prep(by_es[("original", "SEEN-PURE-MATCHED-v1")])
    unseen_items = prep(by_es[("original", "UNSEEN-PURE-v1")])
    pt = bleu_corpus(seen_items, range(622)) - bleu_corpus(unseen_items, range(622))
    out["SEEN-PURE-MATCHED-v1 (canonical, 622)"] = {"n_support": 622, "D": float(pt)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
