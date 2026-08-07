#!/usr/bin/env python3
"""Re-decode GT and PURE with beam=3 and compute matched-subset analysis.

The stored cell files use greedy decoding. The paper's canonical numbers use
beam=3. This script re-decodes GT-v1 and TN-PURE-v1 with beam=3 using
back_translate(), then computes the matched-subset reference sensitivity.
"""
import json, sys, time, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.slrtp_dataset import load_pickle
from src.models import make_back_translation_model, back_translate

DATA_DIR = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
MODEL_DIR = ROOT / "checkpoints/released/backTranslation_PHIX_model"
CSV_FULL = ROOT / "data/sacrebird/test_full_annotations_sacrebirdphoenix.csv"
CSV_HC = ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
TRAIN_PT = DATA_DIR / "train.pt"
TEST_PT = DATA_DIR / "test.pt"
OUT = ROOT / "results/beam3_matched_subset.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def load_csv(path):
    out = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        try:
            conf = float(parts[2])
        except ValueError:
            conf = 0.0
        out[parts[0]] = (parts[1], conf)
    return out


def load_and_decode_test(model):
    """Decode GT (test poses) and PURE (donor retrieval) with beam=3."""
    test_items = load_pickle(TEST_PT)
    train_items = load_pickle(TRAIN_PT)

    # Build donor text lookup
    train_texts = {it["name"]: it["text"] for it in train_items}

    # Build text-nearest donor registry (Jaccard)
    print("Building donor registry...", flush=True)
    import re
    def normalize(s):
        s = re.sub(r'\s+', ' ', s.strip().lower())
        return set(s.split())

    train_norm = [(it["name"], normalize(it["text"])) for it in train_items]

    # For each test item, find nearest donor
    gt_poses = []
    pure_poses = []
    test_ids = []
    test_refs = []
    donor_texts = {}

    t0 = time.time()
    for item in test_items:
        qid = item["name"]
        q_norm = normalize(item["text"])
        # Text-nearest donor (exclude exact text matches)
        best_jaccard = -1
        best_donor = None
        best_donor_text = None
        for did, d_norm in train_norm:
            if item["text"].strip().lower() == train_texts[did].strip().lower():
                continue  # exclude exact text match
            inter = len(q_norm & d_norm)
            union = len(q_norm | d_norm)
            jac = inter / union if union > 0 else 0
            if jac > best_jaccard:
                best_jaccard = jac
                best_donor = did
                best_donor_text = train_texts[did]

        if best_donor is None:
            continue

        # Get donor pose
        donor_item = next(it for it in train_items if it["name"] == best_donor)
        donor_pose = donor_item["poses_3d"]
        if not isinstance(donor_pose, torch.Tensor):
            donor_pose = torch.as_tensor(np.asarray(donor_pose, dtype=np.float32))

        # GT pose (apply [::2] subsampling to match model's 12.5 fps training)
        gt_pose = item["poses_3d"]
        if not isinstance(gt_pose, torch.Tensor):
            gt_pose = torch.as_tensor(np.asarray(gt_pose, dtype=np.float32))
        gt_pose = gt_pose[::2]  # 25fps -> 12.5fps

        gt_poses.append(gt_pose)
        donor_pose = donor_pose[::2]  # 25fps -> 12.5fps
        pure_poses.append(donor_pose)
        test_ids.append(qid)
        test_refs.append(item["text"])
        donor_texts[qid] = best_donor_text

    print(f"Registry built in {time.time()-t0:.1f}s ({len(test_ids)} items)", flush=True)

    # Decode GT with beam=3
    print("Decoding GT with beam=3...", flush=True)
    t0 = time.time()
    gt_hyps = back_translate(model, gt_poses)
    print(f"  GT done in {time.time()-t0:.1f}s", flush=True)

    # Decode PURE with beam=3
    print("Decoding PURE with beam=3...", flush=True)
    t0 = time.time()
    pure_hyps = back_translate(model, pure_poses)
    print(f"  PURE done in {time.time()-t0:.1f}s", flush=True)

    return test_ids, test_refs, gt_hyps, pure_hyps, donor_texts


def main():
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    print(f"Using GPU {gpu}", flush=True)

    print("Loading model...", flush=True)
    model = make_back_translation_model(str(MODEL_DIR))

    ids, refs, gt_hyps, pure_hyps, donor_texts = load_and_decode_test(model)

    # Load human references
    human_full = load_csv(CSV_FULL)
    human_hc = load_csv(CSV_HC)

    ids_set = set(ids)
    refs_orig = dict(zip(ids, refs))
    hyps_gt = dict(zip(ids, gt_hyps))
    hyps_pure = dict(zip(ids, pure_hyps))
    refs_hf = {k: v[0] for k, v in human_full.items() if k in ids_set}
    refs_hc = {k: v[0] for k, v in human_hc.items() if k in ids_set}

    ids_full = sorted(set(ids) & set(refs_hf))
    ids_hc = sorted(set(ids) & set(refs_hc))

    print(f"\nFull: {len(ids_full)}, HC: {len(ids_hc)}", flush=True)

    # Verify headline
    gt_full_bleu = BLEU.corpus_score([hyps_gt[i] for i in ids_full],
                                      [[refs_orig[i] for i in ids_full]]).score
    pure_full_bleu = BLEU.corpus_score([hyps_pure[i] for i in ids_full],
                                        [[refs_orig[i] for i in ids_full]]).score
    print(f"Full 641 beam=3: GT={gt_full_bleu:.2f}, PURE={pure_full_bleu:.2f}, "
          f"gap={pure_full_bleu-gt_full_bleu:+.2f}", flush=True)

    out = {"decoding": "beam=3", "model": "released"}

    # Matched confidence=1 subset
    for label, ids_sub, rh_map in [
        ("matched_confidence1", ids_hc, {k: refs_hc[k] for k in ids_hc}),
        ("full_641", ids_full, {k: refs_hf[k] for k in ids_full}),
    ]:
        ro = {k: refs_orig[k] for k in ids_sub}
        gt_orig = BLEU.corpus_score([hyps_gt[i] for i in ids_sub], [[ro[i] for i in ids_sub]]).score
        pure_orig = BLEU.corpus_score([hyps_pure[i] for i in ids_sub], [[ro[i] for i in ids_sub]]).score
        gt_human = BLEU.corpus_score([hyps_gt[i] for i in ids_sub], [[rh_map[i] for i in ids_sub]]).score
        pure_human = BLEU.corpus_score([hyps_pure[i] for i in ids_sub], [[rh_map[i] for i in ids_sub]]).score
        orig_gap = pure_orig - gt_orig
        human_gap = pure_human - gt_human
        att = orig_gap - human_gap
        att_pct = att / orig_gap * 100 if orig_gap != 0 else None

        print(f"\n{label} ({len(ids_sub)} items):", flush=True)
        print(f"  Orig: GT={gt_orig:.2f}, PURE={pure_orig:.2f}, gap={orig_gap:+.2f}", flush=True)
        print(f"  Human: GT={gt_human:.2f}, PURE={pure_human:.2f}, gap={human_gap:+.2f}", flush=True)
        print(f"  Attenuation: {att:.2f} ({att_pct:.1f}%)", flush=True)

        out[label] = {
            "n": len(ids_sub),
            "original": {"gt": gt_orig, "pure": pure_orig, "gap": orig_gap},
            "human": {"gt": gt_human, "pure": pure_human, "gap": human_gap},
            "attenuation": {"absolute": att, "pct": att_pct},
        }

    # Paired bootstrap on confidence=1
    print("\nComputing paired bootstrap on confidence=1...", flush=True)
    rng = np.random.RandomState(42)
    N = len(ids_hc)
    gt = [hyps_gt[i] for i in ids_hc]
    pure = [hyps_pure[i] for i in ids_hc]
    ro = [refs_orig[i] for i in ids_hc]
    rh = [refs_hc[i] for i in ids_hc]
    atts = []
    for _ in range(10000):
        idx = rng.randint(0, N, N)
        og = BLEU.corpus_score([pure[j] for j in idx], [[ro[j] for j in idx]]).score - \
             BLEU.corpus_score([gt[j] for j in idx], [[ro[j] for j in idx]]).score
        hg = BLEU.corpus_score([pure[j] for j in idx], [[rh[j] for j in idx]]).score - \
             BLEU.corpus_score([gt[j] for j in idx], [[rh[j] for j in idx]]).score
        atts.append(og - hg)
    atts = np.array(atts)
    out["matched_confidence1"]["paired_attenuation_bootstrap"] = {
        "mean": float(atts.mean()),
        "ci_lo": float(np.percentile(atts, 2.5)),
        "ci_hi": float(np.percentile(atts, 97.5)),
    }
    print(f"  Paired attenuation: {atts.mean():.2f} [{np.percentile(atts, 2.5):.2f}, "
          f"{np.percentile(atts, 97.5):.2f}]", flush=True)

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nSaved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
