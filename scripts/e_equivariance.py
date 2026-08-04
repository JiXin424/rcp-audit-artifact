#!/usr/bin/env python3
"""Output-equivariance check for input-side pose permutation [E:E0006].

The input-side permutation showed BLEU collapses when poses are shuffled.
But we should also verify E(pose_j, id_i) ≈ E(pose_j, id_j): the hypothesis
produced when pose_j is decoded in item i's slot should match the original
decode of pose_j in its own slot j. This proves the output FOLLOWS the
swapped-in pose tensor, not just that it's sensitive to corruption.

Additionally checks:
  - Pose length/mask travels with the tensor (not reconstructed from ID)
  - Per-item hash comparison between original and equivariance decode
"""
import json, sys, os, time, hashlib
from pathlib import Path
import numpy as np
import torch
import sacrebleu

ROOT = Path("/ssd/xkb4/RCP")
sys.path.insert(0, str(ROOT))
from src.data.slrtp_dataset import load_pickle
from src.models import make_back_translation_model, back_translate

DATA_DIR = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
MODEL_DIR = ROOT / "checkpoints/released/backTranslation_PHIX_model"
OUT = ROOT / "results/equivariance.json"


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    N = 100  # items for the check

    print(f"Loading first {N} training items...", flush=True)
    train_items = load_pickle(DATA_DIR / "train.pt")[:N]

    ids = [it["name"] for it in train_items]
    refs = [it["text"] for it in train_items]
    poses = []
    for it in train_items:
        p = it["poses_3d"]
        if not isinstance(p, torch.Tensor):
            p = torch.as_tensor(np.asarray(p, dtype=np.float32))
        poses.append(p)
    # Subsample
    poses_sub = [p[::2] for p in poses]

    print("Loading model...", flush=True)
    model = make_back_translation_model(str(MODEL_DIR))

    # Step 1: Decode all N items in original order → original_hyps[i]
    print(f"\nStep 1: Decoding {N} items in original order...", flush=True)
    t0 = time.time()
    original_hyps = back_translate(model, poses_sub)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Step 2: Apply a fixed permutation: item i gets pose perm[i]
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    # perm[i] = j means slot i receives pose_j
    poses_perm = [poses_sub[perm[i]] for i in range(N)]

    print(f"\nStep 2: Decoding {N} items with permuted poses...", flush=True)
    t0 = time.time()
    permuted_hyps = back_translate(model, poses_perm)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Step 3: Equivariance check: permuted_hyps[i] should ≈ original_hyps[perm[i]]
    # Because slot i received pose_{perm[i]}, the model should output what it
    # would output for pose_{perm[i]} = original_hyps[perm[i]]
    matches = 0
    hash_matches = 0
    mismatches = []
    for i in range(N):
        j = perm[i]  # slot i received pose_j
        hyp_perm_slot_i = permuted_hyps[i].strip()
        hyp_orig_slot_j = original_hyps[j].strip()

        if hyp_perm_slot_i == hyp_orig_slot_j:
            matches += 1
            # Also check hash
            h1 = hashlib.sha256(hyp_perm_slot_i.encode()).hexdigest()[:16]
            h2 = hashlib.sha256(hyp_orig_slot_j.encode()).hexdigest()[:16]
            if h1 == h2:
                hash_matches += 1
        else:
            if len(mismatches) < 5:
                mismatches.append({
                    "slot_i": i,
                    "donor_j": j,
                    "id_i": ids[i],
                    "id_j": ids[j],
                    "hyp_perm_slot_i": hyp_perm_slot_i[:120],
                    "hyp_orig_slot_j": hyp_orig_slot_j[:120],
                    "identical": False,
                })

    equivariance_rate = matches / N
    print(f"\n=== EQUIVARIANCE CHECK ===")
    print(f"N = {N}")
    print(f"Exact string matches (permuted_hyps[i] == original_hyps[perm[i]]): "
          f"{matches}/{N} ({equivariance_rate*100:.1f}%)")
    print(f"Hash matches: {hash_matches}/{N}")

    if mismatches:
        print(f"\nFirst {len(mismatches)} mismatches:")
        for m in mismatches:
            print(f"  slot {m['slot_i']} (id={m['id_i'][:30]}) ← pose from donor {m['donor_j']} (id={m['id_j'][:30]})")
            print(f"    perm_slot_i: {m['hyp_perm_slot_i']}")
            print(f"    orig_slot_j: {m['hyp_orig_slot_j']}")

    # Step 4: Length/mask travel check
    # Verify that the permuted decode length matches the donor's original length,
    # not the slot's original length. This confirms pose length info travels with tensor.
    len_matches_donor = 0
    len_matches_slot = 0
    for i in range(N):
        j = perm[i]
        perm_len = len(permuted_hyps[i].split())
        donor_len = len(original_hyps[j].split())
        slot_len = len(original_hyps[i].split())
        if perm_len == donor_len:
            len_matches_donor += 1
        if perm_len == slot_len:
            len_matches_slot += 1

    print(f"\n=== LENGTH/MASK TRAVEL CHECK ===")
    print(f"Permuted decode length matches DONOR original: "
          f"{len_matches_donor}/{N} ({len_matches_donor/N*100:.1f}%)")
    print(f"Permuted decode length matches SLOT original: "
          f"{len_matches_slot}/{N} ({len_matches_slot/N*100:.1f}%)")
    print("(If length follows the pose tensor, donor match should be high, slot match low)")

    # Step 5: Score permuted decode against slot's reference (should be ~0)
    # and against donor's reference (should be high if equivariant)
    BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                                  effective_order=False, force=True)
    bleu_vs_slot_ref = BLEU.corpus_score(
        permuted_hyps, [[refs[i] for i in range(N)]]).score
    bleu_vs_donor_ref = BLEU.corpus_score(
        permuted_hyps, [[refs[perm[i]] for i in range(N)]]).score

    print(f"\n=== SCORING ===")
    print(f"Permuted decode vs SLOT reference: BLEU = {bleu_vs_slot_ref:.2f} (should be ~0)")
    print(f"Permuted decode vs DONOR reference: BLEU = {bleu_vs_donor_ref:.2f} (should be high if equivariant)")
    print(f"Original decode vs own reference:   BLEU = {BLEU.corpus_score(original_hyps, [refs]).score:.2f}")

    # Save results
    results = {
        "n": N,
        "equivariance_rate": equivariance_rate,
        "hash_matches": hash_matches,
        "len_matches_donor": len_matches_donor,
        "len_matches_slot": len_matches_slot,
        "bleu_vs_slot_ref": bleu_vs_slot_ref,
        "bleu_vs_donor_ref": bleu_vs_donor_ref,
        "perm": perm.tolist(),
        "mismatches_sample": mismatches,
    }
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
