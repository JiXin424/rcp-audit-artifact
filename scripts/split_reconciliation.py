#!/usr/bin/env python3
"""Round-34 split reconciliation (advisor comment #6).

The SLRTP2025 public repository reports the official PHOENIX-2014T split
sizes 7096/519/642, while the released pose materialization (and the shipped
evaluator training tensors) contain 7060/515/641. This script enumerates and
classifies every missing ID:

  official annotations (Sign-IDD-SLT base_annotations, gzip-pickled lists)
    vs. released pose materialization (SLRTP2025 data/*.pt)
    vs. upstream openpose skels extraction (slt_data_retrieval_jaccard)

For each missing item we record split, annotation length statistics (frames,
glosses, words), signer, and whether the item is also absent from the local
independent skels extraction (i.e., dropped before SLRTP packaging) or
present there (dropped by the SLRTP pose pipeline itself).

Output: results/split_reconciliation.json
"""
import gzip
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.slrtp_dataset import load_pickle  # noqa: E402

ANN = ROOT.parent / "standard_phoenix14t/Sign-IDD-SLT/base_annotations"
SKELS = ROOT.parent / "standard_phoenix14t/slt_data_retrieval_jaccard/PHOENIX2014T"
DATA = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
OUT = ROOT / "results/split_reconciliation.json"


def load_ann(split):
    with gzip.open(ANN / f"phoenix14t.{split}", "rb") as f:
        return pickle.load(f)


def ann_id(item):
    # Sign-IDD-SLT annotation names carry a "<split>/" prefix
    # (e.g. "train/11August_2010_..."); SLRTP names do not.
    n = item["name"]
    return n.split("/", 1)[1] if "/" in n else n


def load_skels_ids(split):
    """IDs present in the independent openpose skels extraction (gzip-pickled
    list of {name, signer, gloss, text, sign-tensor} records)."""
    p = SKELS / f"phoenix14t.skels.{split}.gz"
    if not p.exists():
        p = SKELS / f"phoenix14t.skels.{split}"
    with gzip.open(p, "rb") as f:
        d = pickle.load(f)
    return {it["name"] for it in d}, d


def stats(item):
    frames = None
    for key in ("frames", "num_frames", "length"):
        if key in item:
            frames = item[key]
            break
    if frames is None and "skeletons" in item and len(item["skeletons"]):
        frames = len(item["skeletons"])
    gls = item.get("gloss", "")
    txt = item.get("text", "")
    return {
        "frames": frames,
        "n_glosses": len(gls.split()) if isinstance(gls, str) else None,
        "n_words": len(txt.split()) if isinstance(txt, str) else None,
        "signer": item.get("signer", item.get("speaker", None)),
        "text": txt if isinstance(txt, str) else None,
    }


def main():
    report = {"splits": {}, "missing": [], "evaluator_manifest_check": {}}

    slrtp_train = load_pickle(DATA / "train.pt")
    slrtp_train_ids = [it["name"] for it in slrtp_train]
    report["evaluator_manifest_check"] = {
        "shipped_train_pt_items": len(slrtp_train_ids),
        "shipped_train_pt_unique": len(set(slrtp_train_ids)),
        "note": ("the evaluator bundle ships its own train.pt; the released "
                 "config's data_path is a private organizer path, so the "
                 "shipped tensors are the only public evidence of the "
                 "training manifest"),
    }
    slrtp = {
        "train": set(slrtp_train_ids),
        "dev": {it["name"] for it in load_pickle(DATA / "dev.pt")},
        "test": {it["name"] for it in load_pickle(DATA / "test.pt")},
    }

    for split in ("train", "dev", "test"):
        ann = load_ann(split)
        ann_ids = [ann_id(it) for it in ann]
        skel_ids, skel_records = load_skels_ids(split)
        ann_set, slrtp_set = set(ann_ids), slrtp[split]
        missing = sorted(ann_set - slrtp_set)
        extra = sorted(slrtp_set - ann_set)
        report["splits"][split] = {
            "official_count": len(ann_ids),
            "official_unique": len(ann_set),
            "slrtp_count": len(slrtp_set),
            "missing_from_slrtp": len(missing),
            "extra_in_slrtp_vs_official": len(extra),
            "also_absent_from_local_skels": sum(1 for i in missing if i not in skel_ids),
            "present_in_local_skels": sum(1 for i in missing if i in skel_ids),
            "local_skels_count": len(skel_ids),
            "extra_ids": extra,
        }
        ann_by_id = {ann_id(it): it for it in ann}
        for mid in missing:
            s = stats(ann_by_id[mid])
            s.update({
                "id": mid, "split": split,
                "in_local_skels": mid in skel_ids,
                "stage": "absent_before_slrtp_pose_release" if mid not in skel_ids
                         else "dropped_by_slrtp_pose_pipeline",
            })
            report["missing"].append(s)

    # distribution summaries for the missing train items vs the retained pool
    skel_by_id = {}
    for split in ("train", "dev", "test"):
        _, recs = load_skels_ids(split)
        for r in recs:
            skel_by_id[r["name"]] = r
    for split in ("train", "dev", "test"):
        miss = [m for m in report["missing"] if m["split"] == split]
        if miss:
            frames = [m["frames"] for m in miss]
            report["splits"][split]["missing_frames"] = {
                "min": min(frames), "median": sorted(frames)[len(frames) // 2],
                "max": max(frames)}
    # duplicate-text check: does any missing train item share its exact
    # normalized text with the retained pool (readout relevance)?
    import re
    def norm(t):
        return re.sub(r"\s+", " ", t.lower()).strip() if t else None
    retained_texts = Counter(norm(it["text"]) for it in slrtp_train)
    for m in report["missing"]:
        if m["split"] == "train":
            m["exact_text_copies_in_retained_train"] = retained_texts.get(norm(m["text"]), 0)

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["splits"], indent=1))
    print("missing item count:", len(report["missing"]))
    print("stage tally:", Counter(m["stage"] for m in report["missing"]))


if __name__ == "__main__":
    main()
