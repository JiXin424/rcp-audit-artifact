#!/usr/bin/env python3
"""E5b: slot-F1 for GT/PURE under all 7 primary evaluators + donor-transcript pass-through control."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/ssd/xkb4/RCP/revision_20260729_round5/scripts")
from e5_slot_f1 import extract, f1, score_cell  # noqa: E402
from collections import Counter  # noqa: E402

ROOT = Path("/ssd/xkb4/RCP")
CELLS = ROOT / "revision_20260728_canonical_rebuild/outputs/evaluations/cells"
REG = ROOT / "revision_20260728_canonical_rebuild/registry/query_donor_registry.jsonl"
OUT = ROOT / "revision_20260729_round5/results/e5_slot_f1_all.json"

FAMS = ["NUM", "TEMP", "TIME", "PLACE", "EVENT", "ALL"]


def micro(rows):
    """rows: list of per-item dict fam -> (i, p, r) counts; return micro F1 per family."""
    agg = {f: [0, 0, 0] for f in FAMS}
    for row in rows:
        for f in FAMS:
            for j in range(3):
                agg[f][j] += row[f][j]
    out = {}
    for f in FAMS:
        i, p_, r_ = agg[f]
        p = i / p_ if p_ else 0.0
        r = i / r_ if r_ else 0.0
        out[f] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def cell_slot_counts(path):
    d = json.load(open(path))
    rows = []
    for it in d["metrics"]["items"]:
        rs = extract(it["reference"]); hs = extract(it["hypothesis"])
        row = {}
        for f in FAMS[:-1]:
            i = sum((rs[f] & hs[f]).values())
            row[f] = (i, sum(hs[f].values()), sum(rs[f].values()))
        ra = sum(rs.values(), Counter()); ha = sum(hs.values(), Counter())
        row["ALL"] = (sum((ra & ha).values()), sum(ha.values()), sum(ra.values()))
        rows.append({"id": it["id"], **row})
    return rows


def main():
    out = {"evaluators": {}, "donor_transcript_control": {}}
    for i, evname in enumerate(["original", "seed_101", "seed_202", "seed_303", "seed_404", "seed_505", "seed_606"]):
        cp = f"cp{i}"
        out["evaluators"][evname] = {
            "GT-v1": micro(cell_slot_counts(CELLS / f"{cp}_GT-v1.json")),
            "TN-PURE-v1": micro(cell_slot_counts(CELLS / f"{cp}_TN-PURE-v1.json")),
        }
        print(evname, "GT", {f: round(v, 3) for f, v in out["evaluators"][evname]["GT-v1"].items()},
              "PURE", {f: round(v, 3) for f, v in out["evaluators"][evname]["TN-PURE-v1"].items()})

    # Donor-transcript pass-through control: slot-F1(donor text, query reference)
    gt_items = json.load(open(CELLS / "cp0_GT-v1.json"))["metrics"]["items"]
    refs = {it["id"]: it["reference"] for it in gt_items}
    rows = []
    for line in open(REG):
        r = json.loads(line)
        qid, dt = r["query_id"], r["donor_text"]
        if qid not in refs:
            continue
        rs = extract(refs[qid]); hs = extract(dt)
        row = {}
        for f in FAMS[:-1]:
            i = sum((rs[f] & hs[f]).values())
            row[f] = (i, sum(hs[f].values()), sum(rs[f].values()))
        ra = sum(rs.values(), Counter()); ha = sum(hs.values(), Counter())
        row["ALL"] = (sum((ra & ha).values()), sum(ha.values()), sum(ra.values()))
        rows.append(row)
    out["donor_transcript_control"] = {"n": len(rows), "micro_f1": micro(rows)}
    print("donor transcript vs query ref:", {f: round(v, 3) for f, v in out["donor_transcript_control"]["micro_f1"].items()})
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
