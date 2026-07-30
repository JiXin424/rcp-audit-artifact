import csv
from pathlib import Path


EXPECTED_LOCAL_GT = {
    "original": 0.12777148564612426,
    "matched_101": 0.08570669944354585,
    "matched_202": 0.08813481380113616,
    "matched_303": 0.09669858031394363,
    "matched_404": 0.07192823265353804,
    "matched_505": 0.09047195411752910,
    "matched_606": 0.08089899257323295,
}


def test_every_transfer_cell_uses_its_evaluators_local_gt():
    path = Path(__file__).resolve().parents[1] / "results" / "transfer_cells.csv"
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 49
    for row in rows:
        assert abs(float(row["local_gt"]) - EXPECTED_LOCAL_GT[row["evaluator"]]) < 1e-12
        assert abs(float(row["margin"]) - (float(row["corpus_bleu"]) - float(row["local_gt"]))) < 1e-12
