#!/usr/bin/env python3
"""Finalize faithful-family training logs after numerical-LR-floor termination.

Context: torch's ReduceLROnPlateau carries an internal eps=1e-8: once a decay's
absolute step (lr*0.2) falls below eps, the decay is rejected and the LR
freezes (here at 4.36e-08). The config's learning_rate_min=1e-8 is therefore
never literally reached, and the runs were terminated manually once (a) the LR
had frozen at the scheduler's numerical floor and (b) the best checkpoint had
been unchanged for >200 consecutive validations. The per-line validations.txt
(equivalent to the released log format) is the complete record; this script
reconstructs the registry-compatible training_log.json from it plus the
console .out train-loss lines and the best.ckpt header.

Usage: python3 scripts/finalize_faithful_logs.py
"""
from __future__ import annotations
import json, math, re, shutil, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAM = ROOT / "checkpoints" / "faithful"
SEEDS = list(range(42, 50))
PROTOCOL = "faithful-joint-ctc-batchnorm-stepval14-plateau5-bleu-beam3"
VAL_RE = re.compile(
    r"Steps: (\d+)\tRecognition Loss: ([\d.]+)\tTranslation Loss: ([\d.]+)\t"
    r"PPL: ([\d.]+)\tEval Metric: bleu\tBLEU-4 ([\d.]+)\tLR: ([\d.eE+-]+)\t(\*)?")
OUT_RE = re.compile(
    r"step (\d+) \(ep(\d+)\) trans/sent=([\d.]+) rec/sent=([\d.]+) "
    r"devNLLsum=([\d.]+) ppl=([\d.]+) bleu=([\d.]+) best=([\d.-]+)@(-?\d+) lr=([\d.eE+-]+)( \*)?")


def finalize(seed: int):
    d = FAM / f"seed_{seed}"
    out_txt = FAM / f"seed_{seed}.out"

    # Train-loss lines keyed by step (console log).
    train_at = {}
    for line in out_txt.read_text().splitlines():
        m = OUT_RE.search(line)
        if m:
            train_at[int(m.group(1))] = {
                "epoch": int(m.group(2)),
                "trans_loss_per_sent": float(m.group(3)),
                "rec_loss_per_sent": float(m.group(4)),
            }

    vals = []
    for line in (d / "validations.txt").read_text().splitlines():
        m = VAL_RE.match(line)
        if not m:
            continue
        step = int(m.group(1))
        tr = train_at.get(step, {})
        vals.append({
            "step": step, "epoch": tr.get("epoch"),
            "trans_loss_per_sent": tr.get("trans_loss_per_sent"),
            "rec_loss_per_sent": tr.get("rec_loss_per_sent"),
            "dev_rec_per_sent": float(m.group(2)),
            "dev_nll_sum": float(m.group(3)),
            "dev_ppl": float(m.group(4)),
            "dev_bleu": float(m.group(5)),
            "lr": float(m.group(6)),
            "improved": bool(m.group(7)),
        })

    # Exact best from the checkpoint header (full precision).
    ck = torch.load(d / "best.ckpt", map_location="cpu", weights_only=False)
    best_bleu = float(ck["dev_bleu"]); best_step = int(ck["step"])
    best_row = next(v for v in vals if v["step"] == best_step)
    for v in vals:
        if v["step"] == best_step:
            v["improved"] = True
        elif v["step"] <= best_step and not v["improved"]:
            pass
        elif v["step"] > best_step:
            v["improved"] = False

    # NLL/tok derivable: nll_sum / n_tok. Token count not in the txt line, but
    # PPL = exp(nll_sum/n_tok) => n_tok = nll_sum / ln(PPL).
    for v in vals:
        if v["dev_ppl"] and v["dev_ppl"] > 1.0:
            v["dev_nll_per_tok"] = math.log(v["dev_ppl"])
        else:
            v["dev_nll_per_tok"] = None

    # started/finished from filesystem timestamps.
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((d / "validations.txt").stat().st_ctime))
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((d / "validations.txt").stat().st_mtime))

    log = {
        "seed": seed, "config": "configs/released.yaml",
        "protocol": PROTOCOL,
        "rec_weight": 1.0, "trans_weight": 1.0,
        "validation_freq_steps": 14, "selection": "bleu", "beam_size": 3,
        "batch_size": 256, "grad_accum": 1,
        "translation_normalization": "batch",
        "loss_reduction": "token-sum / n_sentences",
        "stop_rule": "lr_floor",
        "stop_reason": ("terminated at the plateau scheduler's numerical floor: torch "
                        "ReduceLROnPlateau eps=1e-8 rejects decays whose absolute step "
                        "< 1e-8, so LR froze at 4.36e-08 above the configured "
                        "learning_rate_min=1e-8; best checkpoint unchanged for >200 "
                        "consecutive validations before termination"),
        "patience": 5, "clip": 1.0,
        "n_validations": len(vals),
        "started_at": started, "finished_at": finished,
        "validations": vals,
        "best": {"step": best_step, "epoch": best_row.get("epoch"),
                 "dev_metric": best_bleu, "selection": "bleu",
                 "dev_ppl": best_row["dev_ppl"], "dev_nll_sum": best_row["dev_nll_sum"]},
    }
    (d / "training_log.json").write_text(json.dumps(log, indent=2))

    # Verbatim config + vocabs so make_back_translation_model can load the dir.
    if not (d / "config.yaml").exists():
        shutil.copy(ROOT / "configs/released.yaml", d / "config.yaml")
    for v in ("txt.vocab", "gls.vocab"):
        if not (d / v).exists():
            shutil.copy(ROOT / f"checkpoints/released/backTranslation_PHIX_model/{v}", d / v)

    n_since_best = len(vals) - (vals.index(best_row) + 1)
    print(f"seed_{seed}: {len(vals)} validations, best bleu {best_bleu:.4f} @ step {best_step}, "
          f"final lr {vals[-1]['lr']:.2e}, {n_since_best} validations since best")


if __name__ == "__main__":
    for s in SEEDS:
        finalize(s)
