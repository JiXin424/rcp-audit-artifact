#!/usr/bin/env python3
"""Robustness and training-implementation diagnostics (reviewer M3 + minor).

Part 1 — Spearman readout-vs-competence, with and without the released point.
The main text reports Spearman rho=0.916 (n=31, includes the released evaluator
as an extreme point). Reviewer requests the correlation after removing the
released outlier and per-family.

Part 2 — Training-implementation diagnostics. Reviewer M3 asks for early-epoch
loss / lr / gradient-norm curves, batch packing, gradient accumulation, exact
config commit, to verify the reconstruction training behaved normally (small-
sample BLEU 100 only rules out loading/masking/decoding defects, not full-scale
training). We extract per-epoch train_nll / dev_nll / lr from each
training_log.json and report the early-epoch trajectory medians across
reconstructions vs the released evaluator's archived validation log.

Output: results/robustness_diagnostics.json
"""
import ast
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/full_readout_summary.json"
RELEASED_RO = ROOT / "results/full_readout/backTranslation_PHIX_model.json"
OUT = ROOT / "results/robustness_diagnostics.json"


def parse_bleu(v):
    """full_readout_summary stores dict-as-string; parse out the bleu field."""
    if isinstance(v, dict):
        return v.get("bleu")
    if isinstance(v, str):
        try:
            return ast.literal_eval(v).get("bleu")
        except Exception:
            return None
    return None


def part1_spearman():
    """Readout-vs-competence Spearman, full and released-excluded."""
    d = json.load(open(SUMMARY))
    # Released
    ro = json.load(open(RELEASED_RO))
    rel_train = ro["splits"]["train"]["bleu"]
    rel_dev = ro["splits"]["dev"]["bleu"]

    rows = []
    for ckpt, v in d.items():
        train_b = parse_bleu(v.get("train"))
        dev_b = parse_bleu(v.get("dev"))
        if train_b is None or dev_b is None:
            continue
        # family tag
        if ckpt.startswith("alpha_"):
            fam = "distillation"
        elif ckpt.startswith(("seed_", "cf_", "sf_", "conf_", "ls_",
                              "ladder_", "rescue_")):
            fam = "reconstruction"
        else:
            fam = "other"
        rows.append({"ckpt": ckpt, "train_bleu": train_b, "dev_bleu": dev_b,
                     "family": fam})

    # Add released
    rows_with_rel = rows + [{"ckpt": "released", "train_bleu": rel_train,
                             "dev_bleu": rel_dev, "family": "released"}]

    def spearman(r):
        t = np.array([x["train_bleu"] for x in r])
        dv = np.array([x["dev_bleu"] for x in r])
        rho, p = stats.spearmanr(t, dv)
        return float(rho), float(p), len(r)

    rho_all, p_all, n_all = spearman(rows_with_rel)
    rho_no_rel, p_no_rel, n_no_rel = spearman(rows)
    rho_recon, p_recon, n_recon = spearman([r for r in rows
                                            if r["family"] == "reconstruction"])
    rho_distill, p_distill, n_distill = spearman([r for r in rows
                                                 if r["family"] == "distillation"])

    print("=== Part 1: Spearman(train_readout_bleu, dev_bleu) ===")
    print(f"  All incl released:  rho={rho_all:+.3f} (p={p_all:.2e}, n={n_all})")
    print(f"  Excl released:      rho={rho_no_rel:+.3f} (p={p_no_rel:.2e}, n={n_no_rel})")
    print(f"  Reconstructions:    rho={rho_recon:+.3f} (p={p_recon:.2e}, n={n_recon})")
    print(f"  Distillation:       rho={rho_distill:+.3f} (p={p_distill:.2e}, n={n_distill})")

    return {
        "all_incl_released": {"rho": rho_all, "p": p_all, "n": n_all},
        "excl_released": {"rho": rho_no_rel, "p": p_no_rel, "n": n_no_rel},
        "reconstructions_only": {"rho": rho_recon, "p": p_recon, "n": n_recon},
        "distillation_only": {"rho": rho_distill, "p": p_distill, "n": n_distill},
        "released_point": {"train_bleu": rel_train, "dev_bleu": rel_dev},
    }


def part2_training_diagnostics():
    """Early-epoch train_nll/dev_nll/lr medians across reconstruction logs."""
    log_root = ROOT / "checkpoints"
    # Collect training_log.json across reconstruction families
    families = {
        "paper-derived (legacy)": "reconstructions",
        "config-faithful": "config_faithful",
        "step-faithful": "step_faithful",
        "confirmation": "confirmation",
        "long-schedule": "long_schedule",
        "rescue": "rescue",
        "ladder": "ladder",
        "distillation": "distillation",
    }
    all_logs = []
    family_logs = {f: [] for f in families}
    for fam_name, subdir in families.items():
        d = log_root / subdir
        if not d.exists():
            continue
        for tl in d.rglob("training_log.json"):
            try:
                j = json.load(open(tl))
                el = j.get("epochs_log", [])
                if not el:
                    continue
                rec = {
                    "path": str(tl.relative_to(ROOT)),
                    "family": fam_name,
                    "seed": j.get("seed"),
                    "batch_size": j.get("batch_size"),
                    "grad_accum": j.get("grad_accum"),
                    "effective_batch": (j.get("batch_size", 1)
                                        * j.get("grad_accum", 1)),
                    "selection": j.get("selection"),
                    "config_sha256": j.get("config_sha256"),
                    "txt_vocab_sha256": j.get("txt_vocab_sha256"),
                    "gls_vocab_sha256": j.get("gls_vocab_sha256"),
                    "skeleton_subsample": j.get("skeleton_subsample"),
                    "epochs_intended": j.get("epochs"),
                    "epochs_logged": len(el),
                    "started_at": j.get("started_at"),
                    "finished_at": j.get("finished_at"),
                    "best_epoch": j.get("best", {}).get("epoch"),
                    "best_dev_nll": j.get("best", {}).get("dev_nll"),
                    "epoch1_train_nll": el[0].get("train_nll"),
                    "epoch1_dev_nll": el[0].get("dev_nll"),
                    "epoch1_lr": el[0].get("lr"),
                    "epoch5_train_nll": (el[4].get("train_nll")
                                        if len(el) > 4 else None),
                    "epoch5_dev_nll": (el[4].get("dev_nll")
                                      if len(el) > 4 else None),
                    "epoch10_train_nll": (el[9].get("train_nll")
                                         if len(el) > 9 else None),
                    "epoch10_dev_nll": (el[9].get("dev_nll")
                                       if len(el) > 9 else None),
                    "epoch25_train_nll": (el[24].get("train_nll")
                                         if len(el) > 24 else None),
                    "epoch25_dev_nll": (el[24].get("dev_nll")
                                       if len(el) > 24 else None),
                    "final_train_nll": el[-1].get("train_nll"),
                    "final_dev_nll": el[-1].get("dev_nll"),
                    "lr_changes": int(np.sum([1 for e in el
                                              if e.get("improved") is False])),
                }
                all_logs.append(rec)
                family_logs[fam_name].append(rec)
            except Exception as e:
                print(f"  skip {tl}: {e}")

    print(f"\n=== Part 2: Training diagnostics ({len(all_logs)} logs) ===")

    # Implementation-detail summary (constant across same-recipe runs)
    impl = {}
    for fam, logs in family_logs.items():
        if not logs:
            continue
        impl[fam] = {
            "n_logs": len(logs),
            "batch_size": logs[0]["batch_size"],
            "grad_accum": logs[0]["grad_accum"],
            "effective_batch": logs[0]["effective_batch"],
            "selection": logs[0]["selection"],
            "skeleton_subsample": logs[0]["skeleton_subsample"],
            "config_sha256": logs[0]["config_sha256"],
        }
        print(f"  {fam}: n={len(logs)}, eff_batch={logs[0]['effective_batch']}, "
              f"selection={logs[0]['selection']}")

    # Early-epoch medians across reconstruction (non-distillation) logs
    recon_logs = [l for f, ls in family_logs.items() if f != "distillation"
                  for l in ls]
    epochs_idx = [("epoch1", 1), ("epoch5", 5), ("epoch10", 10), ("epoch25", 25)]
    medians = {}
    for tag, ep in epochs_idx:
        tn = [l[f"{tag}_train_nll"] for l in recon_logs
              if l[f"{tag}_train_nll"] is not None]
        dn = [l[f"{tag}_dev_nll"] for l in recon_logs
              if l[f"{tag}_dev_nll"] is not None]
        medians[tag] = {
            "epoch": ep,
            "train_nll_median": float(np.median(tn)) if tn else None,
            "train_nll_iqr": [float(np.percentile(tn, 25)),
                              float(np.percentile(tn, 75))] if tn else None,
            "dev_nll_median": float(np.median(dn)) if dn else None,
            "dev_nll_iqr": [float(np.percentile(dn, 25)),
                            float(np.percentile(dn, 75))] if dn else None,
            "n": len(tn),
        }
        m = medians[tag]
        print(f"  epoch {ep}: train_nll median={m['train_nll_median']:.3f} "
              f"(IQR {m['train_nll_iqr']}), dev_nll median={m['dev_nll_median']:.3f}, "
              f"n={m['n']}")

    # Note what is NOT recorded
    notes = {
        "recorded_per_epoch": ["train_nll", "dev_nll", "lr", "elapsed_s",
                               "improved"],
        "not_recorded": ["per-step loss", "gradient norm",
                         "per-token accuracy (teacher-forced, per epoch)",
                         "learning-rate warmup schedule (only plateau-reduction "
                         "lr values logged)"],
        "lr_schedule": "Adam(1e-3, wd 1e-3) with plateau x0.8 on dev NLL; "
                       "lr values visible per epoch in epochs_log",
    }

    return {
        "implementation_details_by_family": impl,
        "early_epoch_medians_reconstructions": medians,
        "n_reconstruction_logs": len(recon_logs),
        "field_notes": notes,
        "all_logs_summary": [{"path": l["path"], "family": l["family"],
                              "seed": l["seed"],
                              "eff_batch": l["effective_batch"],
                              "epochs_logged": l["epochs_logged"],
                              "best_epoch": l["best_epoch"],
                              "best_dev_nll": l["best_dev_nll"]}
                             for l in all_logs],
    }


def main():
    out = {
        "schema": "robustness-diagnostics-v1",
        "generated_by": "scripts/e_robustness_diagnostics.py",
    }
    out["part1_spearman_readout_competence"] = part1_spearman()
    out["part2_training_diagnostics"] = part2_training_diagnostics()
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
