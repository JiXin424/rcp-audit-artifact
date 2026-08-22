#!/usr/bin/env python3
"""Update-level gradient diagnostics for the clipping sensitivity (round 35).

Addresses the reviewer's request to move beyond first-ten-step raw gradient
norms and quantify, over the training trajectory:
  (a) clipped-step proportion  -- fraction of optimizer steps whose pre-clip
      gradient norm exceeds the clip threshold;
  (b) pre-/post-clip gradient norms per step;
  (c) actual parameter-update norms ||theta_t - theta_{t-1}||_2 after Adam
      (what the optimizer really did), plus the cosine between the update
      vector and the pre-clip raw gradient;
  (d) CTC vs translation gradient-component norms (separate backwards on a
      retained graph, every --component-every steps);
  (e) matched-arm comparison: a dual lockstep run trains a clipped (1.0) and
      an unclipped model from the identical seed-42 init on the identical
      batch order and identical dropout masks (CUDA RNG state is forked per
      step), so per-step update norms and update cosines between the two
      regimes are directly comparable;
  (f) LR-matched control: single unclipped run with --lr override (update-
      norm matching is via lr because Adam steps scale linearly in lr).

The clipped arm of the dual run consumes the same RNG stream as a standalone
train_faithful.py run (the unclipped arm's dropout draws are taken from a
restored state), so its trajectory replicates the published protocol.

Modes:
  single --clip 1.0            full-horizon clipped run with diagnostics
  single --clip 0 (=none)      unclipped run (optionally --lr for control)
  dual                         clipped + unclipped lockstep, --max-steps

Outputs: results/grad_diag_<tag>.json + standard run dir (best.ckpt,
validations.txt, training_log.json) so eval_step_ckpts.py can score it.
"""
from __future__ import annotations
import argparse, json, math, os, shutil, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml, sacrebleu
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.slrtp_dataset import SLRTPDataset, Vocab, PAD_TOKEN, load_pickle
from src.models import build_model
from src.utils.seed import set_seed
from src.utils.hashing import sha256_file
from src.training.train_faithful import (collate_v2, evaluate_dev_faithful,
                                         format_validation_line, GlsBlank, GlsPad)

PROTOCOL = "grad-diagnostic-faithful-variant"


def forward_losses(model, batch, rec_weight, trans_weight, ctc_norm="sent"):
    """One forward, same math as train_faithful.compute_loss_faithful, but
    returns the loss TENSORS sharing a single graph (for component backwards).
    A single forward per step keeps the dropout RNG stream aligned with
    train_faithful.py runs (component backwards never re-run the model)."""
    B = batch["sgn"].size(0)
    out = model(sgn=batch["sgn"], sgn_mask=batch["sgn_mask"], sgn_lengths=batch["sgn_lengths"],
                txt_input=batch["txt_input"], txt_mask=batch["txt_mask"])
    decoder_outputs, gloss_probs = out
    logits = decoder_outputs[0].reshape(-1, decoder_outputs[0].size(-1))
    targets = batch["txt_output"].reshape(-1)
    # token-sum CE / n_sentences (translation_normalization: batch)
    trans_loss = F.cross_entropy(logits, targets, ignore_index=model.txt_pad_index,
                                 reduction="sum") / B
    rec_loss = None
    if gloss_probs is not None:
        rec_sum = F.ctc_loss(gloss_probs, batch["gls"], batch["sgn_lengths"].long(),
                             batch["gls_lengths"].long(), blank=GlsBlank, reduction="sum",
                             zero_infinity=True)
        if ctc_norm == "sent":
            denom = float(B)
        elif ctc_norm == "gtok":
            denom = float(batch["gls_lengths"].sum().item())
        elif ctc_norm == "frame":
            denom = float(batch["sgn_lengths"].sum().item())
        else:
            denom = 1.0
        rec_loss = rec_sum / max(denom, 1.0)
    joint = trans_weight * trans_loss + (rec_weight * rec_loss if rec_loss is not None else 0.0)
    return joint, trans_loss, rec_loss


class Arm:
    """One model + optimizer + scheduler + per-step diagnostic buffers."""

    def __init__(self, cfg, gls_vocab, txt_vocab, feat_size, device, clip, lr, patience, min_lr):
        self.model = build_model(cfg=cfg["model"], gls_vocab=gls_vocab, txt_vocab=txt_vocab,
                                 sgn_dim=feat_size, do_recognition=True, do_translation=True).to(device)
        self.model.txt_pad_index = txt_vocab.stoi[PAD_TOKEN]
        self.model.txt_vocab = txt_vocab
        self.model.gls_vocab = gls_vocab
        self.model.beam_size = int(cfg["testing"].get("eval_translation_beam_size", 3))
        self.model.beam_alpha = float(cfg["testing"].get("eval_translation_beam_alpha", -1))
        self.model.max_output_len = int(cfg["testing"].get("translation_max_output_length", 30))
        self.clip = clip  # None / 0 => unclipped
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                          betas=tuple(cfg["training"].get("betas", [0.9, 0.998])),
                                          weight_decay=float(cfg["training"].get("weight_decay", 0.001)))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=float(cfg["training"].get("decrease_factor", 0.8)),
            patience=patience, min_lr=min_lr)
        self.params = [p for p in self.model.parameters()]
        self.before = [p.detach().clone() for p in self.params]
        self.best_metric, self.best_step, self.no_improve = float("-inf"), -1, 0

    def step_with_diag(self, do_components):
        """Backward (+optional component backwards), clip, Adam step, measure."""
        m = self.model
        comp = {}
        joint, trans_t, rec_t = forward_losses(m, self._batch, self._rw, self._tw, self._cn)
        if do_components:
            # component backwards on the SAME graph: gradients zeroed between
            for p in self.params:
                p.grad = None
            trans_t.backward(retain_graph=True)
            comp["trans_grad_norm"] = _params_norm(self.params)
            for p in self.params:
                p.grad = None
            rec_t.backward(retain_graph=True)
            comp["ctc_grad_norm"] = _params_norm(self.params)
            for p in self.params:
                p.grad = None
        joint.backward(retain_graph=do_components)
        pre_norm = _params_norm(self.params)
        post_norm = pre_norm
        if self.clip:
            gn = torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.clip)
            post_norm = min(float(gn), self.clip)
        # snapshot params, step, keep grads for the u-vs-g cosine
        for buf, p in zip(self.before, self.params):
            buf.copy_(p.detach())
        self.optimizer.step()
        # update vector stats (grads still present -> cosine with raw gradient)
        u2, gu = 0.0, 0.0
        for buf, p in zip(self.before, self.params):
            d = p.detach() - buf
            u2 += float(d.pow(2).sum())
            if p.grad is not None:
                gu += float((d * p.grad).sum())
        un = math.sqrt(u2)
        diag = {"pre": pre_norm, "post": post_norm,
                "update_norm": un, "grad_update_cos": gu / (un * pre_norm) if un > 0 and pre_norm > 0 else None}
        diag.update(comp)
        self.optimizer.zero_grad(set_to_none=False)
        return diag


def _params_norm(params):
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.pow(2).sum())
    return math.sqrt(s)


def update_cosine_between(armA, armB):
    """cos(uA, uB) using each arm's `before` buffers vs current params."""
    dot = 0.0; nA = 0.0; nB = 0.0
    for bA, pA, bB, pB in zip(armA.before, armA.params, armB.before, armB.params):
        dA = pA.detach() - bA; dB = pB.detach() - bB
        dot += float((dA * dB).sum())
        nA += float(dA.pow(2).sum()); nB += float(dB.pow(2).sum())
    nA, nB = math.sqrt(nA), math.sqrt(nB)
    return dot / (nA * nB) if nA > 0 and nB > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "dual"], default="single")
    ap.add_argument("--config", default="configs/released.yaml")
    ap.add_argument("--train-pickle",
                    default="data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    ap.add_argument("--dev-pickle",
                    default="data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt")
    ap.add_argument("--txt-vocab", default="checkpoints/released/backTranslation_PHIX_model/txt.vocab")
    ap.add_argument("--gls-vocab", default="checkpoints/released/backTranslation_PHIX_model/gls.vocab")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--clip", type=float, default=None,
                    help="single mode: clip threshold; 0 disables. dual mode: clip of arm A (default 1.0).")
    ap.add_argument("--lr", type=float, default=None, help="LR override (LR-matched control).")
    ap.add_argument("--ctc-norm", choices=["sent", "gtok", "frame", "sum"], default="sent")
    ap.add_argument("--max-steps", type=int, default=0, help="hard step cap (0 = none)")
    ap.add_argument("--max-validations", type=int, default=1200)
    ap.add_argument("--stop-frozen", type=int, default=200)
    ap.add_argument("--component-every", type=int, default=14)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default=None, help="results/grad_diag_<tag>.json tag")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else int(cfg["training"]["random_seed"])
    epochs = args.epochs if args.epochs is not None else int(cfg["training"]["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["training"]["batch_size"])
    patience = args.patience if args.patience is not None else int(cfg["training"]["patience"])
    min_lr = float(cfg["training"].get("learning_rate_min", 1e-8))
    base_lr = float(cfg["training"]["learning_rate"])
    lr = args.lr if args.lr is not None else base_lr

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    feat_size = cfg["data"]["feature_size"]
    if isinstance(feat_size, list): feat_size = sum(feat_size)
    subsample = cfg["data"].get("skeleton_subsample", 2)
    max_sent_length = cfg["data"].get("max_sent_length", 400)
    txt_lowercase = cfg["data"].get("txt_lowercase", True)

    txt_vocab = Vocab.from_file(args.txt_vocab)
    gls_vocab = Vocab.from_file(args.gls_vocab)
    pad_idx = txt_vocab.stoi[PAD_TOKEN]

    train_ds = SLRTPDataset(args.train_pickle, txt_vocab=txt_vocab, gls_vocab=gls_vocab,
                            skeleton_subsample=subsample, max_sent_length=max_sent_length,
                            txt_lowercase=txt_lowercase)
    dev_ds = SLRTPDataset(args.dev_pickle, txt_vocab=txt_vocab, gls_vocab=gls_vocab,
                          skeleton_subsample=subsample, max_sent_length=max_sent_length,
                          txt_lowercase=txt_lowercase)
    dev_items_raw = load_pickle(args.dev_pickle)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=lambda b: collate_v2(b, pad_idx, gls_vocab))
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=args.workers,
                            collate_fn=lambda b: collate_v2(b, pad_idx, gls_vocab))

    rec_weight = float(cfg["training"].get("recognition_loss_weight", 1.0))
    trans_weight = float(cfg["training"].get("translation_loss_weight", 1.0))
    val_freq = int(cfg["training"].get("validation_freq", 14))

    if args.mode == "dual":
        # identical init: build A, snapshot state, build B, restore CPU RNG so
        # arm A's downstream RNG stream (epoch sampler seeds) matches a
        # standalone train_faithful.py run of the same seed.
        armA = Arm(cfg, gls_vocab, txt_vocab, feat_size, device,
                   clip=args.clip if args.clip is not None else 1.0, lr=lr,
                   patience=patience, min_lr=min_lr)
        state42 = {k: v.clone() for k, v in armA.model.state_dict().items()}
        cpu_rng_after_A = torch.get_rng_state()
        armB = Arm(cfg, gls_vocab, txt_vocab, feat_size, device,
                   clip=None, lr=lr, patience=patience, min_lr=min_lr)
        armB.model.load_state_dict(state42)
        torch.set_rng_state(cpu_rng_after_A)
        arms = {"clip1.0": armA, "unclipped": armB}
        print(f"Dual arms: clip1.0 + unclipped | init identical: "
              f"{all(torch.equal(a.detach(), b.detach()) for a, b in zip(armA.params, armB.params))}",
              flush=True)
    else:
        clip = args.clip if (args.clip is not None and args.clip > 0) else None
        arms = {f"clip{clip if clip else 'none'}": Arm(cfg, gls_vocab, txt_vocab, feat_size, device,
                                                       clip=clip, lr=lr, patience=patience, min_lr=min_lr)}

    for arm in arms.values():
        arm._rw, arm._tw, arm._cn = rec_weight, trans_weight, args.ctc_norm
    print(f"grad-diag: mode={args.mode} seed={seed} lr={lr} arms={list(arms)} "
          f"params={sum(p.numel() for p in next(iter(arms.values())).params):,}", flush=True)

    log = {"mode": args.mode, "seed": seed, "config": str(args.config), "lr": lr,
           "ctc_norm": args.ctc_norm, "component_every": args.component_every,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "steps": [], "best": {k: None for k in arms}, "stop_reason": None}

    val_fhs = {k: open(out_dir / f"validations_{k}.txt", "w") for k in arms}
    global_step = 0; stop_reason = None
    prev_lr = {k: None for k in arms}; lr_frozen = {k: 0 for k in arms}

    for epoch in range(1, epochs + 1):
        for k in arms: arms[k].model.train()
        for batch in train_loader:
            for kk in batch:
                if torch.is_tensor(batch[kk]): batch[kk] = batch[kk].to(device)
            do_comp = (global_step + 1) % args.component_every == 0
            step_diags = {}
            if args.mode == "dual":
                rng_state = torch.cuda.get_rng_state(device)
                step_diags["clip1.0"] = _forward_backward(arms["clip1.0"], batch, do_comp)
                torch.cuda.set_rng_state(rng_state, device)
                step_diags["unclipped"] = _forward_backward(arms["unclipped"], batch, do_comp)
                step_diags["update_cos_between"] = update_cosine_between(arms["clip1.0"], arms["unclipped"])
            else:
                name = next(iter(arms))
                step_diags[name] = _forward_backward(arms[name], batch, do_comp)
            global_step += 1
            step_diags["step"] = global_step
            log["steps"].append(step_diags)

            if global_step % val_freq == 0:
                for k, arm in arms.items():
                    dev = evaluate_dev_faithful(arm.model, dev_items_raw, dev_loader, device, subsample)
                    metric = dev["bleu"]
                    better = metric > arm.best_metric
                    if better:
                        arm.best_metric, arm.best_step, arm.no_improve = metric, global_step, 0
                        torch.save({"model_state": arm.model.state_dict(), "step": global_step,
                                    "epoch": epoch, "config": cfg, "dev_bleu": metric,
                                    "variant": f"grad-diag-{k}"},
                                   out_dir / f"best_{k}.ckpt")
                    else:
                        arm.no_improve += 1
                    arm.scheduler.step(metric)
                    lr_now = float(arm.optimizer.param_groups[0]["lr"])
                    val_fhs[k].write(format_validation_line(global_step, dev, lr_now, better) + "\n")
                    val_fhs[k].flush()
                    if better or len(log["steps"]) // val_freq % 20 == 0:
                        print(f"  [{k}] step {global_step} bleu={metric:.2f} best={arm.best_metric:.2f} "
                              f"lr={lr_now:.2e}{' *' if better else ''}", flush=True)
                    # frozen-LR external termination (published family's rule)
                    lr_frozen[k] = lr_frozen[k] + 1 if (prev_lr[k] is not None and lr_now == prev_lr[k]) else 0
                    prev_lr[k] = lr_now
                    if lr_frozen[k] >= args.stop_frozen and arm.no_improve > 200:
                        stop_reason = f"lr_frozen+best_unchanged[{k}]"; break
                if len(log["steps"]) // val_freq >= args.max_validations:
                    stop_reason = "max_validations"; break
            if args.max_steps and global_step >= args.max_steps:
                stop_reason = f"max_steps({args.max_steps})"; break
            if stop_reason: break
        if stop_reason: break

    for fh in val_fhs.values(): fh.close()
    log["stop_reason"] = stop_reason
    log["best"] = {k: {"step": arms[k].best_step, "dev_bleu": arms[k].best_metric} for k in arms}
    # summary statistics
    for k in arms:
        pre = [s[k]["pre"] for s in log["steps"]]
        thr = arms[k].clip
        log.setdefault("summary", {})[k] = {
            "n_steps": len(pre),
            "clip_threshold": thr,
            "clipped_steps": sum(1 for p in pre if thr and p > thr),
            "clipped_frac_overall": (sum(1 for p in pre if thr and p > thr) / len(pre)) if pre else None,
            "clipped_frac_first2828": (sum(1 for s in log["steps"][:2828] if thr and s[k]["pre"] > thr)
                                       / min(len(log["steps"]), 2828)) if pre else None,
            "pre_norm_median": float(np.median(pre)) if pre else None,
            "pre_norm_p10_p90": [float(np.percentile(pre, 10)), float(np.percentile(pre, 90))] if pre else None,
            "update_norm_median": float(np.median([s[k]["update_norm"] for s in log["steps"]])) or None,
            "mean_update_cos_to_grad": float(np.mean([s[k]["grad_update_cos"] for s in log["steps"]
                                                      if s[k].get("grad_update_cos") is not None]) ) or None,
        }
    if args.mode == "dual":
        uA = [s["clip1.0"]["update_norm"] for s in log["steps"]]
        uB = [s["unclipped"]["update_norm"] for s in log["steps"]]
        log["summary"]["dual"] = {
            "update_norm_ratio_median": float(np.median([b / a if a > 0 else float("nan")
                                                         for a, b in zip(uA, uB)])),
            "update_norm_ratio_first200_mean": float(np.mean([b / a if a > 0 else float("nan")
                                                              for a, b in zip(uA[:200], uB[:200])])),
            "update_cos_between_median": float(np.median([s["update_cos_between"] for s in log["steps"]
                                                          if s.get("update_cos_between") is not None])),
            "update_cos_between_first100_mean": float(np.mean([s["update_cos_between"] for s in log["steps"][:100]
                                                               if s.get("update_cos_between") is not None])),
        }
    log["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "grad_diag_log.json").write_text(json.dumps(log, indent=2))
    tag = args.tag or (out_dir.parent.name + "_" + out_dir.name)
    dest = Path("results") / f"grad_diag_{tag}.json"
    dest.write_text(json.dumps(log, indent=2))
    # copy config/vocabs for eval_step_ckpts-style loading
    for k in arms:
        d = out_dir / k; d.mkdir(exist_ok=True)
        if not (d / "config.yaml").exists(): shutil.copy(args.config, d / "config.yaml")
        for vn, vs in [("txt.vocab", Path(args.txt_vocab)), ("gls.vocab", Path(args.gls_vocab))]:
            if not (d / vn).exists() and vs.exists(): shutil.copy(vs, d / vn)
        if (out_dir / f"best_{k}.ckpt").exists():
            lnk = d / "best.ckpt"
            if lnk.exists() or lnk.is_symlink(): lnk.unlink()
            lnk.symlink_to((out_dir / f"best_{k}.ckpt").resolve())
    print(f"\nDone (stop={stop_reason}). best per arm: "
          + ", ".join(f"{k}: {log['best'][k]}" for k in arms) + f" | {dest}", flush=True)


def _forward_backward(arm, batch, do_components):
    arm._batch = batch
    return arm.step_with_diag(do_components)


if __name__ == "__main__":
    main()
