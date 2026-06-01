#!/usr/bin/env python3
"""Phase B: evaluate a per-epoch checkpoint ladder across a degradation
severity grid x multiple degradation seeds.

Purpose: distinguish the two competing explanations for the post-epoch-1
degraded-val erosion:
  H1 (real robustness loss / range collapse): later epochs are worse across
      ALL severities and ALL seeds, accompanied by predicted-range (pred_depth_span)
      collapse and boundary_f1 loss.
  H2 (fixed-seed / severity artifact): later epochs are worse only at the single
      fixed validation seed, or only at high severity.

Method: build the val set returning CLEAN frames, then apply v2_lowlight
degradation ourselves at a chosen severity_scale with a chosen RNG seed per clip.
This decouples the degradation realization from the train/val clip split, so we
can sweep severity and seed independently without changing which clips are used.
Reuses the trainer's exact metric helpers and the SSI aligner.

Output: one CSV row per (epoch, severity, seed) with mean aligned_abs_rel, raw
abs_rel, pred_depth_span, boundary_f1, fdv over the clips.
"""
import argparse
import csv
import glob
import os
import re
import sys
from copy import deepcopy

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloader.degradation import build_degradation
from dataloader.tartanair_v2 import TartanAirV2Dataset
from models import build_model
from training.losses.scale_shift import ScaleShiftInvariantLoss
from training.trainer import (
    compute_aligned_depth_metrics,
    compute_boundary_metrics,
    compute_depth_distribution_metrics,
    compute_depth_metrics,
)


def build_val_dataset(cfg, which):
    """Build the degraded val set returning CLEAN frames (we degrade ourselves)."""
    data_cfg = cfg.get("data", {}) or {}
    img = tuple(int(v) for v in data_cfg.get("image_size", (518, 518)))
    nf = int(data_cfg.get("val_num_frames", data_cfg.get("num_frames", 8)))
    seed = int(data_cfg.get("seed", 42))
    vf = float(data_cfg.get("val_fraction", 0.1))
    max_depth = float(data_cfg.get("max_depth", 80.0))
    # locate the matching validation spec for VKITTI scene/variant/camera
    if which == "vkitti":
        from dataloader.vkitti import VKITTIDataset
        spec = None
        for s in (cfg.get("validation", {}) or {}).get("sets", []):
            if s.get("type") == "vkitti" and "degraded" in s.get("name", ""):
                spec = s
                break
        if spec is None:
            spec = (data_cfg.get("extra_train_datasets") or [{}])[0]
        return VKITTIDataset(
            root=spec["root"], num_frames=nf, image_size=img,
            clip_stride=int(spec.get("clip_stride", data_cfg.get("clip_stride", 8))),
            frame_stride=int(spec.get("frame_stride", data_cfg.get("frame_stride", 1))),
            split="val", val_fraction=vf, seed=seed, max_depth=max_depth,
            scenes=spec.get("scenes"), variants=spec.get("variants"),
            cameras=spec.get("cameras"), degradation=None,
            return_clean_and_degraded=False,
        )
    return TartanAirV2Dataset(
        root=str(data_cfg["root"]), num_frames=nf, image_size=img,
        max_trajectories=data_cfg.get("max_val_trajectories"),
        clip_stride=int(data_cfg.get("clip_stride", 8)),
        frame_stride=int(data_cfg.get("frame_stride", 1)),
        split="val", val_fraction=vf, seed=seed,
        difficulty=str(data_cfg.get("difficulty", "Data_easy")),
        camera=str(data_cfg.get("camera", "lcam_front")), max_depth=max_depth,
        envs=data_cfg.get("envs"), teacher_cache_dir=None, degradation=None,
        return_clean_and_degraded=False,
    )


def load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    sd = {k.removeprefix("module."): v for k, v in sd.items() if torch.is_tensor(v)}
    model.load_state_dict(sd, strict=False)


def discover_ladder(ckpt_dir, baseline):
    items = []  # (epoch_int, path)
    if baseline:
        items.append((0, baseline))
    for p in glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")):
        m = re.search(r"epoch_(\d+)\.pt$", p)
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return items


@torch.inference_mode()
def eval_one(model, dataset, degradation, *, severity, seed, device, max_depth,
             aligner, max_clips, batch_size):
    """Mirror trainer.validate(): metrics computed on whole-batch tensors and
    averaged across batches. We degrade the clean clips ourselves at the chosen
    severity_scale + per-clip RNG seed."""
    degradation.set_severity_scale(float(severity))
    n = len(dataset) if max_clips is None else min(max_clips, len(dataset))
    acc = {k: [] for k in ("aligned_abs_rel", "abs_rel", "pred_depth_span", "boundary_f1")}
    idx = 0
    while idx < n:
        clips, depths, masks = [], [], []
        for j in range(idx, min(idx + batch_size, n)):
            sample = dataset[j]
            clean = sample["frames"]            # (C,T,H,W)
            if float(severity) <= 0.0:
                deg_cthw = clean                # severity 0 == clean reference
            else:
                tchw = clean.permute(1, 0, 2, 3).contiguous()
                rng = np.random.RandomState((int(seed) * 100003 + j) % (2 ** 31 - 1))
                deg_cthw = degradation(tchw, rng=rng).permute(1, 0, 2, 3).contiguous()
            clips.append(deg_cthw)
            depths.append(sample["depth"])
            masks.append(sample["mask"])
        frames = torch.stack(clips).to(device)                   # (B,C,T,H,W)
        depth = torch.stack(depths).to(device).float()
        mask = torch.stack(masks).to(device)
        out = model(frames)
        pred = (out["depth"] if isinstance(out, dict) else out).float()
        am = compute_aligned_depth_metrics(pred, depth, aligner=aligner, mask=mask,
                                           max_depth=max_depth)
        dm = compute_depth_metrics(pred, depth, mask=mask)
        dist = compute_depth_distribution_metrics(pred, mask=mask, max_depth=max_depth)
        bd = compute_boundary_metrics(pred, depth, mask=mask)
        acc["aligned_abs_rel"].append(am["aligned_abs_rel"])
        acc["abs_rel"].append(dm["abs_rel"])
        acc["pred_depth_span"].append(dist.get("pred_depth_span", float("nan")))
        acc["boundary_f1"].append(bd.get("boundary_f1", float("nan")))
        idx += batch_size
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in acc.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--baseline-checkpoint", default=None,
                    help="Optional pre-train checkpoint, recorded as epoch 0.")
    ap.add_argument("--dataset", choices=("vkitti", "tartanair"), default="vkitti")
    ap.add_argument("--severities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max-clips", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    max_depth = float((cfg.get("data", {}) or {}).get("max_depth", 80.0))
    ssi_mode = (cfg.get("loss", {}) or {}).get("ssi_mode", "l2")
    aligner = ScaleShiftInvariantLoss(mode=ssi_mode)

    severities = [float(s) for s in args.severities.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    dataset = build_val_dataset(cfg, args.dataset)
    print("val clips available: %d (dataset=%s)" % (len(dataset), args.dataset), flush=True)

    # model skeleton (weights loaded per checkpoint)
    cfg_model = deepcopy(cfg)
    cfg_model.setdefault("model", {})
    cfg_model["model"]["checkpoint_path"] = None
    cfg_model["model"]["strict_checkpoint"] = False
    model = build_model(cfg_model).to(device).eval()

    degradation = build_degradation(cfg)
    if degradation is None:
        raise SystemExit("config has no degradation block to sweep")

    ladder = discover_ladder(args.checkpoint_dir, args.baseline_checkpoint)
    if not ladder:
        raise SystemExit("no epoch_*.pt found in %s" % args.checkpoint_dir)
    print("ladder:", [(e, os.path.basename(p)) for e, p in ladder], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["dataset", "epoch", "severity", "seed", "n_clips",
              "aligned_abs_rel", "abs_rel", "pred_depth_span", "boundary_f1"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for epoch, path in ladder:
            load_weights(model, path)
            for sev in severities:
                for seed in seeds:
                    res, n = eval_one(model, dataset, degradation, severity=sev,
                                      seed=seed, device=device, max_depth=max_depth,
                                      aligner=aligner, max_clips=args.max_clips,
                                      batch_size=args.batch_size)
                    row = dict(dataset=args.dataset, epoch=epoch, severity=sev,
                               seed=seed, n_clips=n, **{k: "%.6g" % res[k] for k in
                               ("aligned_abs_rel", "abs_rel", "pred_depth_span", "boundary_f1")})
                    w.writerow(row)
                    f.flush()
                    print("ep%-2d sev%.2f seed%d  aligned=%.4f abs_rel=%.4f span=%.2f bf1=%.4f"
                          % (epoch, sev, seed, res["aligned_abs_rel"], res["abs_rel"],
                             res["pred_depth_span"], res["boundary_f1"]), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
