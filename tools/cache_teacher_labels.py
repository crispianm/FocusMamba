#!/usr/bin/env python3
"""
Pre-cache Teacher Pseudo-Labels
=================================

Runs all enabled teacher models *once* over the training dataset and saves
the depth predictions to disk.  During subsequent training runs the Trainer
loads these cached tensors instead of running expensive teacher inference on
every step, which eliminates the dominant bottleneck (~70-80% of wall time
at 2.68 s/it comes from synchronous ViT-L teacher forwards).

Cache layout
------------
    <cache_dir>/
    └── <video_id>/          e.g. AbandonedFactory/P000
        └── <start_frame>/   e.g. 0
            ├── video_depth_anything.pt   # float16 (1, T, H, W)
            ├── depth_anything_v3.pt
            └── depth_pro.pt

Usage
-----
    # Dry run — list teachers and exit
    python tools/cache_teacher_labels.py \\
        --config configs/experiments/tartanair_v2.yaml \\
        --dry-run

    # Full cache generation (GPU required for teacher inference)
    python tools/cache_teacher_labels.py \\
        --config configs/experiments/tartanair_v2.yaml \\
        --cache-dir /projects/b5dh/data/teacher_cache/tartanair_v2

    # Resume a partially-complete run (skip clips that already have all files)
    python tools/cache_teacher_labels.py \\
        --config configs/experiments/tartanair_v2.yaml \\
        --cache-dir /projects/b5dh/data/teacher_cache/tartanair_v2 \\
        --resume

    # Limit to a subset of clips for testing
    python tools/cache_teacher_labels.py \\
        --config configs/experiments/tartanair_v2.yaml \\
        --cache-dir /tmp/test_cache \\
        --max-clips 50

Notes
-----
* Predictions are saved as **float16** to save ~50 % disk space vs float32.
  The training dataloader casts them back to float32 on load.
* The cache is independent of training resolution: teacher outputs are saved
  at the resolution that the teacher itself uses, then bilinearly upsampled
  to the training resolution by the dataloader.  This means you can change
  ``image_size`` in the config without re-running the cacher.
* Progress is displayed with tqdm and logged to ``cache_teacher_labels.log``
  in the cache directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path)))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def _all_cached(clip_cache_dir: Path, teacher_names: List[str]) -> bool:
    """Return True if every teacher file already exists for this clip."""
    return all((clip_cache_dir / f"{n}.npy").exists() for n in teacher_names)


# ---------------------------------------------------------------------------
# Dataset wrapper that returns (clip_tensor, video_id, start_frame) only
# ---------------------------------------------------------------------------

class _ClipOnlyDataset(torch.utils.data.Dataset):
    """Thin wrapper that exposes only the fields needed for caching."""

    def __init__(self, inner_dataset):
        self._ds = inner_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        item = self._ds[idx]
        return {
            "frames": item["frames"],          # (3, T, H, W) float32
            "video_id": item["video_id"],       # str
            "start_frame": item["start_frame"], # int
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-cache teacher pseudo-labels for FocusMamba training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/experiments/tartanair_v2.yaml",
                        help="Experiment config YAML (teachers section must be populated)")
    parser.add_argument("--cache-dir", default=None,
                        help="Root directory for the cache. Overrides config value if set.")
    parser.add_argument("--device", default="cuda",
                        help="Device for teacher inference (cuda / cpu)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for teacher inference. 1 is usually safest for "
                             "teachers that internally manage temporal windows.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for reading frames")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Stop after processing this many clips (useful for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip clips that already have all teacher files")
    parser.add_argument("--dry-run", action="store_true",
                        help="List teachers and dataset size then exit immediately")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Which split to cache")
    parser.add_argument("--envs", nargs="+", default=None, metavar="ENV",
                        help="Override config envs: restrict caching to these environment "
                             "names (e.g. --envs AbandonedFactory Ocean).  Useful for "
                             "parallelising across SLURM array tasks.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")

    # Resolve cache dir
    cache_dir_str = (
        args.cache_dir
        or cfg.get("data", {}).get("teacher_cache_dir")
        or cfg.get("training", {}).get("teacher_cache_dir")
    )
    if not cache_dir_str and not args.dry_run:
        parser.error(
            "Specify --cache-dir or set data.teacher_cache_dir in the config YAML."
        )

    cache_dir = Path(cache_dir_str) if cache_dir_str else Path("/tmp/teacher_cache")

    _setup_logging(cache_dir / "cache_teacher_labels.log")
    log = logging.getLogger(__name__)

    # -----------------------------------------------------------------------
    # Build teacher models
    # -----------------------------------------------------------------------
    from models.teachers import build_teacher

    max_depth: float = float(cfg.get("data", {}).get("max_depth", 80.0))
    log.info("max_depth (clip threshold for cached outputs): %.1f m", max_depth)

    teacher_cfgs: List[dict] = cfg.get("teachers", [])
    teachers: Dict[str, object] = {}
    for t_cfg in teacher_cfgs:
        name = t_cfg.get("name", "")
        if not name or not t_cfg.get("enabled", True):
            continue
        try:
            teacher = build_teacher(name, t_cfg, device=str(device))
            teachers[name] = teacher
            log.info("Loaded teacher: %s", name)
        except Exception as exc:
            log.warning("Could not load teacher '%s': %s", name, exc)

    if not teachers:
        log.error("No teachers loaded — nothing to cache.  Check config teachers section.")
        sys.exit(1)

    teacher_names = list(teachers.keys())
    log.info("Teachers to cache: %s", teacher_names)

    # -----------------------------------------------------------------------
    # Build dataset (train split)
    # -----------------------------------------------------------------------
    data_cfg = cfg.get("data", {})
    dataset_type = data_cfg.get("dataset", "tartanair_v2")

    if dataset_type == "tartanair_v2":
        from dataloader.tartanair_v2 import TartanAirV2Dataset
        # --envs CLI flag overrides the config value (used by SLURM array tasks)
        envs_filter = args.envs if args.envs is not None else data_cfg.get("envs", None)
        if envs_filter:
            log.info("Restricting to environments: %s", envs_filter)
        raw_dataset = TartanAirV2Dataset(
            root=data_cfg["root"],
            num_frames=data_cfg.get("num_frames", 8),
            # Use native teacher resolution; the cacher saves at this resolution.
            image_size=tuple(data_cfg.get("image_size", [518, 518])),
            max_trajectories=data_cfg.get("max_trajectories", None),
            clip_stride=data_cfg.get("clip_stride", 8),
            frame_stride=data_cfg.get("frame_stride", 1),
            split=args.split,
            val_fraction=data_cfg.get("val_fraction", 0.1),
            seed=data_cfg.get("seed", 42),
            difficulty=data_cfg.get("difficulty", "Data_easy"),
            camera=data_cfg.get("camera", "lcam_front"),
            max_depth=data_cfg.get("max_depth", 80.0),
            envs=envs_filter,
        )
    else:
        log.error("Unsupported dataset type '%s' for teacher caching.", dataset_type)
        sys.exit(1)

    dataset = _ClipOnlyDataset(raw_dataset)
    n_total = len(dataset) if args.max_clips is None else min(len(dataset), args.max_clips)

    log.info("Dataset split=%s  clips=%d  (using %d)", args.split, len(dataset), n_total)

    if args.dry_run:
        log.info("--- DRY RUN complete ---")
        return

    # -----------------------------------------------------------------------
    # DataLoader — batch_size=1 by default because teachers manage their own
    # temporal windowing.  Larger batches work if VRAM allows.
    # -----------------------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        collate_fn=_collate_fn,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Cache root: %s", cache_dir)

    # -----------------------------------------------------------------------
    # Caching loop
    # -----------------------------------------------------------------------
    n_cached = 0
    n_skipped = 0
    pbar = tqdm(total=n_total, unit="clip", desc="Caching", dynamic_ncols=True)

    for batch in loader:
        videos = batch["video_id"]        # list[str], length B
        starts = batch["start_frame"]     # list[int], length B
        frames = batch["frames"].to(device)  # (B, 3, T, H, W)

        for b_idx in range(frames.shape[0]):
            vid = videos[b_idx]
            sf = int(starts[b_idx])

            clip_cache = cache_dir / vid / str(sf)

            if args.resume and _all_cached(clip_cache, teacher_names):
                n_skipped += 1
                pbar.update(1)
                if n_cached + n_skipped >= n_total:
                    break
                continue

            clip_cache.mkdir(parents=True, exist_ok=True)
            single_frame = frames[b_idx : b_idx + 1]  # (1, 3, T, H, W)

            for t_name, teacher in teachers.items():
                out_path = clip_cache / f"{t_name}.npy"
                try:
                    with torch.no_grad():
                        depth = teacher.predict(single_frame)  # (1, 1, T, H, W)
                    # Clip to valid metric range: caps sky/far pixels that teachers
                    # output at extreme values (e.g. DepthPro sky → ~10 000 m).
                    # These pixels will be excluded from training by the GT mask
                    # (sky GT = 0 → masked out), but clipping prevents fp16 overflow
                    # and keeps the cache interpretable.
                    depth = depth.clamp(min=0.0, max=max_depth)
                    # Save as float16 numpy array — np.load(mmap_mode='r') lets
                    # DataLoader workers memory-map the file instead of deserialising
                    # a pickle, which is faster and shares OS pages across workers.
                    arr = depth.squeeze(0).half().cpu().numpy()  # (1, T, H, W) float16
                    np.save(str(out_path), arr)
                except Exception as exc:
                    log.warning("Teacher '%s' failed on %s/%d: %s", t_name, vid, sf, exc)

            n_cached += 1
            pbar.update(1)
            if n_cached + n_skipped >= n_total:
                break

        if n_cached + n_skipped >= n_total:
            break

    pbar.close()
    log.info(
        "Done.  Cached=%d  Skipped(already existed)=%d  Cache dir: %s",
        n_cached, n_skipped, cache_dir,
    )


# ---------------------------------------------------------------------------
# Custom collate to handle str video IDs in batches
# ---------------------------------------------------------------------------

def _collate_fn(batch):
    """Collate that keeps video_id as a plain list of strings."""
    frames = torch.stack([b["frames"] for b in batch])
    video_ids = [b["video_id"] for b in batch]
    start_frames = [b["start_frame"] for b in batch]
    return {"frames": frames, "video_id": video_ids, "start_frame": start_frames}


if __name__ == "__main__":
    main()
