"""
Live Distillation Dataset
=========================

Dataset returning (clean_frames, degraded_frames, depth) triplets for
multi-teacher distillation training.

Supports multiple data sources discovered automatically:
  - TartanAirV2: metric depth outdoor/indoor sequences
      <seq>/image_left/*.png  +  <seq>/depth_left/*.npy  (float32 metres)
  - YouTube-VOS: clean video sequences without GT depth
      <video_id>/*.jpg  (flat frame directory)
  - Generic (FocusMamba-style):
      <seq>/frames_sharp/*.png  +  <seq>/depth_maps/*.npz  (key: 'depth')

Each ``__getitem__`` returns::

    {
        "clean_frames":    FloatTensor (C, T, H, W) in [0, 1]
        "degraded_frames": FloatTensor (C, T, H, W) in [0, 1]
        "depth":           FloatTensor (1, T, H, W) in metres, or None if no GT
        "metadata":        dict
    }

Usage::

    from dataloader.live_distill_dataset import build_live_distill_dataloaders
    loaders = build_live_distill_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader   = loaders.get("val")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataloader.degradation import LowLightDegradation


# ---------------------------------------------------------------------------
# Frame / depth I/O helpers
# ---------------------------------------------------------------------------

def _load_frame(path: Path) -> np.ndarray:
    """Load an RGB frame as uint8 (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_depth_npy(path: Path) -> np.ndarray:
    """Load depth from a .npy file (float32 H×W, metres)."""
    return np.load(str(path)).astype(np.float32)


def _load_depth_npz(path: Path) -> np.ndarray:
    """Load depth from a .npz file (float32 H×W, metres)."""
    with np.load(str(path)) as f:
        key = "depth" if "depth" in f else f.files[0]
        return f[key].astype(np.float32)


def _load_depth(path: Path) -> np.ndarray:
    """Dispatch depth loading by file extension."""
    if path.suffix == ".npy":
        return _load_depth_npy(path)
    return _load_depth_npz(path)


# ---------------------------------------------------------------------------
# Clip record helpers — discover sequences in a data root
# ---------------------------------------------------------------------------

def _tartanair_clips(
    root: Path,
    window_length: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """Discover TartanAirV2 clips.

    Expected layout::

        <root>/<sequence>/image_left/*.png
        <root>/<sequence>/depth_left/*.npy   (optional — absent = no GT)
    """
    clips: List[Dict[str, Any]] = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        frames_dir = seq_dir / "image_left"
        if not frames_dir.exists():
            continue
        frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        if len(frame_paths) < window_length:
            continue

        depth_dir = seq_dir / "depth_left"
        depth_paths: Optional[List[Path]] = None
        if depth_dir.exists():
            _d = sorted(depth_dir.glob("*.npy")) + sorted(depth_dir.glob("*.npz"))
            if len(_d) >= len(frame_paths):
                depth_paths = _d[: len(frame_paths)]

        n = len(frame_paths)
        for start in range(0, n - window_length + 1, stride):
            clips.append(
                {
                    "frames": frame_paths[start : start + window_length],
                    "depths": (
                        depth_paths[start : start + window_length]
                        if depth_paths else None
                    ),
                    "source": seq_dir.name,
                }
            )
    return clips


def _youtube_vos_clips(
    root: Path,
    window_length: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """Discover YouTube-VOS clips (no GT depth).

    Expected layout::

        <root>/<video_id>/*.jpg
    """
    clips: List[Dict[str, Any]] = []
    for vid_dir in sorted(root.iterdir()):
        if not vid_dir.is_dir():
            continue
        frame_paths = (
            sorted(vid_dir.glob("*.jpg"))
            + sorted(vid_dir.glob("*.png"))
        )
        if len(frame_paths) < window_length:
            continue
        n = len(frame_paths)
        for start in range(0, n - window_length + 1, stride):
            clips.append(
                {
                    "frames": frame_paths[start : start + window_length],
                    "depths": None,
                    "source": vid_dir.name,
                }
            )
    return clips


def _generic_clips(
    root: Path,
    window_length: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """Discover generic FocusMamba-style clips.

    Expected layout::

        <root>/<seq>/frames_sharp/*.png
        <root>/<seq>/depth_maps/*.npz   (optional)
    """
    clips: List[Dict[str, Any]] = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        frames_dir = seq_dir / "frames_sharp"
        if not frames_dir.exists():
            continue
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        if len(frame_paths) < window_length:
            continue

        depth_dir = seq_dir / "depth_maps"
        depth_paths: Optional[List[Path]] = None
        if depth_dir.exists():
            _d = sorted(depth_dir.glob("depth_*.npz")) + sorted(depth_dir.glob("*.npy"))
            if len(_d) >= len(frame_paths):
                depth_paths = _d[: len(frame_paths)]

        n = len(frame_paths)
        for start in range(0, n - window_length + 1, stride):
            clips.append(
                {
                    "frames": frame_paths[start : start + window_length],
                    "depths": (
                        depth_paths[start : start + window_length]
                        if depth_paths else None
                    ),
                    "source": seq_dir.name,
                }
            )
    return clips


def _discover_clips(
    root: Path,
    dataset_type: str,
    window_length: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """Route to the correct clip-discovery function based on *dataset_type*."""
    if dataset_type == "tartanair":
        return _tartanair_clips(root, window_length, stride)
    if dataset_type == "youtube_vos":
        return _youtube_vos_clips(root, window_length, stride)
    # Auto-detect or generic
    clips = _tartanair_clips(root, window_length, stride)
    if clips:
        return clips
    clips = _youtube_vos_clips(root, window_length, stride)
    if clips:
        return clips
    return _generic_clips(root, window_length, stride)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class _Augmentation:
    """Random horizontal flip + random crop."""

    def __init__(self, crop_h: int, crop_w: int, rng: np.random.RandomState):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.rng = rng

    def __call__(
        self,
        frames: np.ndarray,  # (T, H, W, C)
        depth: Optional[np.ndarray],  # (T, H, W) or None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        T, H, W, C = frames.shape

        # Pad if spatial dims are smaller than the crop target
        frames, depth = self._pad(frames, depth)
        H, W = frames.shape[1], frames.shape[2]

        # Random horizontal flip
        if self.rng.rand() > 0.5:
            frames = frames[:, :, ::-1, :].copy()
            if depth is not None:
                depth = depth[:, :, ::-1].copy()

        # Random crop
        y0 = self.rng.randint(0, max(1, H - self.crop_h + 1))
        x0 = self.rng.randint(0, max(1, W - self.crop_w + 1))
        frames = frames[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w, :]
        if depth is not None:
            depth = depth[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w]

        return frames, depth

    def _pad(
        self,
        frames: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Pad frames (and depth) to at least (crop_h, crop_w) with edge values."""
        T, H, W, C = frames.shape
        pad_h = max(0, self.crop_h - H)
        pad_w = max(0, self.crop_w - W)
        if pad_h > 0 or pad_w > 0:
            frames = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            if depth is not None:
                depth = np.pad(depth, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        return frames, depth

    def center_crop(
        self,
        frames: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        frames, depth = self._pad(frames, depth)
        T, H, W, C = frames.shape
        y0 = max(0, (H - self.crop_h) // 2)
        x0 = max(0, (W - self.crop_w) // 2)
        frames = frames[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w, :]
        if depth is not None:
            depth = depth[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w]
        return frames, depth


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LiveDistillDataset(Dataset):
    """Sliding-window video dataset for live multi-teacher distillation.

    Provides clean frames (for teacher inference) and optionally degraded
    frames (for student training).  Ground-truth depth is included when
    available (e.g. TartanAirV2) and set to ``None`` otherwise (YouTube-VOS).

    Args:
        clips: List of clip records, each ``{"frames": [Path], "depths": [Path] | None,
               "source": str}``.
        crop_size: (H, W) spatial crop applied to every sample.
        augment: If True, apply random flip + crop; otherwise centre-crop.
        degradation: Optional :class:`LowLightDegradation` applied to clean
            frames to produce degraded frames.  If *None*, ``degraded_frames``
            is a copy of ``clean_frames``.
        seed: Base seed for numpy RNG.
    """

    def __init__(
        self,
        clips: List[Dict[str, Any]],
        crop_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        degradation: Optional[LowLightDegradation] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.clips = clips
        self.crop_size = crop_size
        self.augment = augment
        self.degradation = degradation
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip = self.clips[idx]
        frame_paths: List[Path] = clip["frames"]
        depth_paths: Optional[List[Path]] = clip["depths"]

        # --- Load frames ---
        frames = np.stack([_load_frame(p) for p in frame_paths], axis=0)  # (T, H, W, 3)

        # --- Load depths ---
        depth: Optional[np.ndarray] = None
        if depth_paths is not None:
            depth = np.stack([_load_depth(p) for p in depth_paths], axis=0)  # (T, H, W)

        # --- Augmentation / crop ---
        aug = _Augmentation(self.crop_size[0], self.crop_size[1], self.rng)
        if self.augment:
            frames, depth = aug(frames, depth)
        else:
            frames, depth = aug.center_crop(frames, depth)

        # --- Convert to float tensor (C, T, H, W) in [0, 1] ---
        clean_t = (
            torch.from_numpy(frames.copy())
            .permute(3, 0, 1, 2)
            .float()
            .div(255.0)
        )  # (C, T, H, W)

        # --- Apply degradation to produce degraded frames ---
        if self.degradation is not None:
            # degradation expects (T, C, H, W)
            frames_tchw = clean_t.permute(1, 0, 2, 3).contiguous()
            degraded_tchw = self.degradation(frames_tchw, rng=self.rng)
            degraded_t = degraded_tchw.permute(1, 0, 2, 3).contiguous()
        else:
            degraded_t = clean_t.clone()

        # --- Depth tensor ---
        depth_t: Optional[torch.Tensor] = None
        if depth is not None:
            depth_t = torch.from_numpy(depth.copy()).unsqueeze(0).float()  # (1, T, H, W)

        return {
            "clean_frames": clean_t,       # (C, T, H, W)
            "degraded_frames": degraded_t,  # (C, T, H, W)
            "depth": depth_t,               # (1, T, H, W) or None
            "metadata": {
                "source": clip.get("source", ""),
                "frame_paths": [str(p) for p in frame_paths],
            },
        }


# ---------------------------------------------------------------------------
# Collate function that handles optional None depth
# ---------------------------------------------------------------------------

def live_distill_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate that stacks tensors and handles None depth gracefully."""
    clean = torch.stack([b["clean_frames"] for b in batch])      # (B, C, T, H, W)
    degraded = torch.stack([b["degraded_frames"] for b in batch])  # (B, C, T, H, W)

    # Depth: only stack if ALL samples have depth; otherwise leave as None
    depths = [b["depth"] for b in batch]
    if all(d is not None for d in depths):
        depth_batch: Optional[torch.Tensor] = torch.stack(depths)  # (B, 1, T, H, W)
    else:
        depth_batch = None

    return {
        "clean_frames": clean,
        "degraded_frames": degraded,
        "depth": depth_batch,
        "metadata": [b["metadata"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Worker init — re-seed per worker to avoid duplicate augmentations
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    base_seed = getattr(dataset, "seed", 42)
    new_seed = (base_seed + worker_id * 1000) % (2 ** 32)
    dataset.rng = np.random.RandomState(new_seed)


# ---------------------------------------------------------------------------
# Builder — config-driven entry point used by train.py
# ---------------------------------------------------------------------------

def build_live_distill_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Build train / val DataLoaders from a YAML config dict.

    Config keys read (all optional with defaults):

    ``data``::

        datasets:
          - path: /mnt/DATA/tartanairv2
            type: tartanair           # tartanair | youtube_vos | auto
            split: train              # train | val | both (default: both → ratio split)
          - path: /mnt/DATA/youtube-vos/train/JPEGImages
            type: youtube_vos
        window_length: 8
        clip_stride: 4
        resolution: [256, 256]
        train_ratio: 0.9
        split_seed: 42

    ``training``::

        batch_size: 4
        num_workers: 4
        seed: 42

    ``degradation``::

        enabled: true
        lux_level: 10.0
        ...

    Returns:
        Dict with ``"train"`` and (optionally) ``"val"`` DataLoader entries.
    """
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    window_length: int = data_cfg.get("window_length", 8)
    stride: int = data_cfg.get("clip_stride", 4)
    resolution = data_cfg.get("resolution", [256, 256])
    crop_size: Tuple[int, int] = (int(resolution[0]), int(resolution[1]))
    train_ratio: float = data_cfg.get("train_ratio", 0.9)
    split_seed: int = data_cfg.get("split_seed", train_cfg.get("seed", 42))

    batch_size: int = train_cfg.get("batch_size", 4)
    num_workers: int = train_cfg.get("num_workers", 4)

    # Build degradation pipeline
    from dataloader.degradation import build_degradation
    degradation = build_degradation(cfg)

    # Collect all clip records
    all_clips: List[Dict[str, Any]] = []

    dataset_entries = data_cfg.get("datasets", [])
    if not dataset_entries:
        # Fallback: single data_root key (legacy / config.yaml style)
        data_root = data_cfg.get("data_root") or cfg.get("data_root")
        if data_root:
            dataset_entries = [{"path": data_root, "type": "auto"}]

    for entry in dataset_entries:
        root = Path(entry["path"])
        if not root.exists():
            print(f"WARNING: dataset path does not exist, skipping: {root}")
            continue
        ds_type = entry.get("type", "auto")
        clips = _discover_clips(root, ds_type, window_length, stride)
        if not clips:
            print(f"WARNING: no clips found in {root} (type={ds_type})")
        else:
            print(f"  {ds_type}: {root.name}  →  {len(clips)} clips")
        all_clips.extend(clips)

    if not all_clips:
        raise RuntimeError(
            "No clips found. Check 'data.datasets' paths in your config."
        )

    # Deterministic split
    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(len(all_clips))
    n_train = max(1, int(len(all_clips) * train_ratio))
    train_clips = [all_clips[i] for i in indices[:n_train]]
    val_clips = [all_clips[i] for i in indices[n_train:]]

    print(f"Clips — train: {len(train_clips)}, val: {len(val_clips)}")

    loaders: Dict[str, DataLoader] = {}

    for split_name, clips in [("train", train_clips), ("val", val_clips)]:
        if not clips:
            continue
        ds = LiveDistillDataset(
            clips=clips,
            crop_size=crop_size,
            augment=(split_name == "train"),
            degradation=degradation if split_name == "train" else None,
            seed=split_seed,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            collate_fn=live_distill_collate_fn,
            pin_memory=True,
            drop_last=(split_name == "train"),
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            persistent_workers=(num_workers > 0),
        )

    return loaders
