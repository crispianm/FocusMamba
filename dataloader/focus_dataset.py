"""
FocusMamba Dataset & DataLoader
================================

Expected directory structure under `data_root`:
    data/
        scene_001/
            frames.npy      # (N, H, W, 3) uint8, memory-mappable
            depth_maps.npy   # (N, H, W)    float32, from Video Depth Anything
        scene_002/
            ...

All .npy files should be saved with ``np.save`` so they can be opened with
``np.load(..., mmap_mode='r')``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Utility: deterministic scene-level split
# ---------------------------------------------------------------------------

def _split_scenes(
    scene_dirs: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Split scene directories deterministically into train / val / test."""
    rng = np.random.RandomState(seed)
    scenes = sorted(scene_dirs, key=lambda p: p.name)
    indices = rng.permutation(len(scenes))
    n_train = int(len(scenes) * train_ratio)
    n_val = int(len(scenes) * val_ratio)
    return {
        "train": [scenes[i] for i in indices[:n_train]],
        "val": [scenes[i] for i in indices[n_train : n_train + n_val]],
        "test": [scenes[i] for i in indices[n_train + n_val :]],
    }


# ---------------------------------------------------------------------------
# Circle-of-Confusion helpers
# ---------------------------------------------------------------------------

def _sample_lens_params(
    rng: np.random.RandomState,
    focal_length_range: Tuple[float, float] = (24.0, 85.0),
    aperture_range: Tuple[float, float] = (1.4, 8.0),
) -> Tuple[float, float]:
    """Sample focal length (uniform) and aperture (log-uniform)."""
    f = rng.uniform(*focal_length_range)
    log_lo, log_hi = np.log(aperture_range[0]), np.log(aperture_range[1])
    N = float(np.exp(rng.uniform(log_lo, log_hi)))
    return f, N


def _sample_focus_distance(
    depth_map: np.ndarray,
    rng: np.random.RandomState,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> float:
    """Sample a focus distance from a Beta(alpha, beta) mapped to [d_min, d_max]."""
    valid = depth_map[depth_map > 0]
    if len(valid) == 0:
        return 1.0  # fallback
    d_min, d_max = float(valid.min()), float(valid.max())
    if d_max - d_min < 1e-6:
        return d_min
    u = rng.beta(alpha, beta)
    return d_min + u * (d_max - d_min)


def compute_focus_map(
    depth: np.ndarray,
    focal_length: float,
    aperture: float,
    s_focus: float,
    max_coc: float,
) -> np.ndarray:
    """Compute a soft focus map using a physically-based Circle of Confusion model.

    CoC(d) = (f^2 / (N * S_focus)) * |d - S_focus| / d

    Returns:
        focus_map in [0, 1] where 1 = perfectly in-focus.
    """
    d = depth.astype(np.float64).copy()
    d[d < 1e-8] = 1e-8  # avoid division by zero
    coc = (focal_length ** 2 / (aperture * s_focus)) * np.abs(d - s_focus) / d
    coc_norm = coc / max_coc
    focus_map = 1.0 - np.clip(coc_norm, 0.0, 1.0)
    return focus_map.astype(np.float32)


# ---------------------------------------------------------------------------
# ROI computation from focus map
# ---------------------------------------------------------------------------

def _focus_map_to_roi(focus_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Derive a normalised [x1, y1, x2, y2] ROI from the mean focus map.

    The ROI is the bounding box of the in-focus region (focus > threshold)
    normalised to [0, 1] relative to image dimensions.  Falls back to
    full-image if nothing is above threshold.
    """
    # focus_map: (T, H, W) or (H, W)
    if focus_map.ndim == 3:
        mask = focus_map.mean(axis=0) >= threshold
    else:
        mask = focus_map >= threshold

    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    x1, x2 = float(xs.min()) / W, float(xs.max() + 1) / W
    y1, y2 = float(ys.min()) / H, float(ys.max() + 1) / H
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

class _TrainAugmentation:
    """Random horizontal flip, random crop, temporal jitter (±1 frame)."""

    def __init__(self, crop_h: int, crop_w: int, rng: np.random.RandomState):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.rng = rng

    def __call__(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        focus_maps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        T, H, W, C = frames.shape

        # --- random horizontal flip ---
        if self.rng.rand() > 0.5:
            frames = frames[:, :, ::-1, :].copy()
            depth_maps = depth_maps[:, :, ::-1].copy()
            focus_maps = focus_maps[:, :, ::-1].copy()

        # --- random crop ---
        if H > self.crop_h:
            y0 = self.rng.randint(0, H - self.crop_h + 1)
        else:
            y0 = 0
        if W > self.crop_w:
            x0 = self.rng.randint(0, W - self.crop_w + 1)
        else:
            x0 = 0
        frames = frames[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w, :]
        depth_maps = depth_maps[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w]
        focus_maps = focus_maps[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w]

        return frames, depth_maps, focus_maps


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FocusDataset(Dataset):
    """Sliding-window video focus-map dataset with depth-guided CoC generation.

    Each sample is a clip of ``window_length`` consecutive frames.
    """

    def __init__(
        self,
        scene_dirs: List[Path],
        window_length: int = 8,
        stride: int = 4,
        crop_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        # CoC parameters
        max_coc: float = 0.05,
        focal_length_range: Tuple[float, float] = (24.0, 85.0),
        aperture_range: Tuple[float, float] = (1.4, 8.0),
        beta_alpha: float = 2.0,
        beta_beta: float = 2.0,
        seed: int = 42,
    ):
        super().__init__()
        self.window_length = window_length
        self.stride = stride
        self.crop_size = crop_size
        self.augment = augment
        self.max_coc = max_coc
        self.focal_length_range = focal_length_range
        self.aperture_range = aperture_range
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.rng = np.random.RandomState(seed)

        # Build an index: list of (scene_dir, start_frame)
        self._clips: List[Tuple[Path, int]] = []
        for sd in sorted(scene_dirs, key=lambda p: p.name):
            frames_path = sd / "frames.npy"
            if not frames_path.exists():
                continue
            # Peek at shape without loading (memmap)
            fp = np.load(str(frames_path), mmap_mode="r")
            n_frames = fp.shape[0]
            for start in range(0, max(1, n_frames - window_length + 1), stride):
                self._clips.append((sd, start))

    # ---- helpers ----
    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_dir, start = self._clips[idx]
        T = self.window_length

        # Temporal jitter for training (±1)
        if self.augment:
            jitter = int(self.rng.randint(-1, 2))  # -1, 0, or 1
        else:
            jitter = 0

        frames_all = np.load(str(scene_dir / "frames.npy"), mmap_mode="r")
        depth_all = np.load(str(scene_dir / "depth_maps.npy"), mmap_mode="r")
        n_frames_total = frames_all.shape[0]

        s = max(0, min(start + jitter, n_frames_total - T))
        e = s + T

        frames = np.array(frames_all[s:e])       # (T, H, W, 3) uint8
        depth_maps = np.array(depth_all[s:e])     # (T, H, W) float32

        # --- Generate focus map via CoC ---
        f_len, aperture = _sample_lens_params(
            self.rng, self.focal_length_range, self.aperture_range
        )
        # Use mean depth across the clip for focus distance sampling
        s_focus = _sample_focus_distance(
            depth_maps.mean(axis=0), self.rng, self.beta_alpha, self.beta_beta
        )
        focus_maps = np.stack(
            [
                compute_focus_map(depth_maps[t], f_len, aperture, s_focus, self.max_coc)
                for t in range(T)
            ],
            axis=0,
        )  # (T, H, W)

        # --- Augmentation ---
        if self.augment:
            aug = _TrainAugmentation(self.crop_size[0], self.crop_size[1], self.rng)
            frames, depth_maps, focus_maps = aug(frames, depth_maps, focus_maps)
        else:
            # Centre crop for val/test
            H, W = frames.shape[1], frames.shape[2]
            ch, cw = self.crop_size
            y0 = max(0, (H - ch) // 2)
            x0 = max(0, (W - cw) // 2)
            frames = frames[:, y0 : y0 + ch, x0 : x0 + cw, :]
            depth_maps = depth_maps[:, y0 : y0 + ch, x0 : x0 + cw]
            focus_maps = focus_maps[:, y0 : y0 + ch, x0 : x0 + cw]

        # --- ROI from focus map ---
        roi = _focus_map_to_roi(focus_maps)

        # --- Convert to tensors (C, T, H, W) ---
        frames_t = (
            torch.from_numpy(frames.copy())
            .permute(3, 0, 1, 2)
            .float()
            .div(255.0)
        )  # (3, T, H, W)
        focus_t = torch.from_numpy(focus_maps.copy()).unsqueeze(0).float()   # (1, T, H, W)
        depth_t = torch.from_numpy(depth_maps.copy()).unsqueeze(0).float()   # (1, T, H, W)
        roi_t = torch.from_numpy(roi)

        return {
            "frames": frames_t,
            "focus_maps": focus_t,
            "depth_maps": depth_t,
            "roi": roi_t,
            "metadata": {
                "scene": scene_dir.name,
                "start_frame": s,
                "focal_length": f_len,
                "aperture": aperture,
                "s_focus": s_focus,
            },
        }


def focus_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that stacks tensors and collects metadata as a list."""
    return {
        "frames": torch.stack([b["frames"] for b in batch]),        # (B,C,T,H,W)
        "focus_maps": torch.stack([b["focus_maps"] for b in batch]),# (B,1,T,H,W)
        "depth_maps": torch.stack([b["depth_maps"] for b in batch]),# (B,1,T,H,W)
        "roi": torch.stack([b["roi"] for b in batch]),              # (B,4)
        "metadata": [b["metadata"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str,
    batch_size: int = 4,
    window_length: int = 8,
    stride: int = 4,
    crop_size: Tuple[int, int] = (256, 256),
    max_coc: float = 0.05,
    focal_length_range: Tuple[float, float] = (24.0, 85.0),
    aperture_range: Tuple[float, float] = (1.4, 8.0),
    beta_alpha: float = 2.0,
    beta_beta: float = 2.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders from a data root directory."""
    root = Path(data_root)
    scene_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    splits = _split_scenes(scene_dirs, train_ratio, val_ratio, split_seed)

    loaders = {}
    for split_name, dirs in splits.items():
        ds = FocusDataset(
            scene_dirs=dirs,
            window_length=window_length,
            stride=stride,
            crop_size=crop_size,
            augment=(split_name == "train"),
            max_coc=max_coc,
            focal_length_range=tuple(focal_length_range),
            aperture_range=tuple(aperture_range),
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            seed=split_seed,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            collate_fn=focus_collate_fn,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )
    return loaders
