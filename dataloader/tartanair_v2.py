"""
TartanAir v2 Clip Dataset
==========================

Loads short RGB + GT depth clips from TartanAir v2 for supervised training.
Ground-truth depth is stored as RGBA PNGs encoding float32 metres.

Dataset structure::

    root/
    ├── AbandonedFactory/
    │   └── Data_easy/
    │       ├── P000/
    │       │   ├── image_lcam_front/
    │       │   │   ├── 000000_lcam_front.png
    │       │   │   └── ...
    │       │   ├── depth_lcam_front/
    │       │   │   ├── 000000_lcam_front_depth.png
    │       │   │   └── ...
    │       │   └── pose_lcam_front.txt
    │       ├── P001/
    │       └── ...
    ├── AmusementPark/
    └── ...

Each depth PNG is RGBA uint8 encoding a float32 depth value in metres.

Returns dict with:
    frames          (C, T, H, W)  float32 [0, 1]   – RGB clip
    depth           (1, T, H, W)  float32 metres    – ground-truth depth
    video_id        str                              – e.g. "AbandonedFactory/P000"
    start_frame     int                              – index of first frame

Usage::

    dataset = TartanAirV2Dataset(
        root="/projects/b5dh/data/tartanair-v2",
        num_frames=8,
        image_size=(256, 256),
        max_trajectories=None,
        clip_stride=8,
        frame_stride=1,
        split="train",
        val_fraction=0.1,
        seed=42,
    )
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def _decode_tartanair_depth(path: str) -> np.ndarray:
    """Decode a TartanAir v2 depth PNG (RGBA uint8 → float32 metres).

    The depth is stored as 4 bytes (RGBA channels) per pixel, which
    when reinterpreted as float32 give the depth in metres.
    """
    img = Image.open(path)
    arr = np.array(img, dtype=np.uint8)  # (H, W, 4)
    depth = arr.view(np.float32).reshape(arr.shape[0], arr.shape[1])  # (H, W)
    return depth


class TartanAirV2Dataset(Dataset):
    """RGB + GT depth clip dataset from TartanAir v2.

    Args:
        root: Path to the TartanAir v2 root directory containing environment folders.
        num_frames: Frames per clip (T).
        image_size: Spatial resolution (H, W) to resize frames to.
        max_trajectories: Cap the number of trajectories used. ``None`` uses all.
        clip_stride: Stride between the start frames of successive clips *within*
            the same trajectory.
        frame_stride: Step between consecutive frames inside a clip.
        split: ``"train"`` or ``"val"``.
        val_fraction: Fraction of trajectories reserved for validation.
        seed: Random seed for train/val split.
        difficulty: Which difficulty folder to use (default ``"Data_easy"``).
        camera: Camera name (default ``"lcam_front"``).
        max_depth: Clip depths above this value (metres). Default 80.0.
        envs: Optional list of environment names to include. None uses all.
    """

    def __init__(
        self,
        root: str,
        num_frames: int = 8,
        image_size: Tuple[int, int] = (256, 256),
        max_trajectories: Optional[int] = None,
        clip_stride: int = 8,
        frame_stride: int = 1,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        difficulty: str = "Data_easy",
        camera: str = "lcam_front",
        max_depth: float = 80.0,
        envs: Optional[List[str]] = None,
        teacher_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.clip_stride = clip_stride
        self.frame_stride = frame_stride
        self.max_depth = max_depth
        self.camera = camera
        self.teacher_cache_dir = Path(teacher_cache_dir) if teacher_cache_dir else None

        # --- Discover all trajectories with both image and depth -----------
        all_trajectories: List[Tuple[Path, Path]] = []  # (image_dir, depth_dir)

        env_dirs = sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        if envs is not None:
            env_dirs = [d for d in env_dirs if d.name in envs]

        for env_dir in env_dirs:
            diff_dir = env_dir / difficulty
            if not diff_dir.is_dir():
                continue
            for traj_dir in sorted(diff_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                img_dir = traj_dir / f"image_{camera}"
                depth_dir = traj_dir / f"depth_{camera}"
                if img_dir.is_dir() and depth_dir.is_dir():
                    all_trajectories.append((img_dir, depth_dir))

        if len(all_trajectories) == 0:
            raise FileNotFoundError(
                f"No trajectories found with both image and depth data in {root}. "
                f"Expected structure: <root>/<EnvName>/{difficulty}/P00X/image_{camera}/"
            )

        # --- Deterministic train/val split ----------------------------------
        rng = random.Random(seed)
        shuffled = all_trajectories[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))

        if split == "val":
            trajectories = shuffled[:n_val]
        else:
            trajectories = shuffled[n_val:]

        if max_trajectories is not None:
            trajectories = trajectories[:max_trajectories]

        # --- Build clip index -----------------------------------------------
        self.clips: List[Tuple[List[Path], List[Path], str, int]] = []
        clip_len_in_frames = (num_frames - 1) * frame_stride + 1

        for img_dir, depth_dir in trajectories:
            # Collect and sort image files
            img_files = sorted(img_dir.glob("*.png"))
            depth_files = sorted(depth_dir.glob("*.png"))

            # Use the minimum of available images and depths
            n = min(len(img_files), len(depth_files))
            if n < clip_len_in_frames:
                continue

            img_files = img_files[:n]
            depth_files = depth_files[:n]

            # Create a video_id from the trajectory path
            # e.g. "AbandonedFactory/P000"
            env_name = img_dir.parent.parent.parent.name
            traj_name = img_dir.parent.name
            video_id = f"{env_name}/{traj_name}"

            for start in range(0, n - clip_len_in_frames + 1, clip_stride):
                self.clips.append((img_files, depth_files, video_id, start))

        # --- Transforms -----------------------------------------------------
        self._to_tensor = transforms.ToTensor()  # uint8 → float [0,1]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        img_files, depth_files, video_id, start = self.clips[idx]

        indices = [start + t * self.frame_stride for t in range(self.num_frames)]

        imgs = []
        depths = []
        for i in indices:
            # Load RGB
            img = Image.open(img_files[i]).convert("RGB")
            t = self._to_tensor(img)  # (3, H, W) float [0, 1]
            imgs.append(t)

            # Load depth
            depth = _decode_tartanair_depth(str(depth_files[i]))  # (H, W) float32
            depth = np.clip(depth, 0.0, self.max_depth)
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()  # (1, H, W)
            depths.append(depth_t)

        frames = torch.stack(imgs, dim=1)   # (3, T, H, W)
        depth = torch.stack(depths, dim=1)  # (1, T, H, W)

        # Resize if needed
        H, W = self.image_size
        if frames.shape[-2:] != (H, W):
            # Resize RGB
            frames = F.interpolate(
                frames.permute(1, 0, 2, 3),  # (T, 3, H_orig, W_orig)
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)  # (3, T, H, W)

            # Resize depth (nearest to avoid interpolation artefacts)
            depth = F.interpolate(
                depth.permute(1, 0, 2, 3),  # (T, 1, H_orig, W_orig)
                size=(H, W),
                mode="nearest",
            ).permute(1, 0, 2, 3)  # (1, T, H, W)

        result: Dict = {
            "frames": frames,
            "depth": depth,
            "video_id": video_id,
            "start_frame": start,
        }

        # ── Load pre-cached teacher pseudo-labels (if available) ────────────
        # This eliminates the need to run expensive teacher forward passes
        # during training.  Run tools/cache_teacher_labels.py to populate.
        if self.teacher_cache_dir is not None:
            clip_cache = self.teacher_cache_dir / video_id / str(start)
            cached: Dict[str, torch.Tensor] = {}
            if clip_cache.is_dir():
                for npy_file in clip_cache.glob("*.npy"):
                    teacher_name = npy_file.stem
                    try:
                        # mmap_mode='r' memory-maps the file — the OS shares pages
                        # across all DataLoader worker processes and skips pickle
                        # deserialization, giving faster multi-worker throughput.
                        arr = np.load(str(npy_file), mmap_mode="r")  # (1, T, H_c, W_c) f16
                        td = torch.from_numpy(arr.astype(np.float32))  # copy out of mmap
                        H, W = self.image_size
                        if td.shape[-2:] != (H, W):
                            # Resize to training resolution if cache was built at a different res
                            td = F.interpolate(
                                td,              # (1, T, H_c, W_c)
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False,
                            )
                        cached[teacher_name] = td  # (1, T, H, W)
                    except Exception:
                        pass  # corrupt / partial file — skip gracefully
            if cached:
                result["cached_teacher_depths"] = cached

        return result
