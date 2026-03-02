"""
YouTube-VOS Clip Dataset
=========================

Loads short RGB clips from YouTube-VOS for teacher-distillation pre-training.
There is no ground-truth depth here — teachers supply pseudo-GT at run time.

Designed to be easily swapped out for datasets that *do* have GT depth
(e.g. TartanAir); the returned dict keys are stable:

    frames          (C, T, H, W)  float32 [0, 1]   – clean RGB clip
    video_id        str                              – source video folder name
    start_frame     int                              – index of first frame

For future datasets that provide GT depth, add:
    depth           (1, T, H, W)  float32 metres    – ground-truth depth
    depth_valid     (1, T, H, W)  bool               – valid depth mask

Usage::

    dataset = YouTubeVOSDataset(
        root="data/youtube-vos/train_all_frames/JPEGImages",
        num_frames=8,
        image_size=(256, 256),
        max_videos=100,          # None → full dataset
        clip_stride=8,           # hop between clips within a video
        frame_stride=1,          # hop between frames within a clip
        split="train",           # "train" or "val"
        val_fraction=0.1,
        seed=42,
    )
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class YouTubeVOSDataset(Dataset):
    """RGB clip dataset from YouTube-VOS (no ground-truth depth).

    Args:
        root: Path to the ``JPEGImages/`` folder (contains one sub-dir per video).
        num_frames: Frames per clip (T).
        image_size: Spatial resolution (H, W) to resize frames to.
        max_videos: Cap the number of videos used. ``None`` uses all.
        clip_stride: Stride between the start frames of successive clips *within*
            the same video.  Controls clip density / dataset size.
        frame_stride: Step between consecutive frames inside a clip .  ``1``
            means consecutive frames; ``2`` means every-other frame, etc.
        split: ``"train"`` or ``"val"``.
        val_fraction: Fraction of videos reserved for validation.
        seed: Random seed for train/val split.
    """

    def __init__(
        self,
        root: str,
        num_frames: int = 8,
        image_size: Tuple[int, int] = (256, 256),
        max_videos: Optional[int] = None,
        clip_stride: int = 8,
        frame_stride: int = 1,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.clip_stride = clip_stride
        self.frame_stride = frame_stride

        # --- collect all video directories ----------------------------------
        all_videos = sorted(p for p in self.root.iterdir() if p.is_dir())

        # deterministic train/val split
        rng = random.Random(seed)
        shuffled = all_videos[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        if split == "val":
            videos = shuffled[:n_val]
        else:
            videos = shuffled[n_val:]

        if max_videos is not None:
            videos = videos[:max_videos]

        # --- build clip index -----------------------------------------------
        # Each entry: (video_dir, start_frame_idx)
        self.clips: List[Tuple[Path, int]] = []
        clip_len_in_frames = (num_frames - 1) * frame_stride + 1  # total span

        for vdir in videos:
            frames = sorted(vdir.glob("*.jpg")) + sorted(vdir.glob("*.png"))
            frames.sort()
            n = len(frames)
            if n < clip_len_in_frames:
                continue
            for start in range(0, n - clip_len_in_frames + 1, clip_stride):
                self.clips.append((vdir, start, frames))

        # --- transforms -----------------------------------------------------
        self._to_tensor = transforms.ToTensor()   # (H, W, C) uint8 → (C, H, W) float [0,1]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        vdir, start, frames = self.clips[idx]

        # load frames for this clip
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]
        imgs = []
        for i in indices:
            img = Image.open(frames[i]).convert("RGB")
            t = self._to_tensor(img)  # (3, H, W) float [0, 1]
            imgs.append(t)

        frames_chw = torch.stack(imgs, dim=1)  # (3, T, H, W)

        # resize if needed
        H, W = self.image_size
        if frames_chw.shape[-2:] != (H, W):
            frames_chw = F.interpolate(
                frames_chw.permute(1, 0, 2, 3),    # (T, 3, H, W)
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)  # (3, T, H, W)

        return {
            "frames": frames_chw,
            "video_id": vdir.name,
            "start_frame": start,
        }
