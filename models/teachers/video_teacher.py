"""
Video Depth Anything — Video Teacher Wrapper
==============================================

Wraps Video Depth Anything for use as a frozen video-level teacher.
Unlike image teachers, this processes temporal sequences natively,
providing temporally consistent depth predictions.

Repository: https://github.com/DepthAnything/Video-Depth-Anything
Checkpoint: checkpoints/metric_video_depth_anything_vitl.pth (already present)

Usage:
    teacher = VideoDepthAnythingTeacher(
        checkpoint_path="./checkpoints/metric_video_depth_anything_vitl.pth"
    )
    with torch.no_grad():
        depth = teacher.predict(clean_frames)  # (B, 1, T, H, W)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

from .teacher_base import TeacherBase

# Path to the Video-Depth-Anything repo root (script-only, no setup.py).
_VDA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "teachers", "Video-Depth-Anything")
)

# ImageNet normalisation used by DINOv2 backbone in Video-Depth-Anything.
_VDA_MEAN = (0.485, 0.456, 0.406)
_VDA_STD  = (0.229, 0.224, 0.225)


class VideoDepthAnythingTeacher(TeacherBase):
    """Video Depth Anything video teacher.

    Processes temporal sequences natively (up to temporal_window frames).
    Provides temporally consistent metric depth predictions.

    Args:
        checkpoint_path: Path to model weights.
        device: Target device.
        input_size: Expected input resolution (H, W).
        temporal_window: Maximum frames per forward pass.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "./checkpoints/metric_video_depth_anything_vitl.pth",
        device: str = "cuda",
        input_size: tuple[int, int] = (518, 518),
        temporal_window: int = 32,
    ):
        super().__init__(checkpoint_path, device, input_size)
        self.temporal_window = temporal_window
        self.model = None

    @property
    def is_video_teacher(self) -> bool:
        return True

    def _load_model(self) -> None:
        """Load Video Depth Anything (metric vitl) from teachers/Video-Depth-Anything."""
        if _VDA_ROOT not in sys.path:
            sys.path.insert(0, _VDA_ROOT)

        # VDA does `from utils.util import ...` at module level.  FocusMamba also
        # has a `utils/` package at the project root.  If it was imported first,
        # Python's module cache returns it instead of VDA's utils/ directory.
        # Temporarily pop both from sys.modules so VDA's copy is discovered first,
        # then restore the originals.  Also evict any previously cached (error-state)
        # VDA module entries so Python re-executes them.
        _stash_keys = ["utils", "utils.util"]
        _evict_keys = [k for k in sys.modules if k.startswith("video_depth_anything")]
        _stash = {k: sys.modules.pop(k) for k in _stash_keys if k in sys.modules}
        for k in _evict_keys:
            sys.modules.pop(k, None)
        try:
            from video_depth_anything.video_depth import VideoDepthAnything
        finally:
            sys.modules.update(_stash)

        self.model = VideoDepthAnything(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False,
            num_frames=self.temporal_window,
            pe="ape",
            metric=True,
        )
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.target_device).eval()

    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Fallback: predict depth for a single frame (image mode).

        For video-level predictions, use predict() directly.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1].

        Returns:
            depth: (1, 1, H, W) float32 metric depth in metres.
        """
        # Run as a 1-frame clip.
        frame = frame.to(self.target_device)
        mean = torch.tensor(_VDA_MEAN, device=self.target_device).view(1, 3, 1, 1)
        std  = torch.tensor(_VDA_STD,  device=self.target_device).view(1, 3, 1, 1)
        frame_norm = (frame - mean) / std  # (1, C, H, W)
        # forward expects (B, T, C, H, W)
        clip_input = frame_norm.unsqueeze(1)  # (1, 1, C, H, W)
        with torch.no_grad():
            depth = self.model(clip_input)  # (1, 1, H, W)  B=1, T=1
        return depth  # (1, 1, H, W) — matches _predict_single_frame contract

    def predict(self, clean_frames: torch.Tensor) -> torch.Tensor:
        """Predict depth for a batch of video clips.

        Processes clips natively as temporal sequences for better consistency.

        Args:
            clean_frames: (B, C, T, H, W) float32 in [0, 1].

        Returns:
            depth: (B, 1, T, H, W) float32 metric depth in metres.
        """
        if not self._loaded:
            self._load_model()
            self._loaded = True
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

        B, C, T, H, W = clean_frames.shape
        depths = []

        for b in range(B):
            clip = clean_frames[b]  # (C, T, H, W)

            # Resize to teacher input size
            if (H, W) != self.input_size:
                clip = torch.nn.functional.interpolate(
                    clip.permute(1, 0, 2, 3),  # (T, C, H, W)
                    size=self.input_size,
                    mode="bilinear",
                    align_corners=False,
                ).permute(1, 0, 2, 3)  # (C, T, H_t, W_t)

            # Process in temporal_window chunks
            chunk_depths = []
            for t_start in range(0, T, self.temporal_window):
                t_end = min(t_start + self.temporal_window, T)
                chunk = clip[:, t_start:t_end]  # (C, T_chunk, H_t, W_t)

                # chunk: (C, T_chunk, H_t, W_t)
                # forward expects (B, T, C, H, W)
                chunk_input = chunk.permute(1, 0, 2, 3).unsqueeze(0)  # (1, T_chunk, C, H_t, W_t)

                mean = torch.tensor(_VDA_MEAN, device=chunk_input.device).view(1, 1, 3, 1, 1)
                std  = torch.tensor(_VDA_STD,  device=chunk_input.device).view(1, 1, 3, 1, 1)
                chunk_input = (chunk_input - mean) / std

                with torch.no_grad():
                    chunk_depth = self.model(chunk_input)  # (1, T_chunk, H_t, W_t)
                chunk_depths.append(chunk_depth.unsqueeze(1))  # (1, 1, T_chunk, H_t, W_t)

            clip_depth = torch.cat(chunk_depths, dim=2)  # (1, 1, T, H_t, W_t)

            # Resize back
            if clip_depth.shape[3:] != (H, W):
                clip_depth = torch.nn.functional.interpolate(
                    clip_depth.squeeze(0),  # (1, T, H_t, W_t)
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).unsqueeze(0)

            depths.append(clip_depth)

        return torch.cat(depths, dim=0)  # (B, 1, T, H, W)
