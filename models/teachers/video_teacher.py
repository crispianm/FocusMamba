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

from typing import Optional

import torch

from .teacher_base import TeacherBase


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
        """Load Video Depth Anything model.

        TODO: Implement actual model loading.
        The Video-Depth-Anything/ directory is already present in the project.
        Expected steps:
            1. Import from Video-Depth-Anything package
            2. Instantiate model
            3. Load checkpoint weights
            4. self.model = model.to(self.target_device).eval()
        """
        raise NotImplementedError(
            "Implement Video Depth Anything model loading. "
            "Repo at: Video-Depth-Anything/ in project root. "
            f"Checkpoint expected at: {self.checkpoint_path}"
        )

    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Fallback: predict depth for a single frame (image mode).

        For video-level predictions, use predict() directly.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1].

        Returns:
            depth: (1, 1, H, W) float32 metric depth in metres.
        """
        raise NotImplementedError

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

                # TODO: Run through video model
                # chunk_depth = self.model(chunk.unsqueeze(0))
                # chunk_depths.append(chunk_depth)
                raise NotImplementedError

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
