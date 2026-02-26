"""
Depth Anything V3 — Image Teacher Wrapper
==========================================

Wraps the Depth Anything V3 model for use as a frozen image-level teacher.
Produces metric depth predictions from individual clean frames.

Repository: https://github.com/ByteDance-Seed/Depth-Anything-3

Usage:
    teacher = DepthAnythingV3Teacher(checkpoint_path="path/to/weights.pth")
    with torch.no_grad():
        depth = teacher.predict(clean_frames)  # (B, 1, T, H, W)
"""

from __future__ import annotations

from typing import Optional

import torch

from .teacher_base import TeacherBase


class DepthAnythingV3Teacher(TeacherBase):
    """Depth Anything V3 image teacher.

    Args:
        checkpoint_path: Path to model weights.
        device: Target device.
        input_size: Expected input resolution (H, W).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        input_size: tuple[int, int] = (518, 518),
    ):
        super().__init__(checkpoint_path, device, input_size)
        self.model = None

    def _load_model(self) -> None:
        """Load Depth Anything V3 model.

        TODO: Implement actual model loading from the official repository.
        Expected steps:
            1. Import the model class from depth_anything_v3 package
            2. Instantiate with appropriate config
            3. Load checkpoint weights
            4. Move to device and set to eval mode
        """
        raise NotImplementedError(
            "Implement Depth Anything V3 model loading. "
            "Install from: https://github.com/ByteDance-Seed/Depth-Anything-3"
        )

    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict metric depth for a single frame.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1].

        Returns:
            depth: (1, 1, H, W) float32 metric depth in metres.
        """
        # TODO: Implement inference
        # depth = self.model(frame)
        # return depth.unsqueeze(1) if depth.dim() == 3 else depth
        raise NotImplementedError
