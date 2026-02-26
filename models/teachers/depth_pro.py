"""
Apple Depth Pro — Image Teacher Wrapper
========================================

Wraps Apple's Depth Pro model for use as a frozen image-level teacher.
Produces metric depth predictions from individual clean frames.

Repository: https://github.com/apple/ml-depth-pro
Checkpoint: checkpoints/depth_pro.pt (already present)

Usage:
    teacher = DepthProTeacher(checkpoint_path="./checkpoints/depth_pro.pt")
    with torch.no_grad():
        depth = teacher.predict(clean_frames)  # (B, 1, T, H, W)
"""

from __future__ import annotations

from typing import Optional

import torch

from .teacher_base import TeacherBase


class DepthProTeacher(TeacherBase):
    """Apple Depth Pro image teacher.

    Memory-heavy model — processes frames one-by-one.

    Args:
        checkpoint_path: Path to model weights (default: checkpoints/depth_pro.pt).
        device: Target device.
        input_size: Expected input resolution (H, W).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "./checkpoints/depth_pro.pt",
        device: str = "cuda",
        input_size: tuple[int, int] = (1536, 1536),
    ):
        super().__init__(checkpoint_path, device, input_size)
        self.model = None

    def _load_model(self) -> None:
        """Load Apple Depth Pro model.

        TODO: Implement actual model loading.
        Expected steps:
            1. Import depth_pro package
            2. cfg = depth_pro.DepthProConfig(...)
            3. model = depth_pro.create_model(cfg)
            4. model.load_state_dict(torch.load(self.checkpoint_path))
            5. self.model = model.to(self.target_device).eval()
        """
        raise NotImplementedError(
            "Implement Depth Pro model loading. "
            "Install from: https://github.com/apple/ml-depth-pro "
            f"Checkpoint expected at: {self.checkpoint_path}"
        )

    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict metric depth for a single frame.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1].

        Returns:
            depth: (1, 1, H, W) float32 metric depth in metres.
        """
        # TODO: Implement inference
        # prediction = self.model.infer(frame)
        # depth = prediction["depth"]  # shape depends on Depth Pro API
        # return depth.unsqueeze(0).unsqueeze(0) if needed
        raise NotImplementedError
