"""
Metric3D V2 — Image Teacher Wrapper
=====================================

Wraps Metric3D V2 for use as a frozen image-level teacher.
Produces metric depth predictions from individual clean frames.

Repository: https://github.com/YvanYin/Metric3D

Usage:
    teacher = Metric3DV2Teacher(checkpoint_path="path/to/weights.pth")
    with torch.no_grad():
        depth = teacher.predict(clean_frames)  # (B, 1, T, H, W)
"""

from __future__ import annotations

from typing import Optional

import torch

from .teacher_base import TeacherBase


class Metric3DV2Teacher(TeacherBase):
    """Metric3D V2 image teacher.

    Args:
        checkpoint_path: Path to model weights.
        device: Target device.
        input_size: Expected input resolution (H, W).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        input_size: tuple[int, int] = (616, 1064),
    ):
        super().__init__(checkpoint_path, device, input_size)
        self.model = None

    def _load_model(self) -> None:
        """Load Metric3D V2 model.

        TODO: Implement actual model loading.
        Expected steps:
            1. Import metric3d package
            2. model = metric3d.build_model(cfg)
            3. model.load_state_dict(torch.load(self.checkpoint_path))
            4. self.model = model.to(self.target_device).eval()
        """
        raise NotImplementedError(
            "Implement Metric3D V2 model loading. "
            "Install from: https://github.com/YvanYin/Metric3D"
        )

    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict metric depth for a single frame.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1].

        Returns:
            depth: (1, 1, H, W) float32 metric depth in metres.
        """
        # TODO: Implement inference
        raise NotImplementedError
