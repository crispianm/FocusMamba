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

import os
import sys
from typing import Optional

import torch

from .teacher_base import TeacherBase

# Path to the ml-depth-pro package (src/ layout).
_DEPTH_PRO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "teachers", "ml-depth-pro", "src")
)


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
        """Load Apple Depth Pro model from teachers/ml-depth-pro."""
        if _DEPTH_PRO_SRC not in sys.path:
            sys.path.insert(0, _DEPTH_PRO_SRC)

        from depth_pro import create_model_and_transforms
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        cfg = DEFAULT_MONODEPTH_CONFIG_DICT
        cfg.checkpoint_uri = self.checkpoint_path

        self.model, _transform = create_model_and_transforms(
            config=cfg,
            device=torch.device(self.target_device),
            precision=torch.float16,
        )
        self.model.eval()

    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict metric depth for a single frame.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1].

        Returns:
            depth: (1, 1, H, W) float32 metric depth in metres.
        """
        # Depth Pro expects the input normalised to [-1, 1]
        # (its own transform: Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        frame = frame.to(self.target_device, dtype=torch.float16)
        frame_norm = frame * 2.0 - 1.0  # [0,1] → [-1,1]

        result = self.model.infer(frame_norm)
        depth = result["depth"]  # shape: (H, W) or (1, H, W) or (1, 1, H, W)

        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0)

        return depth.float()
