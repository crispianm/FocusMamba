"""
Depth Anything V3 — Image Teacher Wrapper (Vendored)
=====================================================

Wraps the vendored da3metric-large model for use as a frozen image-level
teacher.  Calls forward() directly — no numpy round-trip, no OmegaConf,
no HuggingFace Hub, no sys.path manipulation.

Usage:
    teacher = DepthAnythingV3Teacher(checkpoint_path="checkpoints/da3_metric.safetensors")
    with torch.no_grad():
        depth = teacher.predict(clean_frames)  # (B, 1, T, H, W)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .teacher_base import TeacherBase

# ImageNet normalisation constants.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthAnythingV3Teacher(TeacherBase):
    """Depth Anything 3 (da3metric-large) image teacher.

    Args:
        checkpoint_path: Path to ``da3_metric.safetensors`` or ``.pth``.
        model_name: Unused — architecture is hardcoded as da3metric-large.
        device: Target device.
        input_size: Expected input resolution ``(H, W)``.
            Must be divisible by patch_size=14.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "./checkpoints/da3_metric.safetensors",
        model_name: str = "da3metric-large",
        device: str = "cuda",
        input_size: tuple[int, int] = (518, 518),
    ):
        super().__init__(checkpoint_path, device, input_size)
        self.model_name = model_name
        self.model = None

        # Pre-compute normalisation tensors (registered as buffers later).
        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False,
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False,
        )

    def _load_model(self) -> None:
        """Load da3metric-large from vendored code + safetensors checkpoint."""
        from .vendor.depth_anything_v3 import create_da3_metric_large
        from .vendor.depth_anything_v3.da3 import load_da3_metric_large

        if self.checkpoint_path is not None:
            self.model = load_da3_metric_large(
                self.checkpoint_path, device=self.target_device,
            )
        else:
            self.model = create_da3_metric_large()
            self.model = self.model.to(self.target_device).eval()

    @torch.no_grad()
    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict metric depth for a single frame.

        Args:
            frame: ``(1, 3, H, W)`` float32 in ``[0, 1]``.

        Returns:
            depth: ``(1, 1, H, W)`` float32 metric depth in metres.
        """
        # ImageNet normalise: [0, 1] → standardised.
        x = (frame - self._mean.to(frame.device)) / self._std.to(frame.device)

        # DA3 expects (B, N, 3, H, W) where N = number of views (1 for mono).
        x = x.unsqueeze(1)  # (1, 1, 3, H, W)

        output = self.model(x)  # dict: depth (B, N, H, W), sky (B, N, 1, H, W)
        depth = output["depth"]  # (1, 1, H, W) — (B=1, N=1, H, W)

        # Zero out sky pixels so they are excluded by the GT validity mask
        # during training.  sky is relu-activated: any positive value = sky.
        sky = output.get("sky")  # (1, 1, 1, H, W) or None
        if sky is not None:
            sky_mask = sky.squeeze(2) > 0.1  # (1, 1, H, W) bool
            depth = depth.masked_fill(sky_mask, 0.0)

        # depth shape is (1, 1, H, W) which matches expected (B=1, C=1, H, W).
        return depth
