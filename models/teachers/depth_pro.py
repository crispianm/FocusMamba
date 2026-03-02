"""
Apple Depth Pro — Image Teacher Wrapper
========================================

Wraps Apple's Depth Pro model for use as a frozen image-level teacher.
Produces metric depth predictions from individual clean frames.

Uses vendored model files from models/teachers/vendor/depth_pro/ to avoid
dependency on the upstream git repo and to call forward() directly without
the overhead of the inference() preprocessing pipeline.

Original repository: https://github.com/apple/ml-depth-pro
Checkpoint: checkpoints/depth_pro.pt

Usage:
    teacher = DepthProTeacher(checkpoint_path="./checkpoints/depth_pro.pt")
    with torch.no_grad():
        depth = teacher.predict(clean_frames)  # (B, 1, T, H, W)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .teacher_base import TeacherBase


class DepthProTeacher(TeacherBase):
    """Apple Depth Pro image teacher.

    Calls the raw DepthPro.forward() which returns canonical inverse depth +
    FOV estimate, then converts to metric depth internally — no numpy
    round-trips or redundant preprocessing.

    Note: forward() requires exactly 1536x1536 input. Frames are resized
    to that resolution, and the depth output is resized back.

    Args:
        checkpoint_path: Path to model weights (default: checkpoints/depth_pro.pt).
        device: Target device.
        input_size: Internal network resolution (must be 1536x1536).
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
        """Load Depth Pro from vendored model files."""
        from .vendor.depth_pro import DepthProConfig, create_model_and_transforms

        cfg = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=self.checkpoint_path,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )

        self.model, _ = create_model_and_transforms(
            config=cfg,
            device=torch.device(self.target_device),
            precision=torch.float16,
        )
        self.model.eval()

    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict metric depth for a single frame via raw forward().

        Calls DepthPro.forward() directly instead of the infer() wrapper,
        avoiding unnecessary resizing and preprocessing overhead.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1], already resized to
                   self.input_size by the base class.

        Returns:
            depth: (1, 1, H, W) float32 metric depth in metres.
        """
        # forward() expects [-1, 1] normalisation (mean=0.5, std=0.5)
        frame = frame.to(self.target_device, dtype=torch.float16)
        frame_norm = frame * 2.0 - 1.0  # [0,1] → [-1,1]

        # Raw forward: returns (canonical_inverse_depth, fov_deg)
        # canonical_inverse_depth: (1, 1, H_out, W_out)
        # fov_deg: scalar tensor
        canonical_inverse_depth, fov_deg = self.model(frame_norm)

        # Convert canonical inverse depth → metric depth
        # inverse_depth = canonical_inverse_depth * (W / f_px)
        # depth = 1.0 / inverse_depth
        W = frame.shape[-1]  # 1536
        if fov_deg is not None:
            f_px = 0.5 * W / torch.tan(
                0.5 * torch.deg2rad(fov_deg.to(torch.float))
            )
        else:
            # Fallback: assume 60° FOV if head is disabled
            f_px = 0.5 * W / torch.tan(torch.tensor(0.5236))  # ~30° half-angle

        inverse_depth = canonical_inverse_depth * (W / f_px)

        # Resize inverse depth to match input resolution if needed
        if inverse_depth.shape[2:] != (self.input_size[0], self.input_size[1]):
            inverse_depth = F.interpolate(
                inverse_depth,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        return depth.float()  # (1, 1, H, W)
