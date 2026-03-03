"""Trainable Video Depth Anything wrapper.

Provides a model interface compatible with the repository training loop:
input ``(B, C, T, H, W)`` -> output dict with ``depth`` in
``(B, 1, T, H, W)``.

This uses the vendored Video Depth Anything implementation directly, but
without teacher-mode freezing/no-grad behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from models.teachers.vendor.video_depth_anything import VideoDepthAnything


_VDA_VARIANTS: Dict[str, Dict] = {
    # Mirrors upstream model_configs in Video-Depth-Anything/run.py
    "small": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "base": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "large": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    # Encoder aliases
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


class VideoDepthAnythingModel(nn.Module):
    """Trainable Video Depth Anything model for GT-supervised learning.

    Args:
        variant: ``small`` | ``base`` | ``large`` (or encoder aliases).
        num_frames: Temporal length used by temporal positional embeddings.
        positional_encoding: Temporal position embedding type (``ape`` or ``rope``).
        checkpoint_path: Optional path to pretrained VDA weights.
        strict_checkpoint: If True, require exact checkpoint match.
    """

    def __init__(
        self,
        variant: str = "small",
        num_frames: int = 8,
        positional_encoding: str = "ape",
        checkpoint_path: str | None = None,
        strict_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        key = variant.lower()
        if key not in _VDA_VARIANTS:
            valid = ", ".join(sorted(_VDA_VARIANTS.keys()))
            raise ValueError(f"Unknown VDA variant '{variant}'. Available: {valid}")

        cfg = _VDA_VARIANTS[key]
        self.variant = key
        self.patch_size = 14

        self.backbone = VideoDepthAnything(
            encoder=cfg["encoder"],
            features=cfg["features"],
            out_channels=cfg["out_channels"],
            use_bn=False,
            use_clstoken=False,
            num_frames=num_frames,
            pe=positional_encoding,
            metric=True,
        )

        # ImageNet normalization expected by DINOv2 backbone.
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, strict=strict_checkpoint)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load pretrained weights, tolerating positional-shape differences by default."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        cleaned_state = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            key = key.removeprefix("module.")
            key = key.removeprefix("model.")
            cleaned_state[key] = value

        current_state = self.backbone.state_dict()
        compatible_state = {}
        skipped_shape = []
        for key, value in cleaned_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            elif key in current_state:
                skipped_shape.append(key)

        missing_keys, unexpected_keys = self.backbone.load_state_dict(compatible_state, strict=False)
        if strict and (missing_keys or unexpected_keys or skipped_shape):
            raise RuntimeError(
                "Strict checkpoint loading failed for VideoDepthAnythingModel. "
                f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
                f"shape_mismatch={len(skipped_shape)}"
            )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            frames: ``(B, C, T, H, W)`` float tensor in [0, 1].
                    H and W must be multiples of 14.
        Returns:
            Dict with ``depth`` in shape ``(B, 1, T, H, W)``.
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(frames.shape)}")

        _, channels, _, height, width = frames.shape
        if channels != 3:
            raise ValueError(f"VideoDepthAnythingModel expects 3 RGB channels, got {channels}")
        if (height % self.patch_size) != 0 or (width % self.patch_size) != 0:
            raise ValueError(
                f"VideoDepthAnythingModel requires H/W divisible by {self.patch_size}, "
                f"got H={height}, W={width}."
            )

        x = frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        mean = self._mean.to(dtype=x.dtype, device=x.device)
        std = self._std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) / std

        depth = self.backbone(x).unsqueeze(1)  # (B, 1, T, H, W)
        return {"depth": depth}

