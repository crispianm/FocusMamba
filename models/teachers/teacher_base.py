"""
Teacher Model Base Interface
==============================

All teacher wrappers inherit from this base class.
Teachers are run on *clean* (undegraded) frames to produce pseudo-GT depth.
The student learns to replicate these outputs from degraded inputs.

Expected interface:
    teacher = SomeTeacher(checkpoint_path, device)
    with torch.no_grad():
        depth = teacher.predict(clean_frames)  # (B, 1, T, H, W) metric depth
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class TeacherBase(ABC, nn.Module):
    """Abstract base class for frozen teacher depth models.

    All teachers are frozen (no gradients) and run on clean frames.
    They must output metric depth in metres (not relative/affine-invariant).

    Args:
        checkpoint_path: Path to teacher model weights.
        device: Target device ('cuda', 'cpu').
        input_size: (H, W) the teacher model expects.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        input_size: tuple[int, int] = (518, 518),
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.target_device = device
        self.input_size = input_size
        self._loaded = False

    @abstractmethod
    def _load_model(self) -> None:
        """Load the teacher model and weights. Called lazily on first predict()."""
        ...

    @abstractmethod
    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict depth for a single frame.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1], resized to self.input_size.

        Returns:
            depth: (1, 1, H, W) float32 metric depth in metres.
        """
        ...

    def predict(self, clean_frames: torch.Tensor) -> torch.Tensor:
        """Predict depth for a batch of video frames.

        Args:
            clean_frames: (B, C, T, H, W) float32 in [0, 1].

        Returns:
            depth: (B, 1, T, H, W) float32 metric depth in metres.
        """
        if not self._loaded:
            self._load_model()
            self._loaded = True
            # Freeze all parameters
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

        B, C, T, H, W = clean_frames.shape
        depths = []

        for b in range(B):
            frame_depths = []
            for t in range(T):
                frame = clean_frames[b : b + 1, :, t]  # (1, 3, H, W)
                # Resize to teacher input size
                if (H, W) != self.input_size:
                    frame = torch.nn.functional.interpolate(
                        frame, size=self.input_size, mode="bilinear", align_corners=False,
                    )
                d = self._predict_single_frame(frame)  # (1, 1, H_t, W_t)
                # Resize back to original resolution
                if d.shape[2:] != (H, W):
                    d = torch.nn.functional.interpolate(
                        d, size=(H, W), mode="bilinear", align_corners=False,
                    )
                frame_depths.append(d)
            depths.append(torch.stack(frame_depths, dim=2))  # (1, 1, T, H, W)

        result = torch.cat(depths, dim=0)  # (B, 1, T, H, W)
        return result.to(self.target_device)

    @property
    def is_video_teacher(self) -> bool:
        """Whether this teacher processes temporal sequences natively."""
        return False
