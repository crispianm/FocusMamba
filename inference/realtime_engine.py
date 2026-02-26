"""
Realtime Inference Engine
===========================

Stateful inference: maintains Mamba hidden state across frames for
causal, online depth estimation. Key properties:

    - Mamba state persists across frames within a shot
    - Shot-cut detector triggers state reset to prevent cross-scene contamination
    - Target: ≤33ms per frame (30fps) on RTX 4080; ≤16ms for 60fps
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class RealtimeDepthEngine:
    """Stateful real-time depth inference engine.

    Maintains Mamba hidden state across sequential frames. Call
    `process_frame()` for each new frame. Call `reset()` on shot cuts.

    Args:
        model: Trained FocusMamba model.
        device: Target device.
        shot_cut_threshold: Colour histogram change threshold to auto-detect cuts.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        shot_cut_threshold: float = 0.5,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.shot_cut_threshold = shot_cut_threshold

        # Mamba hidden state cache
        self._hidden_state: Optional[Any] = None
        self._prev_frame: Optional[torch.Tensor] = None
        self._frame_count: int = 0

    def reset(self) -> None:
        """Reset all internal state. Call on shot cuts."""
        self._hidden_state = None
        self._prev_frame = None
        self._frame_count = 0

    def detect_shot_cut(self, frame: torch.Tensor) -> bool:
        """Detect shot cut via colour histogram comparison.

        Args:
            frame: (3, H, W) normalised frame tensor.

        Returns:
            True if a shot cut is detected.
        """
        if self._prev_frame is None:
            return False

        # Simple proxy: mean absolute difference across channels
        diff = (frame - self._prev_frame).abs().mean().item()
        return diff > self.shot_cut_threshold

    @torch.no_grad()
    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Process a single frame and return metric depth.

        Args:
            frame: (3, H, W) normalised RGB frame in [0, 1].

        Returns:
            (1, H, W) metric depth in metres.
        """
        # Check for shot cut
        if self.detect_shot_cut(frame):
            self.reset()

        # Reshape to (1, C, 1, H, W) for model
        x = frame.unsqueeze(0).unsqueeze(2).to(self.device)

        # TODO: Integrate Mamba hidden state passing for truly stateful inference.
        # Current implementation processes each frame independently.
        # For full efficiency, the model's Mamba layers should accept and return
        # their hidden states.
        outputs = self.model(x)
        depth = outputs["depth"][:, :, 0]  # (1, 1, H, W) → (1, 1, H, W)

        self._prev_frame = frame
        self._frame_count += 1

        return depth.squeeze(0)  # (1, H, W)

    @torch.no_grad()
    def process_clip(self, frames: torch.Tensor) -> torch.Tensor:
        """Process a clip of T frames in one forward pass.

        Args:
            frames: (T, 3, H, W) clip of frames.

        Returns:
            (T, 1, H, W) metric depth for each frame.
        """
        # Reshape to (1, C, T, H, W)
        x = frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # (1, 3, T, H, W)
        outputs = self.model(x)
        depth = outputs["depth"][0]  # (1, T, H, W)
        return depth.permute(1, 0, 2, 3)  # (T, 1, H, W)
