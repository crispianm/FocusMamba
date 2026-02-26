"""
Temporal Comparison Video Generator
=====================================

Generate side-by-side depth video for supplementary material.
Compares framewise baseline vs. temporal model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import numpy as np


def generate_temporal_comparison_video(
    frames_rgb: torch.Tensor,
    depth_framewise: torch.Tensor,
    depth_temporal: torch.Tensor,
    output_path: Path,
    fps: int = 30,
    depth_gt: Optional[torch.Tensor] = None,
) -> None:
    """Generate a side-by-side comparison video.

    Layout: [RGB | Framewise Depth | Temporal Depth | (optional) GT Depth]

    Args:
        frames_rgb:       (T, 3, H, W) input RGB frames.
        depth_framewise:  (T, 1, H, W) framewise baseline depth.
        depth_temporal:   (T, 1, H, W) temporal model depth.
        output_path:      Path to save the .mp4 file.
        fps:              Frame rate.
        depth_gt:         (T, 1, H, W) optional ground-truth depth.
    """
    raise NotImplementedError(
        "Implement: use imageio or cv2.VideoWriter to compose "
        "side-by-side colourised depth comparison video."
    )
