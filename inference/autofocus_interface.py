"""
Autofocus Interface
=====================

Takes a depth map + subject bounding box → focus distance (mm).
Designed to interface with camera lens control systems.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class AutofocusInterface:
    """Convert depth predictions to focus motor commands.

    Args:
        smoothing_alpha: Exponential smoothing factor for temporal stability.
        min_focus_distance_mm: Minimum allowed focus distance (lens near limit).
        max_focus_distance_mm: Maximum allowed focus distance (lens far limit / ∞).
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        min_focus_distance_mm: float = 300.0,
        max_focus_distance_mm: float = 100_000.0,
    ):
        self.alpha = smoothing_alpha
        self.min_focus = min_focus_distance_mm
        self.max_focus = max_focus_distance_mm
        self._smoothed_distance: Optional[float] = None

    def reset(self) -> None:
        """Reset smoothed state (call on shot cuts)."""
        self._smoothed_distance = None

    def get_focus_distance(
        self,
        depth_map: torch.Tensor,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        use_center: bool = False,
    ) -> float:
        """Compute focus distance from depth map.

        Strategy: median depth within the subject ROI.

        Args:
            depth_map: (H, W) or (1, H, W) metric depth in metres.
            bbox: (x1, y1, x2, y2) subject bounding box. If None, uses center crop.
            use_center: If True and no bbox, use center 20% of the frame.

        Returns:
            Focus distance in millimetres.
        """
        if depth_map.dim() == 3:
            depth_map = depth_map.squeeze(0)

        H, W = depth_map.shape

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = depth_map[y1:y2, x1:x2]
        elif use_center:
            cy, cx = H // 2, W // 2
            rh, rw = max(H // 10, 1), max(W // 10, 1)
            roi = depth_map[cy - rh : cy + rh, cx - rw : cx + rw]
        else:
            roi = depth_map

        # Median depth → focus distance in mm
        focus_m = roi[roi > 0].median().item() if (roi > 0).any() else 1.0
        focus_mm = focus_m * 1000.0

        # Clamp to lens limits
        focus_mm = max(self.min_focus, min(self.max_focus, focus_mm))

        # Exponential smoothing
        if self._smoothed_distance is None:
            self._smoothed_distance = focus_mm
        else:
            self._smoothed_distance = (
                self.alpha * focus_mm + (1 - self.alpha) * self._smoothed_distance
            )

        return self._smoothed_distance
