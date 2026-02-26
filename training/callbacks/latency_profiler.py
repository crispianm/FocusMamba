"""
Latency Profiler Callback
===========================

Measures inference latency (FPS) at each checkpoint and flags regressions.
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


class LatencyProfiler:
    """Measures model inference latency and flags regressions.

    Args:
        input_shape: (B, C, T, H, W) shape for profiling.
        target_fps: Target FPS threshold; flags if below.
        warmup_iters: Number of warmup iterations before timing.
        measure_iters: Number of timed iterations.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 8, 256, 256),
        target_fps: float = 30.0,
        warmup_iters: int = 5,
        measure_iters: int = 20,
    ):
        self.input_shape = input_shape
        self.target_fps = target_fps
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters

    @torch.no_grad()
    def measure(self, model: nn.Module, device: torch.device) -> dict:
        """Measure inference latency.

        Returns:
            Dict with 'fps', 'ms_per_frame', 'meets_target'.
        """
        model.eval()
        dummy = torch.randn(self.input_shape, device=device)

        # Warmup
        for _ in range(self.warmup_iters):
            _ = model(dummy)
        torch.cuda.synchronize() if device.type == "cuda" else None

        # Timed runs
        start = time.perf_counter()
        for _ in range(self.measure_iters):
            _ = model(dummy)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        T = self.input_shape[2]
        total_frames = self.measure_iters * T
        fps = total_frames / elapsed
        ms_per_frame = elapsed / total_frames * 1000

        return {
            "fps": fps,
            "ms_per_frame": ms_per_frame,
            "meets_target": fps >= self.target_fps,
        }
