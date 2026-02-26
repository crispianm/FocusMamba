"""
Latency Evaluation Protocol
=============================

FPS measurement on target hardware configurations.
Tests multiple resolutions and batch sizes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import torch


def run_latency_eval(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    resolutions: List[Tuple[int, int]] = None,
    warmup: int = 10,
    iterations: int = 50,
) -> Dict[str, Any]:
    """Measure inference latency at various resolutions.

    Args:
        model: Depth model in eval mode.
        config: Experiment config.
        device: Target device.
        resolutions: List of (H, W) to test. Defaults to standard set.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.

    Returns:
        Dict mapping resolution string to {fps, ms_per_frame}.
    """
    if resolutions is None:
        resolutions = [(256, 256), (384, 384), (512, 512), (720, 1280)]

    T = config.get("data", {}).get("clip_length", 8)
    results = {}

    model.eval()
    for H, W in resolutions:
        dummy = torch.randn(1, 3, T, H, W, device=device)
        key = f"{H}x{W}"

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(iterations):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        total_frames = iterations * T
        fps = total_frames / elapsed
        results[key] = {
            "fps": round(fps, 1),
            "ms_per_frame": round(elapsed / total_frames * 1000, 2),
        }

    return results
