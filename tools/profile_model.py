"""
Profile Model
===============

FLOPs, parameter count, memory, latency table.

Usage:
    python tools/profile_model.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
    }


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 8, 256, 256),
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 50,
) -> dict:
    """Measure inference latency."""
    model = model.to(device).eval()
    dummy = torch.randn(input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    T = input_shape[2]
    total_frames = iterations * T
    return {
        "fps": round(total_frames / elapsed, 1),
        "ms_per_frame": round(elapsed / total_frames * 1000, 2),
        "ms_per_clip": round(elapsed / iterations * 1000, 2),
    }


def measure_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 8, 256, 256),
    device: str = "cuda",
) -> dict:
    """Measure peak GPU memory usage."""
    if device != "cuda" or not torch.cuda.is_available():
        return {"peak_memory_MB": "N/A (CPU)"}

    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    dummy = torch.randn(input_shape, device=device)

    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1e6
    return {"peak_memory_MB": round(peak, 1)}


def main():
    parser = argparse.ArgumentParser(description="Profile FocusMamba model")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input-shape", type=int, nargs=5, default=[1, 3, 8, 256, 256])
    args = parser.parse_args()

    raise NotImplementedError(
        "Wire up: load config, build model, call count_parameters, "
        "measure_latency, measure_memory, print table."
    )


if __name__ == "__main__":
    main()
