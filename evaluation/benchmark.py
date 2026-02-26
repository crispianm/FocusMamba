"""
Benchmark — Main Evaluation Entry Point
=========================================

Runs all evaluation protocols and outputs a JSON report.

Usage:
    python -m evaluation.benchmark --config configs/base.yaml --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch


def run_benchmark(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    output_path: Path,
) -> Dict[str, Any]:
    """Run all evaluation protocols and save results.

    Args:
        model: Trained depth model in eval mode.
        config: Full experiment config dict.
        device: Device to run on.
        output_path: Path to save JSON report.

    Returns:
        Nested dict of all results.
    """
    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocols": {},
    }

    # 1. Clean baseline
    try:
        from evaluation.protocols.clean_baseline import run_clean_baseline
        results["protocols"]["clean_baseline"] = run_clean_baseline(model, config, device)
    except NotImplementedError:
        results["protocols"]["clean_baseline"] = "not_implemented"

    # 2. Per-degradation sweep
    try:
        from evaluation.protocols.per_degradation import run_degradation_sweep
        results["protocols"]["per_degradation"] = run_degradation_sweep(model, config, device)
    except NotImplementedError:
        results["protocols"]["per_degradation"] = "not_implemented"

    # 3. iPhone LiDAR eval
    try:
        from evaluation.protocols.iphone_lidar_eval import run_iphone_eval
        results["protocols"]["iphone_lidar"] = run_iphone_eval(model, config, device)
    except NotImplementedError:
        results["protocols"]["iphone_lidar"] = "not_implemented"

    # 4. Latency eval
    try:
        from evaluation.protocols.latency_eval import run_latency_eval
        results["protocols"]["latency"] = run_latency_eval(model, config, device)
    except NotImplementedError:
        results["protocols"]["latency"] = "not_implemented"

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="evaluation_report.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    raise NotImplementedError(
        "Wire up: load config, load checkpoint, call run_benchmark()"
    )


if __name__ == "__main__":
    main()
