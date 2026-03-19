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
import yaml

from models import build_model


def _protocol_error(exc: Exception) -> Dict[str, Any]:
    return {
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


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

        results["protocols"]["clean_baseline"] = run_clean_baseline(
            model, config, device
        )
    except NotImplementedError:
        results["protocols"]["clean_baseline"] = "not_implemented"
    except (FileNotFoundError, ValueError) as exc:
        results["protocols"]["clean_baseline"] = _protocol_error(exc)

    # 2. RoboDepth benchmark
    try:
        from evaluation.protocols.robodepth import run_robodepth_eval

        results["protocols"]["robodepth"] = run_robodepth_eval(
            model,
            config,
            device,
            output_dir=output_path.parent / "robodepth",
        )
    except NotImplementedError:
        results["protocols"]["robodepth"] = "not_implemented"
    except (FileNotFoundError, ValueError) as exc:
        results["protocols"]["robodepth"] = _protocol_error(exc)

    # 3. Per-degradation sweep
    try:
        from evaluation.protocols.per_degradation import run_degradation_sweep

        results["protocols"]["per_degradation"] = run_degradation_sweep(
            model, config, device
        )
    except NotImplementedError:
        results["protocols"]["per_degradation"] = "not_implemented"
    except (FileNotFoundError, ValueError) as exc:
        results["protocols"]["per_degradation"] = _protocol_error(exc)

    # 4. iPhone LiDAR eval
    try:
        from evaluation.protocols.iphone_lidar_eval import run_iphone_eval

        results["protocols"]["iphone_lidar"] = run_iphone_eval(model, config, device)
    except NotImplementedError:
        results["protocols"]["iphone_lidar"] = "not_implemented"
    except (FileNotFoundError, ValueError) as exc:
        results["protocols"]["iphone_lidar"] = _protocol_error(exc)

    # 5. Latency eval
    try:
        from evaluation.protocols.latency_eval import run_latency_eval

        results["protocols"]["latency"] = run_latency_eval(model, config, device)
    except NotImplementedError:
        results["protocols"]["latency"] = "not_implemented"
    except (FileNotFoundError, ValueError) as exc:
        results["protocols"]["latency"] = _protocol_error(exc)

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

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    results = run_benchmark(
        model=model,
        config=config,
        device=device,
        output_path=Path(args.output),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
