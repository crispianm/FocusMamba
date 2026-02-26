"""
Demo — Webcam / Video File Real-Time Depth Overlay
=====================================================

Usage:
    python -m inference.demo --checkpoint checkpoints/best.pt --source webcam
    python -m inference.demo --checkpoint checkpoints/best.pt --source video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def run_demo(
    checkpoint_path: str,
    source: str = "webcam",
    device: str = "cuda",
    resolution: int = 256,
) -> None:
    """Run live depth estimation demo.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        source: 'webcam' or path to video file.
        device: 'cuda' or 'cpu'.
        resolution: Resize resolution (shortest side).
    """
    raise NotImplementedError(
        "Implement: \n"
        "1. Load model from checkpoint\n"
        "2. Create RealtimeDepthEngine\n"
        "3. Open video source (cv2.VideoCapture)\n"
        "4. Loop: read frame → process_frame → colorise → overlay → display\n"
        "5. Show focus distance if AutofocusInterface is enabled"
    )


def main():
    parser = argparse.ArgumentParser(description="Real-time depth demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, default="webcam")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    run_demo(args.checkpoint, args.source, args.device, args.resolution)


if __name__ == "__main__":
    main()
