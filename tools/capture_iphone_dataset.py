"""
Capture iPhone Dataset
========================

Instructions and utilities for paired RGB+LiDAR capture on iPhone Pro.
"""

from __future__ import annotations


CAPTURE_INSTRUCTIONS = """
iPhone LiDAR Dataset Capture Guide
=====================================

Requirements:
    - iPhone 12 Pro or later (with LiDAR scanner)
    - Record3D app (recommended) or custom ARKit app

Capture Protocol:
    1. Set Record3D to capture RGB + depth at highest available resolution
    2. Record at 30fps, 10-30 second clips
    3. Capture in diverse environments:
       - Indoor (office, home, studio)
       - Outdoor (street, park, architecture)
       - Mixed lighting (bright, dim, mixed)
    4. Include varied subject distances (0.3m - 10m)
    5. Include camera motion (slow pan, dolly, static tripod)

Export:
    - Export RGB frames as PNG sequence
    - Export depth maps as 16-bit PNG (depth in mm) or NPZ
    - Export camera intrinsics JSON

Directory Structure:
    iphone_dataset/
    ├── scene_001/
    │   ├── rgb/          # RGB frames (0000.png, 0001.png, ...)
    │   ├── depth/        # Depth maps (0000.png, 0001.png, ...)
    │   ├── intrinsics.json
    │   └── metadata.json # Scene description, lighting, etc.
    ├── scene_002/
    │   └── ...
    └── splits.json       # Train/val/test split
"""


def validate_iphone_capture(dataset_dir: str) -> dict:
    """Validate an iPhone capture dataset for completeness.

    Returns:
        Dict with validation results: n_scenes, n_frames, issues.
    """
    raise NotImplementedError(
        "Implement: check directory structure, frame counts, depth validity."
    )
