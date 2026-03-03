"""
Depth Model — Top-Level Module
================================

Assembles encoder + temporal + decoder into a complete depth estimation model.
This is the canonical entry point per the codebase outline's models/depth_model.py.

For the actual implementations, see:
    - models/focus_mamba.py   (Mamba-based, primary model)
    - models/focus_transformer.py (Transformer ablation baseline)
    - models/__init__.py      (build_model factory)
"""

from __future__ import annotations

from models import build_model, FocusMamba, FocusTransformer, VideoDepthAnythingModel

__all__ = ["build_model", "FocusMamba", "FocusTransformer", "VideoDepthAnythingModel"]
