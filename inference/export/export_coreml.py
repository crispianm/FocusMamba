"""
Export to CoreML
==================

For potential on-device iPhone deployment / demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def export_coreml(
    model_path: Path,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 3, 1, 256, 256),
) -> Path:
    """Convert model to CoreML format.

    Args:
        model_path: Path to PyTorch or ONNX model.
        output_path: Path to save .mlmodel / .mlpackage.
        input_shape: Input shape (likely single frame for mobile).

    Returns:
        Path to CoreML model.
    """
    raise NotImplementedError(
        "Implement: use coremltools to convert from PyTorch/ONNX → CoreML. "
        "May need Mamba op simplification for ANE compatibility."
    )
