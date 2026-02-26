"""
Export to TensorRT
====================

TRT optimisation for NVIDIA Jetson / camera workstations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def export_tensorrt(
    onnx_path: Path,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 3, 8, 256, 256),
    fp16: bool = True,
) -> Path:
    """Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model.
        output_path: Path to save .engine file.
        input_shape: Input shape for optimisation.
        fp16: Enable FP16 mode.

    Returns:
        Path to TensorRT engine.
    """
    raise NotImplementedError(
        "Implement: use tensorrt Python API or trtexec to convert "
        "ONNX → TensorRT engine with FP16 optimisation."
    )
