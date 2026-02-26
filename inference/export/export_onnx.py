"""
Export to ONNX
================

Export trained FocusMamba model to ONNX format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def export_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 3, 8, 256, 256),
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export model to ONNX.

    Args:
        model: Trained model in eval mode.
        output_path: Path to save .onnx file.
        input_shape: (B, C, T, H, W) input tensor shape.
        opset_version: ONNX opset.
        dynamic_batch: Allow dynamic batch dimension.

    Returns:
        Path to exported file.
    """
    model.eval()
    dummy = torch.randn(input_shape)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "depth": {0: "batch"}}

    # NOTE: Mamba custom ops may not be directly exportable to ONNX.
    # May need to trace with simplified Mamba forward or use
    # torch.onnx.export with custom op registration.
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["depth"],
        dynamic_axes=dynamic_axes,
    )

    return output_path
