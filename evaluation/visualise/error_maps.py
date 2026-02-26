"""
Error Maps Visualisation
=========================

Generate depth error maps for qualitative comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import numpy as np


def generate_error_map(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    output_path: Optional[Path] = None,
    cmap: str = "hot",
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Generate a colourised absolute error map.

    Args:
        pred_depth: (H, W) predicted depth.
        gt_depth:   (H, W) ground-truth depth.
        output_path: If provided, save the image.
        cmap: Matplotlib colormap.
        vmax: Max value for colour normalisation.

    Returns:
        (H, W, 3) uint8 RGB error map.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    error = (pred_depth - gt_depth).abs().cpu().numpy()
    if vmax is None:
        vmax = np.percentile(error, 95)

    normalised = np.clip(error / max(vmax, 1e-8), 0, 1)
    colormap = cm.get_cmap(cmap)
    colored = (colormap(normalised)[:, :, :3] * 255).astype(np.uint8)

    if output_path is not None:
        from PIL import Image
        Image.fromarray(colored).save(output_path)

    return colored
