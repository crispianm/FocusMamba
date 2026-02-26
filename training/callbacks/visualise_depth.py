"""
Visualization Callback — Logs Colourised Depth Predictions
============================================================

Logs depth map visualisations to TensorBoard during training/validation.
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np


def colorise_depth(
    depth: torch.Tensor,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "turbo",
) -> torch.Tensor:
    """Convert a single-channel depth map to a colourised RGB image.

    Args:
        depth: (H, W) or (1, H, W) depth tensor.
        vmin: Minimum depth value for normalisation.
        vmax: Maximum depth value for normalisation.
        cmap: Matplotlib colormap name.

    Returns:
        (3, H, W) RGB tensor in [0, 1].
    """
    try:
        import matplotlib.cm as cm
    except ImportError:
        # Fallback: grayscale
        if depth.dim() == 3:
            depth = depth.squeeze(0)
        d = depth.float()
        if vmin is None:
            vmin = d.min().item()
        if vmax is None:
            vmax = d.max().item()
        d = (d - vmin) / max(vmax - vmin, 1e-8)
        return d.unsqueeze(0).repeat(3, 1, 1)

    if depth.dim() == 3:
        depth = depth.squeeze(0)

    d = depth.float().cpu().numpy()
    if vmin is None:
        vmin = np.nanpercentile(d, 2)
    if vmax is None:
        vmax = np.nanpercentile(d, 98)

    d = np.clip((d - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    colormap = cm.get_cmap(cmap)
    colored = colormap(d)[:, :, :3]  # (H, W, 3)
    return torch.from_numpy(colored).permute(2, 0, 1).float()


def log_depth_visualisation(
    writer,
    tag: str,
    pred_depth: torch.Tensor,
    gt_depth: Optional[torch.Tensor] = None,
    step: int = 0,
    max_samples: int = 4,
) -> None:
    """Log colourised depth maps to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        tag: Tag prefix for the images.
        pred_depth: (B, 1, T, H, W) predicted depth.
        gt_depth: (B, 1, T, H, W) optional ground-truth depth.
        step: Global step.
        max_samples: Max number of samples to log.
    """
    B = min(pred_depth.shape[0], max_samples)

    for b in range(B):
        # Take the middle frame
        t_mid = pred_depth.shape[2] // 2
        pred_frame = pred_depth[b, 0, t_mid]  # (H, W)

        # Compute normalisation range from this batch
        vmax = pred_frame.quantile(0.98).item()
        vmin = pred_frame.quantile(0.02).item()

        pred_rgb = colorise_depth(pred_frame, vmin=vmin, vmax=vmax)
        writer.add_image(f"{tag}/pred_{b}", pred_rgb, step)

        if gt_depth is not None:
            gt_frame = gt_depth[b, 0, t_mid]
            gt_rgb = colorise_depth(gt_frame, vmin=vmin, vmax=vmax)
            writer.add_image(f"{tag}/gt_{b}", gt_rgb, step)
