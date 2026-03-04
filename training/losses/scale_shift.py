"""
Scale-Shift Invariant (SSI) Depth Loss
======================================

Implements the single-frame spatial supervision term used in MiDaS/VDA-style
training: prediction is first aligned to target depth using a per-sample
least-squares scale+shift, then a reconstruction loss is computed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleShiftInvariantLoss(nn.Module):
    """Scale-shift invariant depth reconstruction loss.

    Args:
        mode: Error metric after alignment: "l2" (default) or "l1".
        eps: Numerical guard for near-singular normal equations.
    """

    def __init__(self, mode: str = "l2", eps: float = 1e-6):
        super().__init__()
        if mode not in ("l1", "l2"):
            raise ValueError(f"Unknown SSI mode: {mode!r}. Use 'l1' or 'l2'.")
        self.mode = mode
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        aligned_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute SSI loss.

        Args:
            pred: (B, 1, T, H, W) prediction.
            target: (B, 1, T, H, W) target depth.
            mask: optional validity mask, same shape.
            aligned_pred: pre-aligned prediction (optional).
        """
        if aligned_pred is None:
            aligned = self.align_scale_shift(pred, target, mask=mask, eps=self.eps)
        else:
            aligned = aligned_pred

        err = aligned - target
        if mask is not None:
            err = err[mask.bool()]
        if err.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if self.mode == "l1":
            return err.abs().mean()
        return (err * err).mean()

    @staticmethod
    def align_scale_shift(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Align pred to target with per-sample affine transform s*pred + t.

        Alignment is solved over all valid pixels in each sample's full clip
        (shared scale+shift across time), matching VDA's shared video alignment.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )

        if mask is None:
            valid = torch.isfinite(target) & torch.isfinite(pred)
        else:
            valid = mask.bool() & torch.isfinite(target) & torch.isfinite(pred)

        B = pred.shape[0]
        aligned = pred.clone()

        for b in range(B):
            vb = valid[b]
            if vb.sum() < 2:
                continue

            p = pred[b][vb]
            g = target[b][vb]

            a00 = (p * p).sum()
            a01 = p.sum()
            a11 = torch.tensor(float(p.numel()), device=pred.device, dtype=pred.dtype)
            b0 = (p * g).sum()
            b1 = g.sum()

            det = a00 * a11 - a01 * a01
            if torch.abs(det) <= eps:
                continue

            s = (a11 * b0 - a01 * b1) / det
            t = (-a01 * b0 + a00 * b1) / det
            aligned[b] = s * pred[b] + t

        return aligned

