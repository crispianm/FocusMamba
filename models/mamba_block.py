"""
Mamba blocks for FocusMamba
============================

Provides:
- ``MambaBlock``: Single-direction Mamba with pre-norm + residual.
- ``CrossScanMambaBlock``: 4-way cross-scan (row LR/RL, col TB/BT).
- ``SpatialMambaBlock``: Reshapes (B,T,H,W,C) → spatial tokens, applies cross-scan.
- ``TemporalMambaBlock``: Reshapes (B,T,H,W,C) → temporal tokens, applies Mamba.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None  # graceful degrade for environments without mamba_ssm


# ---------------------------------------------------------------------------
# RMSNorm (from the original Mamba paper convention)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Single Mamba block (pre-norm, residual)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Single-direction Mamba with pre-RMSNorm and residual connection.

    Args:
        dim: Model dimension.
        d_state: SSM state expansion factor.
        d_conv: Local convolution width.
        expand: Block expansion factor.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        if Mamba is not None:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: simple linear layer for testing without mamba_ssm
            self.mamba = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.SiLU(),
                nn.Linear(dim * expand, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, L, C)"""
        return x + self.mamba(self.norm(x))


# ---------------------------------------------------------------------------
# Cross-scan Mamba (4-way spatial scan)
# ---------------------------------------------------------------------------

class CrossScanMambaBlock(nn.Module):
    """Four independent Mamba SSMs scanning in 4 spatial directions.

    Given tokens arranged on a 2-D grid (H', W'), the scans are:
        0: row-major left→right   (standard raster)
        1: row-major right→left   (reversed raster)
        2: column-major top→bottom
        3: column-major bottom→top

    The four outputs are summed (learned-weighted) and passed through a
    residual connection with pre-norm.

    Args:
        dim: Token dimension C.
        d_state, d_conv, expand: Mamba hyper-params.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ssms = nn.ModuleList([
            self._make_ssm(dim, d_state, d_conv, expand) for _ in range(4)
        ])
        # Learnable weights for combining the 4 scan directions
        self.scan_weights = nn.Parameter(torch.ones(4) / 4.0)

    @staticmethod
    def _make_ssm(dim, d_state, d_conv, expand):
        if Mamba is not None:
            return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        return nn.Sequential(nn.Linear(dim, dim * expand), nn.SiLU(), nn.Linear(dim * expand, dim))

    @staticmethod
    def _row_major(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, H*W, C) already in row-major order → identity."""
        return x

    @staticmethod
    def _row_major_rev(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return x.flip(1)

    @staticmethod
    def _col_major(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        return x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)

    @staticmethod
    def _col_major_rev(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        return x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C).flip(1)

    @staticmethod
    def _undo_col_major(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        return x.view(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C) tokens in row-major spatial order.
            H, W: spatial grid dimensions.
        Returns:
            (B, H*W, C)
        """
        residual = x
        x_normed = self.norm(x)

        # Build scan sequences
        scans = [
            self._row_major(x_normed, H, W),
            self._row_major_rev(x_normed, H, W),
            self._col_major(x_normed, H, W),
            self._col_major_rev(x_normed, H, W),
        ]

        # Run through independent SSMs
        outs = [ssm(s) for ssm, s in zip(self.ssms, scans)]

        # Reverse the reordering to get back to row-major
        outs[1] = outs[1].flip(1)                        # undo reverse
        outs[2] = self._undo_col_major(outs[2], H, W)    # undo col-major
        outs[3] = self._undo_col_major(outs[3].flip(1), H, W)  # undo col-major+reverse

        # Weighted sum
        w = self.scan_weights.softmax(dim=0)
        combined = sum(w[i] * outs[i] for i in range(4))

        return residual + combined


# ---------------------------------------------------------------------------
# Spatial Mamba Block: (B, T, H, W, C) → cross-scan over spatial dims
# ---------------------------------------------------------------------------

class SpatialMambaBlock(nn.Module):
    """Applies 4-way CrossScanMamba over the spatial dimensions.

    Reshapes (B, T, H, W, C) → (B*T, H*W, C), applies cross-scan,
    then reshapes back.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.cross_scan = CrossScanMambaBlock(dim, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C)"""
        B, T, H, W, C = x.shape
        x_flat = x.reshape(B * T, H * W, C)
        x_flat = self.cross_scan(x_flat, H, W)
        return x_flat.reshape(B, T, H, W, C)


# ---------------------------------------------------------------------------
# Temporal Mamba Block: (B, T, H, W, C) → Mamba over temporal dim
# ---------------------------------------------------------------------------

class TemporalMambaBlock(nn.Module):
    """Applies single-direction Mamba along the temporal dimension.

    Reshapes (B, T, H, W, C) → (B*H*W, T, C), applies Mamba, reshapes back.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.mamba_block = MambaBlock(dim, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C)"""
        B, T, H, W, C = x.shape
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        x_flat = self.mamba_block(x_flat)
        return x_flat.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)
