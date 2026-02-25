"""
Transformer blocks for FocusMamba (Bidirectional Baseline)
==========================================================

Provides drop-in replacements for the Mamba-based blocks:
- ``TransformerBlock``: Standard bidirectional self-attention + MLP with pre-norm.
- ``SpatialTransformerBlock``: Reshapes (B,T,H,W,C) → spatial tokens, applies MHSA.
- ``TemporalTransformerBlock``: Reshapes (B,T,H,W,C) → temporal tokens, applies MHSA.

These accept the **same d_model** as the Mamba blocks so the encoder/decoder
channel widths remain identical and parameter counts can be matched by tuning
``num_heads`` and ``mlp_ratio``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RMSNorm (shared with Mamba blocks for consistency)
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
# Standard Bidirectional Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard Transformer encoder block: pre-norm MHSA + pre-norm MLP.

    Args:
        dim: Model / token dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension = dim * mlp_ratio.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, L, C)"""
        # Pre-norm self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Cross-Scan Transformer Block (matches CrossScanMambaBlock interface)
# ---------------------------------------------------------------------------

class CrossScanTransformerBlock(nn.Module):
    """Bidirectional self-attention over spatial tokens.

    Unlike the Mamba cross-scan variant which runs four independent SSMs in
    different scan orders, the Transformer attends to *all* tokens globally,
    so scan ordering is unnecessary.  This block is kept as a 1:1 drop-in
    for ``CrossScanMambaBlock`` so the same encoder/decoder code can be reused.

    Args:
        dim: Token dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        # Accept d_state/d_conv/expand for API compatibility but ignore them
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.block = TransformerBlock(dim, num_heads, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C) tokens in row-major spatial order.
            H, W: spatial grid dimensions (unused by attention but kept for API).
        Returns:
            (B, H*W, C)
        """
        return self.block(x)


# ---------------------------------------------------------------------------
# Spatial Transformer Block: (B, T, H, W, C) → attention over spatial dims
# ---------------------------------------------------------------------------

class SpatialTransformerBlock(nn.Module):
    """Applies bidirectional self-attention over the spatial dimensions.

    Reshapes (B, T, H, W, C) → (B*T, H*W, C), applies Transformer,
    then reshapes back.  Drop-in replacement for ``SpatialMambaBlock``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        # Accept Mamba-style kwargs for API compatibility
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.block = TransformerBlock(dim, num_heads, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C)"""
        B, T, H, W, C = x.shape
        x_flat = x.reshape(B * T, H * W, C)
        x_flat = self.block(x_flat)
        return x_flat.reshape(B, T, H, W, C)


# ---------------------------------------------------------------------------
# Temporal Transformer Block: (B, T, H, W, C) → attention over temporal dim
# ---------------------------------------------------------------------------

class TemporalTransformerBlock(nn.Module):
    """Applies bidirectional self-attention along the temporal dimension.

    Reshapes (B, T, H, W, C) → (B*H*W, T, C), applies Transformer, reshapes back.
    Drop-in replacement for ``TemporalMambaBlock``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        # Accept Mamba-style kwargs for API compatibility
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.block = TransformerBlock(dim, num_heads, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C)"""
        B, T, H, W, C = x.shape
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        x_flat = self.block(x_flat)
        return x_flat.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)
