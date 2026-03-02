"""FocusMamba — Degradation-Robust Metric Video Depth Estimation
================================================================

Assembles TubeletEmbedding → Encoder → Decoder → Metric Depth Head.

Inputs:
    frames: (B, C, T, H, W) — video clip (RGB, float [0,1])

Outputs:
    depth: (B, 1, T, H, W) — metric depth in metres (positive, via exp(log_depth))
    uncertainty: (B, 1, T, H, W) — per-pixel aleatoric uncertainty (optional)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .encoder import FocusMambaEncoder
from .decoder import FocusMambaDecoder


class FocusMamba(nn.Module):
    """End-to-end FocusMamba depth estimation model.

    Predicts metric depth (in metres) from degraded video input.
    The decoder predicts log-depth which is exponentiated to enforce positivity.
    Optionally predicts per-pixel aleatoric uncertainty.

    Args:
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Base embedding dimension.
        depths: Number of Spatial+Temporal block pairs per encoder stage.
        patch_size: Spatial patch size for tubelet embedding.
        t_patch: Temporal patch size for tubelet embedding.
        d_state: Mamba SSM state dimension.
        d_conv: Mamba local convolution width.
        expand: Mamba inner expansion factor.
        predict_uncertainty: If True, also predict per-pixel uncertainty.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] | None = None,
        patch_size: int = 4,
        t_patch: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        predict_uncertainty: bool = False,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 4, 2]

        self.predict_uncertainty = predict_uncertainty

        self.encoder = FocusMambaEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            patch_size=patch_size,
            t_patch=t_patch,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.decoder = FocusMambaDecoder(
            embed_dim=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            predict_uncertainty=predict_uncertainty,
        )

        self.patch_size = patch_size
        self.t_patch = t_patch

    def forward(
        self, frames: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, C, T, H, W) float32 in [0,1].
        Returns:
            dict with:
                'depth': (B, 1, T, H, W) metric depth in metres (always positive)
                'uncertainty': (B, 1, T, H, W) aleatoric uncertainty (if enabled)
        """
        B, C, T, H, W = frames.shape
        skips, bottleneck = self.encoder(frames)
        outputs = self.decoder(skips, bottleneck)

        # Ensure output matches input spatial/temporal resolution
        for key in outputs:
            if outputs[key].shape[2:] != (T, H, W):
                outputs[key] = torch.nn.functional.interpolate(
                    outputs[key], size=(T, H, W), mode="trilinear", align_corners=False,
                )
        return outputs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(
        self, input_shape: Tuple[int, ...] = (1, 3, 8, 256, 256)
    ) -> int:
        """Rough FLOPs estimate for a single forward pass.

        Tries to use ``fvcore`` if available, otherwise falls back to a
        parameter-count-based heuristic (2 × params × tokens).
        """
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy_frames = torch.zeros(input_shape, device=next(self.parameters()).device)
            flops = FlopCountAnalysis(self, (dummy_frames)).total()
            return int(flops)
        except Exception:
            # Heuristic: ~2 MACs per parameter per spatial-temporal token
            B, C, T, H, W = input_shape
            n_tokens = (T // self.t_patch) * (H // self.patch_size) * (W // self.patch_size)
            return 2 * self.count_parameters() * n_tokens


if __name__ == "__main__":
    # Quick smoke test
    model = FocusMamba(
        in_channels=3, embed_dim=96, depths=[2, 2, 4, 2],
        patch_size=4, t_patch=2,
    )
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Estimated FLOPs: {model.estimate_flops():,}")

    x = torch.randn(1, 3, 8, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (1, 1, 8, 256, 256), f"Shape mismatch: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output out of [0,1] range"
    print("Smoke test passed!")
