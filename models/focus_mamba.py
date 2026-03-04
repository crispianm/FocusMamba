"""FocusMamba v2: VDA-style depth model with a Mamba backbone.

Design goals:
    - Stay close to Video Depth Anything small in patch pipeline and head.
    - Replace the ViT encoder with a spatiotemporal Mamba token backbone.
    - Keep the repository interface: input (B,C,T,H,W) -> {"depth": ...}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mamba_block as mamba_core
from .mamba_block import SpatialMambaBlock, TemporalMambaBlock
from .teachers.vendor.video_depth_anything.dpt_temporal import DPTHeadTemporal


_V2_VARIANTS = {
    "small": {
        "embed_dim": 384,
        "num_blocks": 12,
        "out_indices": (2, 5, 8, 11),
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "base": {
        "embed_dim": 768,
        "num_blocks": 12,
        "out_indices": (2, 5, 8, 11),
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
}


def _clean_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        key = key.removeprefix("module.")
        key = key.removeprefix("model.")
        cleaned[key] = value
    return cleaned


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class _SpatioTemporalMambaBlock(nn.Module):
    """One encoder block: spatial Mamba -> temporal Mamba -> channel FFN."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.spatial = SpatialMambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.temporal = TemporalMambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ffn = _FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.ffn(x)
        return x


class _FocusMambaBackbone(nn.Module):
    """VDA-like patch/token backbone with Mamba blocks.

    Input:  (B, T, C, H, W)
    Output: list of 4 intermediate token maps in VDA-compatible format:
            [(tokens, cls_dummy), ...], where tokens = (B*T, N, C)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_blocks: int,
        out_indices: Sequence[int],
        patch_size: int = 14,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        base_img_size: int = 518,
        require_mamba: bool = True,
    ) -> None:
        super().__init__()
        if require_mamba and mamba_core.Mamba is None:
            raise ImportError(
                "FocusMamba v2 requires mamba_ssm CUDA extensions, but import failed "
                "(missing selective_scan_cuda). Build/install mamba_ssm first, or set "
                "model.require_mamba=false to allow fallback blocks."
            )
        if patch_size != 14:
            warnings.warn(
                "FocusMamba v2 is designed for patch_size=14 to match VDA head scaling. "
                f"Received patch_size={patch_size}; forcing 14.",
                stacklevel=2,
            )
            patch_size = 14

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.out_indices = tuple(sorted({int(i) for i in out_indices}))
        self._out_set = set(self.out_indices)

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        base_grid = base_img_size // patch_size
        self.base_grid = base_grid
        self.pos_embed = nn.Parameter(torch.zeros(1, base_grid * base_grid, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _SpatioTemporalMambaBlock(
                embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _interpolate_pos_embed(
        self,
        h_tokens: int,
        w_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        c = self.pos_embed.shape[-1]
        pos = self.pos_embed.view(1, self.base_grid, self.base_grid, c).permute(0, 3, 1, 2)
        if (self.base_grid, self.base_grid) != (h_tokens, w_tokens):
            pos = F.interpolate(pos, size=(h_tokens, w_tokens), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).to(device=device, dtype=dtype)

    def forward(
        self,
        frames: torch.Tensor,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, int, int, int]:
        b, t, c, h, w = frames.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        x = frames.reshape(b * t, c, h, w)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        h_pad, w_pad = x.shape[-2:]

        x = self.patch_embed(x)  # (B*T, C, Ht, Wt)
        _, c_tok, h_tok, w_tok = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, t, h_tok, w_tok, c_tok)

        pos = self._interpolate_pos_embed(h_tok, w_tok, x.dtype, x.device).unsqueeze(1)
        x = x + pos

        intermediates: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self._out_set:
                tokens = x.reshape(b * t, h_tok * w_tok, c_tok)
                tokens = self.norm(tokens)
                cls_dummy = tokens.new_zeros(tokens.shape[0], c_tok)
                intermediates.append((tokens, cls_dummy))

        return intermediates, h_tok, w_tok, h_pad, w_pad


class FocusMamba(nn.Module):
    """FocusMamba v2 (overwrites original model name).

    This model mirrors VDA-small structure at a high level:
      patch embedding -> 12 token blocks -> 4 multi-level taps -> DPT temporal head.
    The novelty is in the encoder token mixer: Mamba (spatial+temporal) blocks
    replace ViT attention blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int | None = None,
        depths: list[int] | None = None,  # kept for legacy config compatibility
        patch_size: int = 14,
        t_patch: int = 1,  # unused, kept for compatibility
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        predict_uncertainty: bool = False,
        variant: str = "small",
        num_blocks: int | None = None,
        out_indices: Sequence[int] | None = None,
        mlp_ratio: float = 4.0,
        num_frames: int = 8,
        positional_encoding: str = "ape",
        checkpoint_path: str | None = None,
        strict_checkpoint: bool = False,
        require_mamba: bool = True,
    ) -> None:
        super().__init__()

        key = variant.lower()
        if key not in _V2_VARIANTS:
            valid = ", ".join(sorted(_V2_VARIANTS))
            raise ValueError(f"Unknown FocusMamba variant '{variant}'. Available: {valid}")
        vcfg = _V2_VARIANTS[key]

        legacy_layout = depths is not None and (num_blocks is None and out_indices is None)
        if legacy_layout:
            warnings.warn(
                "FocusMamba v2 ignores legacy 'depths/encoder_depths' by default. "
                "Use model.num_blocks/model.out_indices to override token depth.",
                stacklevel=2,
            )
            if embed_dim is not None:
                warnings.warn(
                    "Legacy embed_dim was provided together with encoder_depths; "
                    "using variant default embed_dim for FocusMamba v2. "
                    "Set model.embed_dim explicitly without encoder_depths to override.",
                    stacklevel=2,
                )
                embed_dim = None
        if t_patch != 1:
            warnings.warn("FocusMamba v2 uses per-frame patch embedding; t_patch is ignored.", stacklevel=2)

        enc_dim = int(embed_dim if embed_dim is not None else vcfg["embed_dim"])
        enc_blocks = int(num_blocks if num_blocks is not None else vcfg["num_blocks"])
        taps = tuple(int(i) for i in (out_indices if out_indices is not None else vcfg["out_indices"]))
        if len(taps) != 4:
            raise ValueError(f"FocusMamba expects 4 out_indices for DPT head, got {taps!r}")

        # For non-standard embed_dim, scale VDA-small/base head widths proportionally.
        if enc_dim == vcfg["embed_dim"]:
            head_features = vcfg["features"]
            head_out_channels = vcfg["out_channels"]
        else:
            ratio = float(enc_dim) / float(vcfg["embed_dim"])
            head_features = max(16, int(round(vcfg["features"] * ratio)))
            head_out_channels = [max(8, int(round(ch * ratio))) for ch in vcfg["out_channels"]]
            head_out_channels[-1] = enc_dim

        self.variant = key
        self.patch_size = 14  # fixed by the VDA DPT head upsample logic
        self.predict_uncertainty = bool(predict_uncertainty)

        self.backbone = _FocusMambaBackbone(
            in_channels=in_channels,
            embed_dim=enc_dim,
            num_blocks=enc_blocks,
            out_indices=taps,
            patch_size=patch_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            require_mamba=require_mamba,
        )
        self.head = DPTHeadTemporal(
            in_channels=enc_dim,
            features=head_features,
            use_bn=False,
            out_channels=head_out_channels,
            use_clstoken=False,
            num_frames=num_frames,
            pe=positional_encoding,
        )

        # ImageNet normalization (same as VDA / DINOv2 input contract).
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        if self.predict_uncertainty:
            # Lightweight uncertainty output for compatibility with optional losses.
            self.uncertainty_bias = nn.Parameter(torch.zeros(()))

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, strict=strict_checkpoint)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load a FocusMamba-v2 checkpoint (scratch-training resume path)."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")
        cleaned = _clean_state_dict(checkpoint)

        mapped: dict[str, torch.Tensor] = {}
        for key, value in cleaned.items():
            if key.startswith("backbone.") or key.startswith("head.") or key == "uncertainty_bias":
                mapped[key] = value

        current_state = self.state_dict()
        compatible_state = {}
        skipped_shape: list[str] = []
        for key, value in mapped.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            elif key in current_state:
                skipped_shape.append(key)

        missing_keys, unexpected_keys = self.load_state_dict(compatible_state, strict=False)
        if strict and (missing_keys or unexpected_keys or skipped_shape):
            raise RuntimeError(
                "Strict checkpoint loading failed for FocusMamba. "
                f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
                f"shape_mismatch={len(skipped_shape)}"
            )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            frames: (B, C, T, H, W) float tensor in [0, 1].
        Returns:
            Dict with `depth` in shape (B, 1, T, H, W), and optional `uncertainty`.
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(frames.shape)}")
        b, c, t, h, w = frames.shape
        if c != 3:
            raise ValueError(f"FocusMamba expects 3 RGB channels, got {c}")

        x = frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        mean = self._mean.to(dtype=x.dtype, device=x.device)
        std = self._std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) / std

        features, patch_h, patch_w, h_pad, w_pad = self.backbone(x)
        if len(features) != 4:
            raise RuntimeError(f"Backbone produced {len(features)} feature maps, expected 4.")

        depth_bt = self.head(features, patch_h, patch_w, t)[0]   # (B*T,1,H',W')
        depth_bt = F.interpolate(depth_bt, size=(h_pad, w_pad), mode="bilinear", align_corners=True)
        depth_bt = F.relu(depth_bt)

        depth_bthw = depth_bt.squeeze(1).unflatten(0, (b, t))  # (B,T,H_pad,W_pad)
        depth_bthw = depth_bthw[:, :, :h, :w]
        depth = depth_bthw.unsqueeze(1)  # (B,1,T,H,W)

        outputs = {"depth": depth}
        if self.predict_uncertainty:
            outputs["uncertainty"] = torch.ones_like(depth) * self.uncertainty_bias
        return outputs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, input_shape: Tuple[int, ...] = (1, 3, 8, 392, 392)) -> int:
        try:
            from fvcore.nn import FlopCountAnalysis

            dummy = torch.zeros(input_shape, device=next(self.parameters()).device)
            return int(FlopCountAnalysis(self, (dummy,)).total())
        except Exception:
            b, _, t, h, w = input_shape
            tokens = t * ((h // self.patch_size) * (w // self.patch_size))
            return 2 * self.count_parameters() * b * tokens


if __name__ == "__main__":
    model = FocusMamba(variant="small", require_mamba=False)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Estimated FLOPs: {model.estimate_flops():,}")

    x = torch.randn(1, 3, 4, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Depth:  {out['depth'].shape}")
