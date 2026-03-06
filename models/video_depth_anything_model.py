"""Trainable Video Depth Anything wrapper.

Provides a model interface compatible with the repository training loop:
input ``(B, C, T, H, W)`` -> output dict with ``depth`` in
``(B, 1, T, H, W)``.

For tiny variants, this file now provides a novel dual-head architecture:
- DINOv3-small encoder (timm EVA implementation)
- temporal DPT decoder for relative depth
- clip-level metric calibration head to map relative depth -> metric depth

For legacy VDA checkpoints trained for relative depth, the wrapper can also
insert a lightweight inverse-depth bridge so fine-tuning in metric space starts
from the correct depth ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.teachers.vendor.video_depth_anything import VideoDepthAnything
from models.teachers.vendor.video_depth_anything.dpt_temporal import DPTHeadTemporal


_VDA_VARIANTS: Dict[str, Dict] = {
    # Legacy VDA-style variants.
    "tiny": {
        "encoder": "vits",
        "features": 32,
        "out_channels": [24, 48, 96, 192],
    },
    "small": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "base": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "large": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    # Encoder aliases
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vittiny": {
        "encoder": "vits",
        "features": 32,
        "out_channels": [24, 48, 96, 192],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


class TinyDualHeadDINOv3(nn.Module):
    """Tiny dual-head model with DINOv3 encoder and temporal DPT decoder.

    The decoder predicts relative depth, then a dedicated calibration head
    predicts clip-level affine parameters used to recover metric depth.
    """

    def __init__(
        self,
        num_frames: int = 8,
        positional_encoding: str = "ape",
        dino_model_name: str = "vit_small_patch16_dinov3",
        pretrained: bool = True,
        out_indices: Sequence[int] = (2, 5, 8, 11),
        features: int = 64,
        out_channels: Sequence[int] = (64, 128, 256, 384),
        relative_bounded: bool = True,
        relative_activation: str = "softplus",
        metric_activation: str = "softplus",
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "TinyDualHeadDINOv3 requires timm. Install with `uv add timm`."
            ) from exc

        self.backbone = timm.create_model(
            dino_model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.embed_dim = int(getattr(self.backbone, "num_features", 384))
        patch_size = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", (16, 16))
        self.patch_size = int(patch_size[0]) if isinstance(patch_size, tuple) else int(patch_size)

        self.out_indices = tuple(int(i) for i in out_indices)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        # Relative depth decoder (no final activation; applied explicitly below).
        self.relative_head = DPTHeadTemporal(
            in_channels=self.embed_dim,
            features=features,
            use_bn=False,
            out_channels=list(out_channels),
            use_clstoken=False,
            num_frames=num_frames,
            pe=positional_encoding,
            patch_size=self.patch_size,
            output_activation="none",
        )

        # Clip-level metric calibration from final backbone descriptors.
        hidden = max(self.embed_dim // 2, 128)
        self.metric_calibrator = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # scale_raw, shift_raw
        )

        self.relative_bounded = bool(relative_bounded)
        self.relative_activation = str(relative_activation).lower()
        self.metric_activation = str(metric_activation).lower()

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

    def _apply_relative_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.relative_bounded:
            return torch.sigmoid(x)
        if self.relative_activation == "softplus":
            return F.softplus(x)
        if self.relative_activation == "exp":
            return torch.exp(x)
        if self.relative_activation in ("none", "identity", "linear"):
            return x
        return F.softplus(x)

    def _apply_metric_affine(self, rel_depth: torch.Tensor, scale_raw: torch.Tensor, shift_raw: torch.Tensor) -> torch.Tensor:
        if self.metric_activation == "exp":
            scale = torch.exp(scale_raw).view(-1, 1, 1, 1, 1)
        else:
            # Softplus is more stable than ReLU for early-stage calibration.
            scale = F.softplus(scale_raw).view(-1, 1, 1, 1, 1) + 1e-4
        shift = shift_raw.view(-1, 1, 1, 1, 1)
        return scale * rel_depth + shift

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load checkpoint with shape-compatible filtering by default."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        cleaned_state = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            key = key.removeprefix("module.")
            key = key.removeprefix("model.")
            cleaned_state[key] = value

        current_state = self.state_dict()
        compatible_state = {}
        skipped_shape = []
        for key, value in cleaned_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            elif key in current_state:
                skipped_shape.append(key)

        missing_keys, unexpected_keys = self.load_state_dict(compatible_state, strict=False)
        if strict and (missing_keys or unexpected_keys or skipped_shape):
            raise RuntimeError(
                "Strict checkpoint loading failed for TinyDualHeadDINOv3. "
                f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
                f"shape_mismatch={len(skipped_shape)}"
            )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(frames.shape)}")

        b, channels, t, h, w = frames.shape
        if channels != 3:
            raise ValueError(f"TinyDualHeadDINOv3 expects 3 RGB channels, got {channels}")

        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            frames = F.pad(frames, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")

        _, _, _, h_pad, w_pad = frames.shape
        patch_h, patch_w = h_pad // self.patch_size, w_pad // self.patch_size

        x = frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        mean = self._mean.to(dtype=x.dtype, device=x.device)
        std = self._std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) / std

        x_bt = x.flatten(0, 1)  # (B*T, C, H, W)
        final_tokens, inter = self.backbone.forward_intermediates(
            x_bt,
            indices=self.out_indices,
            return_prefix_tokens=True,
            norm=False,
            output_fmt="NCHW",
        )

        out_features = []
        for feat_map, prefix in inter:
            tokens = feat_map.flatten(2).transpose(1, 2).contiguous()  # (B*T, N, C)
            if prefix is not None and prefix.shape[1] > 0:
                cls_token = prefix[:, 0]
            else:
                cls_token = tokens.mean(dim=1)
            out_features.append((tokens, cls_token))

        rel_logits_bt = self.relative_head(out_features, patch_h, patch_w, t)[0]  # (B*T,1,H_pad,W_pad)
        rel_logits_bt = rel_logits_bt[:, :, :h_pad, :w_pad]

        rel_depth_bt = self._apply_relative_activation(rel_logits_bt)
        rel_depth = rel_depth_bt.squeeze(1).unflatten(0, (b, t)).unsqueeze(1)  # (B,1,T,H_pad,W_pad)

        # Clip-level metric calibration (shared over all frames for consistency).
        patch_tokens = final_tokens[:, self.num_prefix_tokens :, :]
        pooled = patch_tokens.mean(dim=1)  # (B*T, C)
        calib_bt = self.metric_calibrator(pooled).unflatten(0, (b, t))  # (B, T, 2)
        calib_clip = calib_bt.mean(dim=1)  # (B, 2)
        scale_raw = calib_clip[:, 0]
        shift_raw = calib_clip[:, 1]

        metric_depth = self._apply_metric_affine(rel_depth, scale_raw, shift_raw)

        rel_depth = rel_depth[:, :, :, :h, :w]
        metric_depth = metric_depth[:, :, :, :h, :w]

        return {
            "depth": metric_depth,
            "depth_relative": rel_depth,
            "metric_scale": F.softplus(scale_raw).detach(),
            "metric_shift": shift_raw.detach(),
        }


class LegacyInverseAffineBridge(nn.Module):
    """Convert disparity-like legacy VDA output into relative + metric depth.

    Relative VDA checkpoints produce larger values for nearer objects. For
    metric supervision we first flip the ordering via a reciprocal transform,
    then apply a clip-level affine calibration predicted from the backbone
    descriptors.
    """

    def __init__(
        self,
        embed_dim: int,
        reciprocal_floor: float = 1e-2,
        metric_activation: str = "softplus",
    ) -> None:
        super().__init__()
        hidden = max(int(embed_dim) // 2, 128)
        self.metric_calibrator = nn.Sequential(
            nn.LayerNorm(int(embed_dim)),
            nn.Linear(int(embed_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # scale_raw, shift_raw
        )
        self.reciprocal_floor = float(reciprocal_floor)
        self.metric_activation = str(metric_activation).lower()

    def _apply_metric_affine(
        self,
        rel_depth: torch.Tensor,
        scale_raw: torch.Tensor,
        shift_raw: torch.Tensor,
    ) -> torch.Tensor:
        if self.metric_activation == "exp":
            scale = torch.exp(scale_raw).view(-1, 1, 1, 1, 1)
        else:
            scale = F.softplus(scale_raw).view(-1, 1, 1, 1, 1) + 1e-4
        shift = shift_raw.view(-1, 1, 1, 1, 1)
        return scale * rel_depth + shift

    def forward(
        self,
        raw_inverse_depth: torch.Tensor,
        pooled_descriptors: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> dict[str, torch.Tensor]:
        rel_depth = torch.reciprocal(raw_inverse_depth.clamp(min=self.reciprocal_floor))

        calib_bt = self.metric_calibrator(pooled_descriptors).unflatten(0, (batch_size, num_frames))
        calib_clip = calib_bt.mean(dim=1)
        scale_raw = calib_clip[:, 0]
        shift_raw = calib_clip[:, 1]
        metric_depth = self._apply_metric_affine(rel_depth, scale_raw, shift_raw)

        return {
            "depth": metric_depth,
            "depth_relative": rel_depth,
            "metric_scale": F.softplus(scale_raw).detach(),
            "metric_shift": shift_raw.detach(),
        }


class VideoDepthAnythingModel(nn.Module):
    """Trainable Video Depth Anything model for GT-supervised learning.

    Tiny defaults to a novel DINOv3 dual-head architecture. Other variants use
    the legacy vendored Video-Depth-Anything implementation.
    """

    def __init__(
        self,
        variant: str = "small",
        num_frames: int = 8,
        positional_encoding: str = "ape",
        checkpoint_path: str | None = None,
        strict_checkpoint: bool = False,
        tiny_arch: str = "dual_dinov3",
        tiny_dino_model_name: str = "vit_small_patch16_dinov3",
        tiny_pretrained: bool = True,
        tiny_relative_bounded: bool = True,
        tiny_relative_activation: str = "softplus",
        tiny_metric_activation: str = "softplus",
        tiny_out_indices: Iterable[int] | None = None,
        legacy_metric_bridge: str = "auto",
        legacy_reciprocal_floor: float = 1e-2,
        legacy_metric_activation: str = "softplus",
    ) -> None:
        super().__init__()

        key = variant.lower()
        if key not in _VDA_VARIANTS:
            valid = ", ".join(sorted(_VDA_VARIANTS.keys()))
            raise ValueError(f"Unknown VDA variant '{variant}'. Available: {valid}")

        self.variant = key
        use_dual_tiny = key in ("tiny", "vittiny") and str(tiny_arch).lower() != "legacy"

        self.metric_bridge_mode = "none"

        if use_dual_tiny:
            self.model = TinyDualHeadDINOv3(
                num_frames=num_frames,
                positional_encoding=positional_encoding,
                dino_model_name=tiny_dino_model_name,
                pretrained=bool(tiny_pretrained),
                out_indices=tuple(tiny_out_indices) if tiny_out_indices is not None else (2, 5, 8, 11),
                features=64,
                out_channels=(64, 128, 256, 384),
                relative_bounded=bool(tiny_relative_bounded),
                relative_activation=tiny_relative_activation,
                metric_activation=tiny_metric_activation,
            )
            self.patch_size = self.model.patch_size
            self._legacy = False
        else:
            cfg = _VDA_VARIANTS[key]
            self.model = VideoDepthAnything(
                encoder=cfg["encoder"],
                features=cfg["features"],
                out_channels=cfg["out_channels"],
                use_bn=False,
                use_clstoken=False,
                num_frames=num_frames,
                pe=positional_encoding,
                metric=True,
            )
            self.patch_size = 14
            self._legacy = True
            self.metric_bridge_mode = self._resolve_legacy_metric_bridge_mode(
                bridge_mode=legacy_metric_bridge,
                checkpoint_path=checkpoint_path,
            )
            if self.metric_bridge_mode == "inverse_affine":
                self.metric_calibrator = LegacyInverseAffineBridge(
                    embed_dim=int(self.model.pretrained.embed_dim),
                    reciprocal_floor=float(legacy_reciprocal_floor),
                    metric_activation=legacy_metric_activation,
                )
            else:
                self.metric_calibrator = None

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

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, strict=strict_checkpoint)

    @staticmethod
    def _resolve_legacy_metric_bridge_mode(
        bridge_mode: str,
        checkpoint_path: str | None,
    ) -> str:
        mode = str(bridge_mode or "auto").strip().lower()
        if mode in ("none", "off", "false", "disabled"):
            return "none"
        if mode in ("inverse_affine", "on", "true", "enabled"):
            return "inverse_affine"
        if mode != "auto":
            raise ValueError(
                "Unsupported legacy_metric_bridge mode. "
                "Expected one of: auto, inverse_affine, none."
            )

        if checkpoint_path:
            ckpt_name = Path(checkpoint_path).name.lower()
            if "metric" in ckpt_name:
                return "none"
            if "video_depth_anything" in ckpt_name:
                return "inverse_affine"
        return "none"

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        if hasattr(self.model, "load_checkpoint"):
            self.model.load_checkpoint(checkpoint_path, strict=strict)
            return

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        cleaned_state = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            key = key.removeprefix("module.")
            key = key.removeprefix("model.")
            cleaned_state[key] = value

        current_state = self.model.state_dict()
        compatible_state = {}
        skipped_shape = []
        for key, value in cleaned_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            elif key in current_state:
                skipped_shape.append(key)

        missing_keys, unexpected_keys = self.model.load_state_dict(compatible_state, strict=False)
        if strict and (missing_keys or unexpected_keys or skipped_shape):
            raise RuntimeError(
                "Strict checkpoint loading failed for VideoDepthAnythingModel. "
                f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
                f"shape_mismatch={len(skipped_shape)}"
            )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(frames.shape)}")

        if not self._legacy:
            return self.model(frames)

        _, channels, _, height, width = frames.shape
        if channels != 3:
            raise ValueError(f"VideoDepthAnythingModel expects 3 RGB channels, got {channels}")
        if (height % self.patch_size) != 0 or (width % self.patch_size) != 0:
            raise ValueError(
                f"VideoDepthAnythingModel requires H/W divisible by {self.patch_size}, "
                f"got H={height}, W={width}."
            )

        x = frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        mean = self._mean.to(dtype=x.dtype, device=x.device)
        std = self._std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) / std

        features = self.model.pretrained.get_intermediate_layers(
            x.flatten(0, 1),
            self.model.intermediate_layer_idx[self.model.encoder],
            return_class_token=True,
        )
        raw_depth_bt = self.model.head(features, height // self.patch_size, width // self.patch_size, frames.shape[2])[0]
        raw_depth_bt = F.interpolate(raw_depth_bt, size=(height, width), mode="bilinear", align_corners=True)
        raw_depth_bt = F.relu(raw_depth_bt)
        raw_depth = raw_depth_bt.squeeze(1).unflatten(0, (frames.shape[0], frames.shape[2])).unsqueeze(1)

        if self.metric_bridge_mode != "inverse_affine" or self.metric_calibrator is None:
            return {"depth": raw_depth}

        patch_tokens, cls_token = features[-1]
        pooled = cls_token if cls_token is not None else patch_tokens.mean(dim=1)
        outputs = self.metric_calibrator(
            raw_inverse_depth=raw_depth,
            pooled_descriptors=pooled,
            batch_size=frames.shape[0],
            num_frames=frames.shape[2],
        )
        outputs["depth_inverse_relative"] = raw_depth
        return outputs
