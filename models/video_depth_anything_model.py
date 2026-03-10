"""Metric-only Video Depth Anything wrapper with causal streaming support."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.teachers.vendor.video_depth_anything import VideoDepthAnything


_VDA_VARIANTS: Dict[str, Dict[str, Any]] = {
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
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
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

_STREAM_MODES = {"offline", "streaming_emulated"}
_NUM_TEMPORAL_STAGES = 4


class FastClassicalPrefilter(nn.Module):
    """Very cheap denoise+deblur front-end with learnable scalar strengths.

    The filter uses a fixed Gaussian blur for denoising and an unsharp-mask
    refinement for mild deblurring. Only the blend strengths are learned.
    """

    def __init__(
        self,
        *,
        kernel_size: int = 5,
        sigma: float = 1.0,
        denoise_init: float = 0.20,
        sharpen_init: float = 0.10,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"prefilter kernel_size must be a positive odd integer, got {kernel_size}")

        self.kernel_size = int(kernel_size)
        self.sigma = float(max(sigma, 1e-3))
        self.learnable = bool(learnable)
        self.register_buffer("kernel", self._build_gaussian_kernel(self.kernel_size, self.sigma), persistent=False)

        denoise_logit = self._to_logit(denoise_init)
        sharpen_logit = self._to_logit(sharpen_init)
        if self.learnable:
            self.denoise_logit = nn.Parameter(torch.tensor(denoise_logit, dtype=torch.float32))
            self.sharpen_logit = nn.Parameter(torch.tensor(sharpen_logit, dtype=torch.float32))
        else:
            self.register_buffer("denoise_logit", torch.tensor(denoise_logit, dtype=torch.float32), persistent=False)
            self.register_buffer("sharpen_logit", torch.tensor(sharpen_logit, dtype=torch.float32), persistent=False)

    @staticmethod
    def _to_logit(value: float) -> float:
        clipped = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return float(math.log(clipped / (1.0 - clipped)))

    @staticmethod
    def _build_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-8)
        return kernel_2d.view(1, 1, kernel_size, kernel_size)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        kernel = self.kernel.to(device=x.device, dtype=x.dtype).expand(x.shape[1], 1, -1, -1)
        return F.conv2d(x, kernel, groups=x.shape[1])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = frames.shape
        input_dtype = frames.dtype
        x = frames.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        # Keep the cheap classical prefilter numerically stable and dtype-consistent
        # under bf16 autocast, then restore the caller's dtype at the end.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            blurred = self._blur(x).to(dtype=x.dtype)

            denoise_strength = torch.sigmoid(self.denoise_logit).to(device=x.device, dtype=x.dtype).view(1, 1, 1, 1)
            denoised = torch.lerp(x, blurred, denoise_strength)

            lowpass = self._blur(denoised).to(dtype=x.dtype)
            sharpen_strength = torch.sigmoid(self.sharpen_logit).to(device=x.device, dtype=x.dtype).view(1, 1, 1, 1)
            sharpened = denoised + sharpen_strength * (denoised - lowpass)
            filtered = sharpened.clamp_(0.0, 1.0)

        filtered = filtered.to(dtype=input_dtype)
        return filtered.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


class ChannelStatsAlignPrefilter(nn.Module):
    """Per-frame channel-stat alignment toward fixed or learned target stats."""

    def __init__(
        self,
        *,
        target_mean: Sequence[float] = (0.485, 0.456, 0.406),
        target_std: Sequence[float] = (0.229, 0.224, 0.225),
        learnable: bool = False,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if len(target_mean) != 3 or len(target_std) != 3:
            raise ValueError("target_mean/target_std must have length 3 for RGB input")
        self.eps = float(eps)
        mean_tensor = torch.tensor(target_mean, dtype=torch.float32).view(1, 3, 1, 1)
        log_std_tensor = torch.log(torch.tensor(target_std, dtype=torch.float32).clamp_min(self.eps)).view(1, 3, 1, 1)
        if learnable:
            self.target_mean = nn.Parameter(mean_tensor)
            self.target_log_std = nn.Parameter(log_std_tensor)
        else:
            self.register_buffer("target_mean", mean_tensor, persistent=False)
            self.register_buffer("target_log_std", log_std_tensor, persistent=False)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        frame_mean = x.mean(dim=(2, 3), keepdim=True)
        frame_std = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.eps)
        target_mean = self.target_mean.to(device=x.device, dtype=x.dtype)
        target_std = self.target_log_std.to(device=x.device, dtype=x.dtype).exp().clamp_min(self.eps)
        aligned = (x - frame_mean) / frame_std * target_std + target_mean
        aligned = aligned.clamp_(0.0, 1.0)
        return aligned.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


class TinyAffineNormalizer(nn.Module):
    """Tiny per-frame affine normalizer predicted from shallow depthwise features."""

    def __init__(self, *, max_scale_delta: float = 0.25, max_shift: float = 0.10) -> None:
        super().__init__()
        self.max_scale_delta = float(max_scale_delta)
        self.max_shift = float(max_shift)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=True),
            nn.GELU(),
            nn.Conv2d(3, 6, kernel_size=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
        )
        nn.init.zeros_(self.encoder[0].weight)
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.zeros_(self.encoder[2].weight)
        nn.init.zeros_(self.encoder[2].bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        stats = self.encoder(x).flatten(1)
        scale_delta, shift = stats[:, :channels], stats[:, channels:]
        scale = 1.0 + self.max_scale_delta * torch.tanh(scale_delta).view(-1, channels, 1, 1)
        shift = self.max_shift * torch.tanh(shift).view(-1, channels, 1, 1)
        normalized = (x * scale + shift).clamp_(0.0, 1.0)
        return normalized.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


class DepthwiseResidualNormalizer(nn.Module):
    """Single-layer depthwise residual normalizer with tiny overhead."""

    def __init__(self, *, residual_scale_init: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        nn.init.zeros_(self.depthwise.weight)
        self.residual_logit = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        residual = self.depthwise(x)
        scale = 0.25 * torch.tanh(self.residual_logit).to(device=x.device, dtype=x.dtype)
        normalized = (x + scale * residual).clamp_(0.0, 1.0)
        return normalized.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


def _clean_checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        key = key.removeprefix("module.")
        key = key.removeprefix("model.")
        cleaned[key] = value
    return cleaned


class VideoDepthAnythingModel(nn.Module):
    """Metric-only VDA wrapper with clip and online streaming entrypoints."""

    def __init__(
        self,
        variant: str = "small",
        num_frames: int = 32,
        positional_encoding: str = "ape",
        checkpoint_path: str | None = None,
        strict_checkpoint: bool = False,
        mode: str = "offline",
        stream_max_cache_len: int | None = None,
        stream_reset_interval: int = 0,
        state_gate_enabled: bool = False,
        state_gate_reduction: int = 8,
        state_gate_stage_mask: Sequence[bool] | None = None,
        prefilter_enabled: bool = False,
        prefilter_type: str = "fast_classical",
        prefilter_target_mean: Sequence[float] | None = None,
        prefilter_target_std: Sequence[float] | None = None,
        prefilter_kernel_size: int = 5,
        prefilter_sigma: float = 1.0,
        prefilter_denoise_init: float = 0.20,
        prefilter_sharpen_init: float = 0.10,
        prefilter_learnable: bool = True,
    ) -> None:
        super().__init__()

        key = str(variant).lower()
        if key not in _VDA_VARIANTS:
            valid = ", ".join(sorted(_VDA_VARIANTS))
            raise ValueError(f"Unknown VDA variant '{variant}'. Available: {valid}")
        if mode not in _STREAM_MODES:
            valid = ", ".join(sorted(_STREAM_MODES))
            raise ValueError(f"Unsupported VideoDepthAnything mode '{mode}'. Use one of: {valid}")

        cfg = _VDA_VARIANTS[key]
        self.variant = key
        self.metric_bridge_mode = "none"
        self.patch_size = 14
        self.max_temporal_length = int(num_frames)
        self.default_mode = mode
        self.stream_reset_interval = max(0, int(stream_reset_interval))
        cache_limit = self.max_temporal_length if stream_max_cache_len is None else int(stream_max_cache_len)
        self.stream_max_cache_len = max(1, cache_limit)
        self.state_gate_enabled = bool(state_gate_enabled)
        self.prefilter_enabled = bool(prefilter_enabled)
        self.prefilter_type = str(prefilter_type).lower()

        self.model = VideoDepthAnything(
            encoder=cfg["encoder"],
            features=cfg["features"],
            out_channels=cfg["out_channels"],
            use_bn=False,
            use_clstoken=False,
            num_frames=self.max_temporal_length,
            pe=positional_encoding,
            metric=True,
            state_gate_enabled=self.state_gate_enabled,
            state_gate_reduction=int(state_gate_reduction),
            state_gate_stage_mask=state_gate_stage_mask,
        )
        self.prefilter = self._build_prefilter(
            enabled=self.prefilter_enabled,
            prefilter_type=self.prefilter_type,
            target_mean=prefilter_target_mean,
            target_std=prefilter_target_std,
            kernel_size=int(prefilter_kernel_size),
            sigma=float(prefilter_sigma),
            denoise_init=float(prefilter_denoise_init),
            sharpen_init=float(prefilter_sharpen_init),
            learnable=bool(prefilter_learnable),
        )

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

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cleaned_state = _clean_checkpoint_state_dict(checkpoint)

        current_state = self.model.state_dict()
        compatible_state: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
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

    def reset_stream_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        return {
            "step": 0,
            "batch_size": int(batch_size),
            "device": None if device is None else str(device),
            "stage_caches": [[] for _ in range(_NUM_TEMPORAL_STAGES)],
        }

    def stream_step(
        self,
        frame_t: torch.Tensor,
        state: dict[str, Any] | None,
        return_gate_stats: bool = False,
        return_feature_summary: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        if frame_t.ndim == 4:
            clip = frame_t.unsqueeze(2)
        elif frame_t.ndim == 5 and frame_t.shape[2] == 1:
            clip = frame_t
        else:
            raise ValueError(
                "stream_step expects a single frame with shape (B, C, H, W) "
                f"or (B, C, 1, H, W), got {tuple(frame_t.shape)}"
            )

        batch_size = clip.shape[0]
        if state is None:
            state = self.reset_stream_state(batch_size=batch_size, device=clip.device)

        state = self._maybe_reset_stream_state(state, clip.device)
        head_outputs = self._run_vda_head(
            clip,
            cached_hidden_state_list=self._flatten_stage_caches(state["stage_caches"]),
            return_gate_stats=return_gate_stats,
            return_feature_summary=return_feature_summary,
        )

        next_state = {
            "step": int(state.get("step", 0)) + 1,
            "batch_size": batch_size,
            "device": str(clip.device),
            "stage_caches": self._update_stage_caches(
                previous_stage_caches=state["stage_caches"],
                new_hidden_state_list=head_outputs["hidden_state_list"],
            ),
        }

        outputs: dict[str, torch.Tensor | dict[str, Any]] = {
            "depth": head_outputs["depth"],
            "stream_state": next_state,
        }
        if "feature_summary" in head_outputs:
            outputs["feature_summary"] = head_outputs["feature_summary"]
        if "gate_stats" in head_outputs:
            outputs["gate_stats"] = head_outputs["gate_stats"]
        if "feature_summary" in head_outputs:
            outputs["feature_summary"] = head_outputs["feature_summary"]
        return outputs

    def forward(
        self,
        frames: torch.Tensor,
        mode: str | None = None,
        return_state: bool = False,
        return_gate_stats: bool = False,
        return_feature_summary: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        active_mode = self.default_mode if mode is None else str(mode)
        if active_mode not in _STREAM_MODES:
            valid = ", ".join(sorted(_STREAM_MODES))
            raise ValueError(f"Unsupported forward mode '{active_mode}'. Use one of: {valid}")

        if active_mode == "offline":
            outputs = self._run_vda_head(
                frames,
                cached_hidden_state_list=None,
                return_gate_stats=False,
                return_feature_summary=return_feature_summary,
            )
            result: dict[str, torch.Tensor | dict[str, Any]] = {"depth": outputs["depth"]}
            if "feature_summary" in outputs:
                result["feature_summary"] = outputs["feature_summary"]
            if return_state:
                result["stream_state"] = self.reset_stream_state(frames.shape[0], frames.device)
            return result

        batch_size, _, num_frames, _, _ = self._validate_frames(frames)
        state = self.reset_stream_state(batch_size=batch_size, device=frames.device)
        depth_steps: list[torch.Tensor] = []
        gate_steps: list[torch.Tensor] = []
        feature_steps: list[torch.Tensor] = []
        collect_gate_stats = bool(return_gate_stats and self.state_gate_enabled)

        for frame_idx in range(num_frames):
            step_outputs = self.stream_step(
                frames[:, :, frame_idx],
                state,
                return_gate_stats=collect_gate_stats,
                return_feature_summary=return_feature_summary,
            )
            depth_steps.append(step_outputs["depth"])
            if "feature_summary" in step_outputs:
                feature_steps.append(step_outputs["feature_summary"])
            if "gate_stats" in step_outputs:
                gate_steps.append(step_outputs["gate_stats"])
            state = step_outputs["stream_state"]

        outputs = {"depth": torch.cat(depth_steps, dim=2)}
        if return_feature_summary and feature_steps:
            outputs["feature_summary"] = torch.cat(feature_steps, dim=1)
        if gate_steps:
            outputs["gate_stats"] = torch.stack(gate_steps, dim=1)
        if return_state:
            outputs["stream_state"] = state
        return outputs

    def _validate_frames(self, frames: torch.Tensor) -> tuple[int, int, int, int, int]:
        if frames.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(frames.shape)}")
        batch_size, channels, num_frames, height, width = frames.shape
        if channels != 3:
            raise ValueError(f"VideoDepthAnythingModel expects 3 RGB channels, got {channels}")
        return batch_size, channels, num_frames, height, width

    def _prepare_frames(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int, int, int, int]:
        batch_size, _, num_frames, height, width = self._validate_frames(frames)
        if self.prefilter is not None:
            frames = self.prefilter(frames)
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            frames = F.pad(frames, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")

        _, _, _, padded_h, padded_w = frames.shape
        x = frames.permute(0, 2, 1, 3, 4)
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return x, batch_size, num_frames, height, width, padded_h, padded_w

    def _run_vda_head(
        self,
        frames: torch.Tensor,
        *,
        cached_hidden_state_list: Sequence[torch.Tensor] | None,
        return_gate_stats: bool,
        return_feature_summary: bool,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        x, batch_size, num_frames, height, width, padded_h, padded_w = self._prepare_frames(frames)
        patch_h = padded_h // self.patch_size
        patch_w = padded_w // self.patch_size

        features = self.model.pretrained.get_intermediate_layers(
            x.flatten(0, 1),
            self.model.intermediate_layer_idx[self.model.encoder],
            return_class_token=True,
        )
        feature_summary = None
        if return_feature_summary and features:
            last_tokens = features[-1][0] if isinstance(features[-1], (tuple, list)) else features[-1]
            feature_summary = last_tokens.mean(dim=1).unflatten(0, (batch_size, num_frames)).contiguous()

        head_outputs = self.model.head(
            features,
            patch_h,
            patch_w,
            num_frames,
            cached_hidden_state_list=cached_hidden_state_list,
            return_gate_stats=return_gate_stats,
        )
        depth_bt = head_outputs[0]
        hidden_state_list = head_outputs[1]
        gate_stats = head_outputs[2] if len(head_outputs) > 2 else None

        depth_bt = F.interpolate(depth_bt, size=(padded_h, padded_w), mode="bilinear", align_corners=True)
        depth_bt = F.relu(depth_bt)
        depth_bt = depth_bt[:, :, :height, :width]
        depth = depth_bt.unflatten(0, (batch_size, num_frames)).permute(0, 2, 1, 3, 4).contiguous()

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "depth": depth,
            "hidden_state_list": hidden_state_list,
        }
        if feature_summary is not None:
            outputs["feature_summary"] = feature_summary
        if gate_stats is not None:
            outputs["gate_stats"] = gate_stats
        return outputs

    def _maybe_reset_stream_state(
        self,
        state: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        step = int(state.get("step", 0))
        if self.stream_reset_interval <= 0 or step == 0 or (step % self.stream_reset_interval) != 0:
            return state
        return self.reset_stream_state(batch_size=int(state["batch_size"]), device=device)

    def _flatten_stage_caches(
        self,
        stage_caches: Sequence[Sequence[torch.Tensor]],
    ) -> list[torch.Tensor] | None:
        flat: list[torch.Tensor] = []
        for stage in stage_caches:
            flat.extend(stage)
        return flat if flat else None

    def _update_stage_caches(
        self,
        *,
        previous_stage_caches: Sequence[Sequence[torch.Tensor]],
        new_hidden_state_list: Sequence[torch.Tensor],
    ) -> list[list[torch.Tensor]]:
        if len(new_hidden_state_list) == 0:
            return [list(stage) for stage in previous_stage_caches]
        if len(new_hidden_state_list) % _NUM_TEMPORAL_STAGES != 0:
            raise ValueError(
                "Temporal head returned a hidden-state list that cannot be split into "
                f"{_NUM_TEMPORAL_STAGES} stages: len={len(new_hidden_state_list)}"
            )

        tensors_per_stage = len(new_hidden_state_list) // _NUM_TEMPORAL_STAGES
        updated: list[list[torch.Tensor]] = []
        for stage_idx in range(_NUM_TEMPORAL_STAGES):
            start = stage_idx * tensors_per_stage
            end = (stage_idx + 1) * tensors_per_stage
            prev_stage = list(previous_stage_caches[stage_idx]) if stage_idx < len(previous_stage_caches) else []
            new_stage = [tensor.detach() for tensor in new_hidden_state_list[start:end]]
            if not prev_stage:
                stage_cache = [self._truncate_cache_tensor(tensor) for tensor in new_stage]
            else:
                if len(prev_stage) != len(new_stage):
                    raise ValueError(
                        f"Streaming cache tensor count changed within stage {stage_idx}: "
                        f"prev={len(prev_stage)} new={len(new_stage)}"
                    )
                stage_cache = []
                for prev_tensor, new_tensor in zip(prev_stage, new_stage):
                    merged = torch.cat([prev_tensor, new_tensor], dim=1)
                    stage_cache.append(self._truncate_cache_tensor(merged.detach()))
            updated.append(stage_cache)
        return updated

    def _truncate_cache_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] <= self.stream_max_cache_len:
            return tensor
        return tensor[:, -self.stream_max_cache_len :, :].contiguous()

    def _build_prefilter(
        self,
        *,
        enabled: bool,
        prefilter_type: str,
        target_mean: Sequence[float] | None,
        target_std: Sequence[float] | None,
        kernel_size: int,
        sigma: float,
        denoise_init: float,
        sharpen_init: float,
        learnable: bool,
    ) -> nn.Module | None:
        if not enabled:
            return None
        if prefilter_type == "fast_classical":
            return FastClassicalPrefilter(
                kernel_size=kernel_size,
                sigma=sigma,
                denoise_init=denoise_init,
                sharpen_init=sharpen_init,
                learnable=learnable,
            )
        if prefilter_type == "stats_align":
            return ChannelStatsAlignPrefilter(
                target_mean=(0.485, 0.456, 0.406) if target_mean is None else target_mean,
                target_std=(0.229, 0.224, 0.225) if target_std is None else target_std,
                learnable=learnable,
            )
        if prefilter_type == "learned_affine":
            return TinyAffineNormalizer()
        if prefilter_type == "depthwise":
            return DepthwiseResidualNormalizer()
        valid = ", ".join(("fast_classical", "stats_align", "learned_affine", "depthwise"))
        raise ValueError(f"Unknown prefilter_type={prefilter_type!r}. Valid: {valid}")
