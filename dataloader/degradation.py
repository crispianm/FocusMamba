from __future__ import annotations

"""
Fast low-light video degradation for training.

This module replaces the original list-based pipeline with a tensor-first API,
linear-light processing, a vectorized per-frame anisotropic blur, and a more
physically plausible noise model.

Core assumptions
----------------
- Inputs are RGB tensors with shape (T, C, H, W) or (B, T, C, H, W).
- Inputs may be uint8 in [0, 255] or floating point in [0, 1] / [0, 255].
- The degradation operates in linear RGB.
- Noise is applied in this order:
    1) exposure scaling
    2) optical blur
    3) shot noise
    4) read noise
    5) banding / fixed-pattern readout noise
    6) quantization
    7) conversion back to sRGB

Public API
----------
- LowLightDegrader: reusable callable class for training/data pipelines
- LowLightDegradation: legacy-compatible wrapper backed by the new pipeline
- degrade_video: one-shot helper
- NpySequenceParamSource: optional loader for legacy (1, T, 9) .npy param files
- build_degradation: config helper that preserves the old entrypoint

Typical usage patterns
----------------------
1. Fast random clip degradation for training:
    degrader = LowLightDegrader(device="cuda")
    y = degrader(x)
2. Dataset-guided temporal parameters from legacy .npy files:
    source = NpySequenceParamSource("noise_params", LowLightConfig())
    y = degrade_video(x, param_source=source)
3. Fully explicit temporal schedules for ablations or visualization:
    params = LowLightParams(...)
    y = degrade_video(x, params=params)

Temporal behavior
-----------------
- RandomParamSampler samples one value per clip for each stochastic control and
  broadcasts it across all frames. This keeps the default training path free of
  artificial frame-to-frame flicker.
- NpySequenceParamSource preserves time-varying parameters from legacy files.
- Passing LowLightParams directly gives full per-frame control.

Example
-------
    x = torch.randint(0, 256, (8, 3, 256, 256), dtype=torch.uint8)
    degrader = LowLightDegrader(device="cuda")
    y = degrader(x)  # same shape, dtype matches input by default
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Configuration and parameter containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LowLightConfig:
    """Sampling ranges and numerical settings.

    Notes
    -----
    - exposure_ev is negative for darkening.
    - shot_scale controls the Poisson variance using:
          y ~ Poisson(x * 255 / shot_scale) * shot_scale / 255
      so larger values are noisier.
    - quant_step is the ADC step size in normalized linear-RGB units.
        - The config only defines ranges and processing options; the actual temporal
            behavior depends on which parameter source is used.
    """

    # Random sampling ranges used by RandomParamSampler.
    exposure_ev_range: tuple[float, float] = (-3.5, 0.0)
    shot_scale_range: tuple[float, float] = (0.08, 12.0)
    read_noise_range: tuple[float, float] = (0.0, 0.03)
    quant_step_range: tuple[float, float] = (0.0, 1.0 / 255.0)
    band_noise_range: tuple[float, float] = (0.0, 0.02)
    blur_sigma1_range: tuple[float, float] = (0.35, 2.25)
    blur_sigma2_range: tuple[float, float] = (0.35, 2.25)
    blur_angle_range: tuple[float, float] = (0.0, math.pi / 2.0)

    # Blur kernel sizing.
    blur_truncate: float = 3.0
    max_blur_kernel_size: int = 25
    min_blur_sigma_for_convolution: float = 0.05

    # Processing behavior.
    quantization_mode: str = "round"  # "round" or "noise"
    shot_noise_mode: str = "gaussian"  # "gaussian" or "poisson"
    use_channels_last_for_blur: bool = True
    fixed_pattern_per_clip: bool = True
    clamp_linear_before_srgb: bool = True

    # Legacy normalized label interpretation for (1, T, 9) files.
    legacy_exposure_ev_range: tuple[float, float] = (0.0, -3.5)
    legacy_shot_log_range: tuple[float, float] = (math.log(1e-1), math.log(30.0))
    legacy_read_noise_range: tuple[float, float] = (0.0, 0.1)
    legacy_quant_noise_range: tuple[float, float] = (0.0, 0.1)
    legacy_band_noise_range: tuple[float, float] = (0.0, 0.03)
    legacy_blur_sigma1_range: tuple[float, float] = (0.5, 10.0)
    legacy_blur_sigma2_range: tuple[float, float] = (0.5, 10.0)
    legacy_blur_angle_frac_range: tuple[float, float] = (0.0, 0.25)  # fraction of pi

    def validate(self) -> None:
        if self.quantization_mode not in {"round", "noise"}:
            raise ValueError(f"quantization_mode must be 'round' or 'noise', got {self.quantization_mode!r}")
        if self.shot_noise_mode not in {"gaussian", "poisson"}:
            raise ValueError(f"shot_noise_mode must be 'gaussian' or 'poisson', got {self.shot_noise_mode!r}")
        if self.max_blur_kernel_size < 1 or self.max_blur_kernel_size % 2 == 0:
            raise ValueError("max_blur_kernel_size must be a positive odd integer")
        if self.blur_truncate <= 0:
            raise ValueError("blur_truncate must be > 0")


@dataclass
class LowLightParams:
    """Per-frame degradation parameters.

    All tensors use shape (B, T).
    Angles are in radians, with 0 = horizontal major axis and pi/2 = vertical.
    band_direction is rounded to {0,1}; 0 = horizontal stripes, 1 = vertical.

    This container is the canonical representation used internally by
    degrade_video(). Construct it directly when you want deterministic or
    hand-designed temporal schedules.
    """

    exposure_ev: Tensor
    shot_scale: Tensor
    read_noise: Tensor
    quant_step: Tensor
    band_noise: Tensor
    band_direction: Tensor
    blur_sigma1: Tensor
    blur_sigma2: Tensor
    blur_angle: Tensor

    def to(self, device: Optional[torch.device | str] = None, dtype: Optional[torch.dtype] = None) -> "LowLightParams":
        """Return a copy moved to a new device and/or dtype."""
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return LowLightParams(
            exposure_ev=self.exposure_ev.to(**kwargs),
            shot_scale=self.shot_scale.to(**kwargs),
            read_noise=self.read_noise.to(**kwargs),
            quant_step=self.quant_step.to(**kwargs),
            band_noise=self.band_noise.to(**kwargs),
            band_direction=self.band_direction.to(**kwargs),
            blur_sigma1=self.blur_sigma1.to(**kwargs),
            blur_sigma2=self.blur_sigma2.to(**kwargs),
            blur_angle=self.blur_angle.to(**kwargs),
        )

    def broadcast_batch(self, batch_size: int) -> "LowLightParams":
        """Broadcast a single-clip parameter set across a larger batch.

        This is mainly used so callers can pass parameters shaped (1, T) for a
        video batch shaped (B, T, C, H, W).
        """
        current = self.exposure_ev.shape[0]
        if current == batch_size:
            return self
        if current != 1:
            raise ValueError(f"Cannot broadcast params with batch={current} to batch={batch_size}")
        return LowLightParams(
            exposure_ev=self.exposure_ev.expand(batch_size, -1),
            shot_scale=self.shot_scale.expand(batch_size, -1),
            read_noise=self.read_noise.expand(batch_size, -1),
            quant_step=self.quant_step.expand(batch_size, -1),
            band_noise=self.band_noise.expand(batch_size, -1),
            band_direction=self.band_direction.expand(batch_size, -1),
            blur_sigma1=self.blur_sigma1.expand(batch_size, -1),
            blur_sigma2=self.blur_sigma2.expand(batch_size, -1),
            blur_angle=self.blur_angle.expand(batch_size, -1),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.exposure_ev.shape)  # type: ignore[return-value]

    def validate(self, batch_size: int, num_frames: int) -> None:
        """Ensure every field has the expected (B, T) layout."""
        names = (
            "exposure_ev",
            "shot_scale",
            "read_noise",
            "quant_step",
            "band_noise",
            "band_direction",
            "blur_sigma1",
            "blur_sigma2",
            "blur_angle",
        )
        for name in names:
            tensor = getattr(self, name)
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape (B, T), got {tuple(tensor.shape)}")
            if tensor.shape != (batch_size, num_frames):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {num_frames}), got {tuple(tensor.shape)}"
                )

    def as_dict(self) -> dict[str, Tensor]:
        """Expose the parameter tensors as a plain dictionary."""
        return {
            "exposure_ev": self.exposure_ev,
            "shot_scale": self.shot_scale,
            "read_noise": self.read_noise,
            "quant_step": self.quant_step,
            "band_noise": self.band_noise,
            "band_direction": self.band_direction,
            "blur_sigma1": self.blur_sigma1,
            "blur_sigma2": self.blur_sigma2,
            "blur_angle": self.blur_angle,
        }

    @classmethod
    def from_legacy_normalized(cls, x: Tensor, config: LowLightConfig) -> "LowLightParams":
        """Convert legacy normalized labels of shape (B, T, 9) into parameters.

        Legacy channel order:
            0 exposure_value
            1 shot_noise_log
            2 read_noise
            3 quant_noise
            4 band_noise
            5 band_noise_angle    (0=horiz, 1=vert after rounding)
            6 blur_sigma1
            7 blur_sigma2
            8 blur_angle          (fraction of pi; original used 0..0.25*pi)
        """
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError(f"Expected legacy params with shape (B, T, 9), got {tuple(x.shape)}")
        x = x.float().clamp(0.0, 1.0)

        def lerp(u: Tensor, lo: float, hi: float) -> Tensor:
            return lo + (hi - lo) * u

        exposure_ev = lerp(x[..., 0], *config.legacy_exposure_ev_range)
        shot_log = lerp(x[..., 1], *config.legacy_shot_log_range)
        shot_scale = torch.exp(shot_log)
        read_noise = lerp(x[..., 2], *config.legacy_read_noise_range)
        quant_step = lerp(x[..., 3], *config.legacy_quant_noise_range)
        band_noise = lerp(x[..., 4], *config.legacy_band_noise_range)
        band_direction = torch.round(x[..., 5]).float()
        blur_sigma1 = lerp(x[..., 6], *config.legacy_blur_sigma1_range)
        blur_sigma2 = lerp(x[..., 7], *config.legacy_blur_sigma2_range)
        max_supported_sigma = (config.max_blur_kernel_size // 2) / config.blur_truncate
        blur_sigma1 = blur_sigma1.clamp_max(max_supported_sigma)
        blur_sigma2 = blur_sigma2.clamp_max(max_supported_sigma)
        blur_angle_frac = lerp(x[..., 8], *config.legacy_blur_angle_frac_range)
        blur_angle = blur_angle_frac * math.pi

        return cls(
            exposure_ev=exposure_ev,
            shot_scale=shot_scale,
            read_noise=read_noise,
            quant_step=quant_step,
            band_noise=band_noise,
            band_direction=band_direction,
            blur_sigma1=blur_sigma1,
            blur_sigma2=blur_sigma2,
            blur_angle=blur_angle,
        )


# -----------------------------------------------------------------------------
# Parameter sources
# -----------------------------------------------------------------------------


class RandomParamSampler:
    """Sample clip-consistent parameters from the config ranges.

    The random sampler draws one value per clip for each stochastic control and
    broadcasts it across frames. This avoids frame-to-frame flicker from i.i.d.
    parameter sampling while still allowing temporally varying parameters via
    explicit LowLightParams or legacy .npy sequences.
    """

    def __init__(self, config: LowLightConfig):
        self.config = config

    def sample(
        self,
        batch_size: int,
        num_frames: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> LowLightParams:
        """Sample one stochastic parameter set per clip.

        Each sampled scalar has shape (B, 1) and is expanded to (B, T). The
        result is temporally constant within a clip but still varies across the
        batch.
        """
        cfg = self.config

        def uniform_clip(lo: float, hi: float) -> Tensor:
            return lo + (hi - lo) * torch.rand(
                (batch_size, 1), device=device, dtype=dtype, generator=generator
            ).expand(batch_size, num_frames)

        # Band direction is sampled once per clip and repeated across frames.
        band_direction = torch.randint(
            0,
            2,
            (batch_size, 1),
            device=device,
            generator=generator,
        ).to(dtype).expand(batch_size, num_frames)

        return LowLightParams(
            exposure_ev=uniform_clip(*cfg.exposure_ev_range),
            shot_scale=uniform_clip(*cfg.shot_scale_range).clamp_min(1e-6),
            read_noise=uniform_clip(*cfg.read_noise_range).clamp_min(0.0),
            quant_step=uniform_clip(*cfg.quant_step_range).clamp_min(0.0),
            band_noise=uniform_clip(*cfg.band_noise_range).clamp_min(0.0),
            band_direction=band_direction,
            blur_sigma1=uniform_clip(*cfg.blur_sigma1_range).clamp_min(1e-6),
            blur_sigma2=uniform_clip(*cfg.blur_sigma2_range).clamp_min(1e-6),
            blur_angle=uniform_clip(*cfg.blur_angle_range),
        )


class NpySequenceParamSource:
    """Sample temporally aligned legacy parameters from .npy files.

    Expected file format
    --------------------
    - Each file contains a float array shaped (1, T, 9), normalized to [0, 1].
    - Files inside the same directory are treated as belonging to the same video.
    - Files are ordered by the integer suffix in `_idx_<n>.npy` when present,
      otherwise lexicographically by filename.

    This class indexes the directory tree once at construction time.
    """

    def __init__(self, root: str | Path, config: LowLightConfig):
        self.root = Path(root)
        self.config = config
        if not self.root.exists():
            raise FileNotFoundError(f"Parameter directory does not exist: {self.root}")
        self._groups = self._index(self.root)
        if not self._groups:
            raise FileNotFoundError(f"No .npy files found under {self.root}")

    @staticmethod
    def _parse_idx(path: Path) -> tuple[int, str]:
        match = re.search(r"_idx_(\d+)\.npy$", path.name)
        if match is None:
            return (2**31 - 1, path.name)
        return (int(match.group(1)), path.name)

    @classmethod
    def _index(cls, root: Path) -> dict[str, list[Path]]:
        groups: dict[str, list[Path]] = {}
        for directory, _, files in os_walk_sorted(root):
            npys = [Path(directory) / f for f in files if f.endswith(".npy")]
            if npys:
                groups[str(directory)] = sorted(npys, key=cls._parse_idx)
        return groups

    def _sample_one(self, num_frames: int, *, generator: Optional[torch.Generator] = None) -> Tensor:
        group_keys = list(self._groups.keys())
        group_idx = _randint_python_friendly(len(group_keys), generator)
        paths = self._groups[group_keys[group_idx]]

        arrays: list[Tensor] = []
        total_frames = 0
        for path in paths:
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[2] != 9:
                raise ValueError(f"Expected (1, T, 9), got {arr.shape} in {path}")
            t = torch.from_numpy(arr).float()
            arrays.append(t)
            total_frames += t.shape[1]

        if total_frames < num_frames:
            raise ValueError(
                f"Sequence {group_keys[group_idx]} contains only {total_frames} frames of params; "
                f"requested {num_frames}."
            )

        seq = torch.cat(arrays, dim=1)
        start = _randint_python_friendly(seq.shape[1] - num_frames + 1, generator)
        return seq[:, start : start + num_frames, :]

    def sample(
        self,
        batch_size: int,
        num_frames: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> LowLightParams:
        """Sample a batch of temporally aligned legacy parameter sequences."""
        batch = [self._sample_one(num_frames, generator=generator) for _ in range(batch_size)]
        labels = torch.cat(batch, dim=0).to(device=device, dtype=dtype)
        labels = labels.clamp(0.0, 1.0)
        return LowLightParams.from_legacy_normalized(labels, self.config)


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------


def os_walk_sorted(root: Path):
    """Yield os.walk results with deterministic directory and file ordering."""
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        yield directory, dirs, files


def _randint_python_friendly(high: int, generator: Optional[torch.Generator]) -> int:
    """torch.randint wrapper that returns a Python int and honors a generator."""
    if high <= 0:
        raise ValueError(f"high must be > 0, got {high}")
    device = getattr(generator, "device", None) if generator is not None else None
    return int(torch.randint(0, high, (1,), generator=generator, device=device).item())


# os is only needed by NpySequenceParamSource; keep it near the helpers.
import os  # noqa: E402


def _srgb_to_linear(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(x: Tensor) -> Tensor:
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def _prepare_video(x: Tensor) -> tuple[Tensor, bool, bool]:
    """Normalize user input into a contiguous float tensor in [0, 1].

    Returns
    -------
    video:
        Tensor with shape (B, T, C, H, W).
    squeezed:
        True when the original input was a single clip shaped (T, C, H, W).
    was_integer_like:
        True when the input dtype was non-floating-point, used to decide the
        default output format for output="match_input".
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected a torch.Tensor, got {type(x)!r}")
    if x.ndim == 4:
        x = x.unsqueeze(0)
        squeezed = True
    elif x.ndim == 5:
        squeezed = False
    else:
        raise ValueError(f"Expected (T,C,H,W) or (B,T,C,H,W), got shape {tuple(x.shape)}")

    if x.shape[-3] != 3:
        raise ValueError(f"Expected RGB input with C=3, got {x.shape[-3]}")

    was_integer_like = not torch.is_floating_point(x)
    x = x.float()

    if x.numel() == 0:
        raise ValueError("Input video is empty")

    if x.max() > 1.0 or was_integer_like:
        x = x / 255.0
    x = x.clamp(0.0, 1.0)
    return x.contiguous(), squeezed, was_integer_like


def _restore_output(x: Tensor, *, squeezed: bool, output: str, was_integer_like: bool) -> Tensor:
    """Convert the processed tensor back to the caller's requested output type."""
    if output not in {"float", "uint8", "match_input"}:
        raise ValueError(f"output must be 'float', 'uint8', or 'match_input', got {output!r}")

    if output == "match_input":
        output = "uint8" if was_integer_like else "float"

    if output == "uint8":
        x = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        x = x.clamp(0.0, 1.0)

    if squeezed:
        x = x.squeeze(0)
    return x


def _kernel_size_from_sigma(max_sigma: float, truncate: float, max_kernel_size: int) -> int:
    """Choose an odd kernel size large enough to cover the blur support."""
    if max_sigma <= 0:
        return 1
    radius = max(1, int(math.ceil(max_sigma * truncate)))
    k = 2 * radius + 1
    if k > max_kernel_size:
        k = max_kernel_size
    if k % 2 == 0:
        k -= 1
    return max(k, 1)


def _anisotropic_gaussian_kernels(
    sigma1: Tensor,
    sigma2: Tensor,
    angle: Tensor,
    kernel_size: int,
) -> Tensor:
    """Create one normalized anisotropic Gaussian kernel per element.

    Parameters
    ----------
    sigma1, sigma2, angle: shape (BN,)
    Returns: kernels with shape (BN, 1, K, K)
    """
    device = sigma1.device
    dtype = sigma1.dtype

    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size // 2)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    s1_2 = sigma1.clamp_min(1e-6).square().view(-1, 1, 1)
    s2_2 = sigma2.clamp_min(1e-6).square().view(-1, 1, 1)
    c = torch.cos(angle).view(-1, 1, 1)
    s = torch.sin(angle).view(-1, 1, 1)

    inv11 = c * c / s1_2 + s * s / s2_2
    inv22 = s * s / s1_2 + c * c / s2_2
    inv12 = c * s * (1.0 / s1_2 - 1.0 / s2_2)

    exponent = -0.5 * (inv11 * xx * xx + 2.0 * inv12 * xx * yy + inv22 * yy * yy)
    kernels = torch.exp(exponent)
    kernels = kernels / kernels.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return kernels.unsqueeze(1)


def _apply_per_frame_blur(
    x: Tensor,
    sigma1: Tensor,
    sigma2: Tensor,
    angle: Tensor,
    *,
    truncate: float,
    max_kernel_size: int,
    min_sigma_for_convolution: float,
    use_channels_last: bool,
) -> Tensor:
    """Blur x with a distinct anisotropic Gaussian kernel per frame.

    x: (BN, C, H, W)
    sigma1, sigma2, angle: (BN,)

    All frames in the flattened batch share the same kernel size for efficient
    grouped convolution, but each frame receives its own kernel weights.
    """
    max_sigma = max(float(sigma1.max().item()), float(sigma2.max().item()))
    if max_sigma < min_sigma_for_convolution:
        return x

    kernel_size = _kernel_size_from_sigma(max_sigma, truncate, max_kernel_size)
    if kernel_size == 1:
        return x

    kernels = _anisotropic_gaussian_kernels(sigma1, sigma2, angle, kernel_size)
    bn, c, h, w = x.shape
    pad = kernel_size // 2

    if use_channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x_padded = x_padded.reshape(1, bn * c, h + 2 * pad, w + 2 * pad)
    weight = kernels.repeat_interleave(c, dim=0)  # (BN*C, 1, K, K)
    y = F.conv2d(x_padded, weight, groups=bn * c)
    return y.reshape(bn, c, h, w)


def _apply_shot_noise(
    x: Tensor,
    shot_scale: Tensor,
    mode: str,
    generator: Optional[torch.Generator],
) -> Tensor:
    """Apply signal-dependent shot noise in linear RGB."""
    if mode == "poisson":
        rate = x.clamp_min(0.0) * (255.0 / shot_scale.clamp_min(1e-6))
        return torch.poisson(rate, generator=generator) * (shot_scale / 255.0)
    if mode == "gaussian":
        std = torch.sqrt(x.clamp_min(0.0) * shot_scale.clamp_min(1e-6) / 255.0)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return x + noise * std
    raise ValueError(f"Unknown shot noise mode: {mode!r}")


def _apply_read_noise(x: Tensor, read_noise: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    """Apply additive Gaussian read noise."""
    return x + torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * read_noise


def _apply_banding_noise(
    x: Tensor,
    band_noise: Tensor,
    band_direction: Tensor,
    *,
    fixed_pattern_per_clip: bool,
    generator: Optional[torch.Generator],
) -> Tensor:
    """Add horizontal or vertical readout banding.

    x shape: (B, T, C, H, W)
    band_noise, band_direction: (B, T)
    """
    if float(band_noise.max().item()) <= 0:
        return x

    b, t, c, h, w = x.shape
    device, dtype = x.device, x.dtype
    amp = band_noise.view(b, t, 1, 1, 1)
    direction = band_direction.round().long().clamp(0, 1).view(b, t, 1, 1, 1)

    if fixed_pattern_per_clip:
        horiz = torch.randn((b, 1, c, h, 1), device=device, dtype=dtype, generator=generator).expand(b, t, c, h, w)
        vert = torch.randn((b, 1, c, 1, w), device=device, dtype=dtype, generator=generator).expand(b, t, c, h, w)
    else:
        horiz = torch.randn((b, t, c, h, 1), device=device, dtype=dtype, generator=generator).expand(b, t, c, h, w)
        vert = torch.randn((b, t, c, 1, w), device=device, dtype=dtype, generator=generator).expand(b, t, c, h, w)

    pattern = torch.where(direction == 0, horiz, vert)
    return x + amp * pattern


def _apply_quantization(x: Tensor, quant_step: Tensor, mode: str, generator: Optional[torch.Generator]) -> Tensor:
    """Apply quantization either by deterministic rounding or uniform dither."""
    q = quant_step.clamp_min(0.0)
    if float(q.max().item()) <= 0:
        return x

    if mode == "round":
        q_safe = q.clamp_min(1e-12)
        return torch.where(q > 0, torch.round(x / q_safe) * q_safe, x)
    if mode == "noise":
        return x + (torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator) - 0.5) * q
    raise ValueError(f"Unknown quantization mode: {mode!r}")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


@torch.no_grad()
def degrade_video(
    video: Tensor,
    *,
    params: Optional[LowLightParams] = None,
    param_source: Optional[object] = None,
    config: Optional[LowLightConfig] = None,
    device: Optional[torch.device | str] = None,
    output: str = "match_input",
    return_params: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor | tuple[Tensor, LowLightParams]:
    """Apply low-light degradation to a video tensor.

    Parameters
    ----------
    video:
        RGB tensor shaped (T, C, H, W) or (B, T, C, H, W).
    params:
        Explicit degradation parameters. If omitted, parameters are sampled from
        param_source, or from RandomParamSampler(config) if param_source is None.
    param_source:
        Object with a .sample(batch_size, num_frames, device, dtype, generator)
        method returning LowLightParams. Ignored if params is provided.
    config:
        Configuration. A default config is used when omitted.
    device:
        Processing device. Defaults to video.device.
    output:
        "float", "uint8", or "match_input".
    return_params:
        If True, return (degraded_video, params_used).
    generator:
        Optional torch.Generator for reproducible sampling.

        Returns
        -------
        Tensor or tuple[Tensor, LowLightParams]
                The degraded video, optionally paired with the parameters that were
                used. The output shape matches the input shape; dtype follows the
                output argument.

        Notes
        -----
        - All photometric operations happen in linear RGB, not sRGB.
        - When params is omitted, the default RandomParamSampler produces
            clip-consistent stochastic controls.
        - To intentionally create temporal drift or ramps, pass explicit
            LowLightParams or use NpySequenceParamSource.
    """
    cfg = config or LowLightConfig()
    cfg.validate()

    x, squeezed, was_integer_like = _prepare_video(video)
    device = torch.device(device) if device is not None else x.device
    x = x.to(device=device, dtype=torch.float32, non_blocking=True)

    b, t, c, h, w = x.shape

    if params is None:
        if param_source is None:
            param_source = RandomParamSampler(cfg)
        params = param_source.sample(
            b,
            t,
            device=device,
            dtype=x.dtype,
            generator=generator,
        )
    else:
        params = params.broadcast_batch(b).to(device=device, dtype=x.dtype)

    params.validate(b, t)

    # Convert to linear light, apply exposure, then blur and noise in linear RGB.
    lin = _srgb_to_linear(x)
    exposure = torch.exp2(params.exposure_ev).view(b, t, 1, 1, 1)
    lin = lin * exposure

    flat = lin.reshape(b * t, c, h, w)
    flat = _apply_per_frame_blur(
        flat,
        params.blur_sigma1.reshape(-1),
        params.blur_sigma2.reshape(-1),
        params.blur_angle.reshape(-1),
        truncate=cfg.blur_truncate,
        max_kernel_size=cfg.max_blur_kernel_size,
        min_sigma_for_convolution=cfg.min_blur_sigma_for_convolution,
        use_channels_last=cfg.use_channels_last_for_blur,
    )

    shot_scale = params.shot_scale.view(b * t, 1, 1, 1)
    flat = _apply_shot_noise(flat, shot_scale, cfg.shot_noise_mode, generator)

    read_noise = params.read_noise.view(b * t, 1, 1, 1)
    flat = _apply_read_noise(flat, read_noise, generator)

    lin = flat.reshape(b, t, c, h, w)
    lin = _apply_banding_noise(
        lin,
        params.band_noise,
        params.band_direction,
        fixed_pattern_per_clip=cfg.fixed_pattern_per_clip,
        generator=generator,
    )

    quant = params.quant_step.view(b, t, 1, 1, 1)
    lin = _apply_quantization(lin, quant, cfg.quantization_mode, generator)

    if cfg.clamp_linear_before_srgb:
        lin = lin.clamp(0.0, 1.0)
    srgb = _linear_to_srgb(lin).clamp(0.0, 1.0)
    out = _restore_output(srgb, squeezed=squeezed, output=output, was_integer_like=was_integer_like)

    if return_params:
        return out, params
    return out


class LowLightDegrader:
    """Reusable callable wrapper around degrade_video.

    This class stores a configuration, parameter source, and default device so
    training code can reuse the same degradation policy across batches without
    repeatedly passing keyword arguments.

    Example
    -------
        degrader = LowLightDegrader(device="cuda")
        y = degrader(x)  # x: (T,C,H,W) or (B,T,C,H,W)
    """

    def __init__(
        self,
        *,
        config: Optional[LowLightConfig] = None,
        param_source: Optional[object] = None,
        device: Optional[torch.device | str] = None,
        output: str = "match_input",
    ):
        self.config = config or LowLightConfig()
        self.config.validate()
        self.param_source = param_source
        self.device = device
        self.output = output

    def __call__(
        self,
        video: Tensor,
        *,
        params: Optional[LowLightParams] = None,
        return_params: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor | tuple[Tensor, LowLightParams]:
        """Degrade one clip or batch of clips using the stored defaults."""
        return degrade_video(
            video,
            params=params,
            param_source=self.param_source,
            config=self.config,
            device=self.device,
            output=self.output,
            return_params=return_params,
            generator=generator,
        )


class LowLightDegradation(nn.Module):
    """Legacy-compatible wrapper around the tensor-first degradation pipeline.

    This preserves the constructor and `rng=np.random.RandomState` call pattern
    used by the older dataloader code while routing all work through
    `degrade_video()` and `LowLightParams`.
    """

    def __init__(
        self,
        lux_level: float = 10.0,
        poisson_scale: float = 0.02,
        gaussian_std: float = 0.01,
        blur_kernel_size: int = 15,
        blur_intensity: float = 1.0,
        temporal_variance: float = 0.05,
        apply_brightness_drop: bool = True,
        *,
        device: Optional[torch.device | str] = None,
        output: str = "match_input",
    ):
        super().__init__()
        self.lux_level = float(lux_level)
        self.poisson_scale_base = float(poisson_scale)
        self.gaussian_std_base = float(gaussian_std)
        self.blur_kernel_size = int(max(1, blur_kernel_size))
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
        self.blur_intensity = float(blur_intensity)
        self.temporal_variance = float(temporal_variance)
        self.apply_brightness_drop = bool(apply_brightness_drop)
        self.device = device
        self.output = output
        self.config = LowLightConfig(max_blur_kernel_size=self.blur_kernel_size)
        self.config.validate()

    @property
    def iso_gain(self) -> float:
        """Effective ISO gain factor derived from lux level."""
        return 1.0 / max(self.lux_level / 50.0, 0.01)

    @property
    def effective_poisson_scale(self) -> float:
        return self.poisson_scale_base * self.iso_gain

    @property
    def effective_gaussian_std(self) -> float:
        return self.gaussian_std_base * self.iso_gain

    @staticmethod
    def _sample_uniform(
        shape: tuple[int, ...],
        low: float,
        high: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rng: Optional[np.random.RandomState],
        generator: Optional[torch.Generator],
    ) -> Tensor:
        if low == high:
            return torch.full(shape, low, device=device, dtype=dtype)
        if rng is not None:
            values = rng.uniform(low, high, size=shape).astype(np.float32)
            return torch.from_numpy(values).to(device=device, dtype=dtype)
        return low + (high - low) * torch.rand(shape, device=device, dtype=dtype, generator=generator)

    @staticmethod
    def _noise_generator(
        device: torch.device,
        rng: Optional[np.random.RandomState],
        generator: Optional[torch.Generator],
    ) -> Optional[torch.Generator]:
        if generator is not None:
            return generator
        if rng is None:
            return None
        seed = int(rng.randint(0, 2**31 - 1))
        try:
            return torch.Generator(device=device).manual_seed(seed)
        except (RuntimeError, TypeError):
            return torch.Generator().manual_seed(seed)

    def _blur_sigmas(self) -> tuple[float, float]:
        if self.blur_intensity <= 0 or self.blur_kernel_size <= 1:
            return 0.0, 0.0

        # Map the old trajectory length heuristic to an anisotropic Gaussian.
        max_extent = max(1.0, self.blur_intensity * self.blur_kernel_size * 0.35)
        max_supported_sigma = (self.config.max_blur_kernel_size // 2) / self.config.blur_truncate
        sigma_major = min(max_extent / math.sqrt(12.0), max_supported_sigma)
        sigma_minor = min(max(sigma_major * 0.25, 0.35), sigma_major)
        return sigma_major, sigma_minor

    def _build_params(
        self,
        batch_size: int,
        num_frames: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rng: Optional[np.random.RandomState],
        generator: Optional[torch.Generator],
    ) -> LowLightParams:
        exposure_ev_value = 0.0
        if self.apply_brightness_drop:
            exposure_ev_value = math.log2(max(self.lux_level / 50.0, 0.01))

        jitter = torch.ones((batch_size, num_frames), device=device, dtype=dtype)
        if self.temporal_variance > 0:
            delta = self._sample_uniform(
                (batch_size, num_frames),
                -self.temporal_variance,
                self.temporal_variance,
                device=device,
                dtype=dtype,
                rng=rng,
                generator=generator,
            )
            jitter = 1.0 + delta

        shot_scale = torch.full(
            (batch_size, num_frames),
            self.effective_poisson_scale,
            device=device,
            dtype=dtype,
        ) * jitter
        read_noise = torch.full(
            (batch_size, num_frames),
            self.effective_gaussian_std,
            device=device,
            dtype=dtype,
        ) * jitter

        blur_sigma1_value, blur_sigma2_value = self._blur_sigmas()
        if blur_sigma1_value > 0:
            blur_angle = self._sample_uniform(
                (batch_size, 1),
                0.0,
                math.pi,
                device=device,
                dtype=dtype,
                rng=rng,
                generator=generator,
            ).expand(batch_size, num_frames)
            blur_sigma1 = torch.full(
                (batch_size, num_frames),
                blur_sigma1_value,
                device=device,
                dtype=dtype,
            )
            blur_sigma2 = torch.full(
                (batch_size, num_frames),
                blur_sigma2_value,
                device=device,
                dtype=dtype,
            )
        else:
            blur_angle = torch.zeros((batch_size, num_frames), device=device, dtype=dtype)
            blur_sigma1 = torch.zeros((batch_size, num_frames), device=device, dtype=dtype)
            blur_sigma2 = torch.zeros((batch_size, num_frames), device=device, dtype=dtype)

        zeros = torch.zeros((batch_size, num_frames), device=device, dtype=dtype)
        return LowLightParams(
            exposure_ev=torch.full(
                (batch_size, num_frames),
                exposure_ev_value,
                device=device,
                dtype=dtype,
            ),
            shot_scale=shot_scale.clamp_min(1e-6),
            read_noise=read_noise.clamp_min(0.0),
            quant_step=zeros.clone(),
            band_noise=zeros.clone(),
            band_direction=zeros.clone(),
            blur_sigma1=blur_sigma1,
            blur_sigma2=blur_sigma2,
            blur_angle=blur_angle,
        )

    @staticmethod
    def _normalize_layout(video: Tensor) -> tuple[Tensor, bool]:
        if video.ndim == 4:
            if video.shape[1] == 3:
                return video, False
            if video.shape[-1] == 3:
                return video.permute(0, 3, 1, 2).contiguous(), True
        elif video.ndim == 5:
            if video.shape[-3] == 3:
                return video, False
            if video.shape[-1] == 3:
                return video.permute(0, 1, 4, 2, 3).contiguous(), True
        raise ValueError(
            "Expected RGB video with shape (T,C,H,W), (T,H,W,C), (B,T,C,H,W), or (B,T,H,W,C), "
            f"got {tuple(video.shape)}"
        )

    @staticmethod
    def _restore_layout(video: Tensor, channels_last: bool) -> Tensor:
        if not channels_last:
            return video
        if video.ndim == 4:
            return video.permute(0, 2, 3, 1).contiguous()
        return video.permute(0, 1, 3, 4, 2).contiguous()

    def forward(
        self,
        frames: Tensor,
        rng: Optional[np.random.RandomState] = None,
        *,
        params: Optional[LowLightParams] = None,
        return_params: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor | tuple[Tensor, LowLightParams]:
        """Apply low-light degradation using the new pipeline with old call semantics."""
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)

        frames_cf, channels_last = self._normalize_layout(frames)
        process_device = torch.device(self.device) if self.device is not None else frames_cf.device
        param_generator = generator
        noise_generator = self._noise_generator(process_device, rng, generator)

        if params is None:
            if frames_cf.ndim == 4:
                batch_size = 1
                num_frames = frames_cf.shape[0]
            else:
                batch_size = frames_cf.shape[0]
                num_frames = frames_cf.shape[1]
            params = self._build_params(
                batch_size,
                num_frames,
                device=process_device,
                dtype=torch.float32,
                rng=rng,
                generator=param_generator,
            )

        result = degrade_video(
            frames_cf,
            params=params,
            config=self.config,
            device=process_device,
            output=self.output,
            return_params=return_params,
            generator=noise_generator,
        )

        if return_params:
            degraded, used_params = result
            return self._restore_layout(degraded, channels_last), used_params
        return self._restore_layout(result, channels_last)


def build_degradation(cfg: dict) -> Optional[LowLightDegradation]:
    """Build the legacy-compatible degradation wrapper from a config dict."""
    deg_cfg = cfg.get("degradation", {})
    if not deg_cfg.get("enabled", False):
        return None

    return LowLightDegradation(
        lux_level=deg_cfg.get("lux_level", 10.0),
        poisson_scale=deg_cfg.get("poisson_scale", 0.02),
        gaussian_std=deg_cfg.get("gaussian_std", 0.01),
        blur_kernel_size=deg_cfg.get("blur_kernel_size", 15),
        blur_intensity=deg_cfg.get("blur_intensity", 1.0),
        temporal_variance=deg_cfg.get("temporal_variance", 0.05),
        apply_brightness_drop=deg_cfg.get("apply_brightness_drop", True),
        device=deg_cfg.get("device"),
        output=deg_cfg.get("output", "match_input"),
    )
