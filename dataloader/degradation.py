"""
Low-Light Degradation Pipeline
================================

Physically accurate low-light simulation for focal sweep sequences.

Components:
    1. **Heteroscedastic Poisson-Gaussian noise**: Calibrated to simulate high ISO
       levels.  The variance is signal-dependent (Poisson component) plus a read-noise
       floor (Gaussian component), matching real sensor behaviour.
    2. **Parameterised motion blur**: Simulates handheld camera shake via random
       trajectory kernels applied consistently across the temporal sequence.
    3. **Temporal coherence**: Degradation parameters (trajectory, noise seed offsets)
       are sampled *once per clip* and applied with controlled per-frame variance to
       maintain realistic temporal consistency across a 49-frame focal sweep.

Usage:
    degradation = LowLightDegradation(
        lux_level=5.0,           # target illuminance (lower = darker)
        poisson_scale=0.02,      # shot-noise scaling
        gaussian_std=0.01,       # read-noise floor
        blur_kernel_size=15,     # max motion-blur kernel spatial extent
        blur_intensity=1.0,      # strength of camera-shake trajectory
        temporal_variance=0.05,  # per-frame jitter around clip-level params
    )
    degraded_frames = degradation(frames_tensor)  # (T, H, W, C) uint8 numpy or float tensor
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Motion blur kernel generation
# ---------------------------------------------------------------------------

def _random_trajectory(
    num_steps: int = 64,
    max_total_length: float = 20.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate a random 2-D camera-shake trajectory.

    Returns an array of shape (num_steps, 2) representing cumulative (x, y)
    positions.  The trajectory is normalised so its bounding-box diagonal does
    not exceed *max_total_length* pixels.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Correlated random walk: smoothed acceleration → velocity → position
    accel = rng.randn(num_steps, 2) * 0.5
    # Low-pass via cumsum + mild damping
    velocity = np.cumsum(accel, axis=0)
    velocity *= 0.95 ** np.arange(num_steps)[:, None]
    trajectory = np.cumsum(velocity, axis=0)

    # Normalise to max_total_length
    span = trajectory.max(axis=0) - trajectory.min(axis=0) + 1e-6
    diag = np.linalg.norm(span)
    if diag > 1e-4:
        trajectory = trajectory / diag * max_total_length

    # Centre around origin
    trajectory -= trajectory.mean(axis=0)
    return trajectory


def _trajectory_to_kernel(
    trajectory: np.ndarray,
    kernel_size: int = 15,
) -> np.ndarray:
    """Rasterise a 2-D trajectory into a PSF kernel of shape (kernel_size, kernel_size).

    Each trajectory point is splatted onto the nearest pixel with bilinear
    weighting.  The kernel is normalised to sum to 1.
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    centre = kernel_size / 2.0

    for x, y in trajectory:
        px = x + centre
        py = y + centre
        # Clamp to kernel bounds
        px = np.clip(px, 0, kernel_size - 1.001)
        py = np.clip(py, 0, kernel_size - 1.001)
        ix, iy = int(px), int(py)
        fx, fy = px - ix, py - iy

        # Bilinear splatting
        if ix < kernel_size - 1 and iy < kernel_size - 1:
            kernel[iy, ix] += (1 - fx) * (1 - fy)
            kernel[iy, ix + 1] += fx * (1 - fy)
            kernel[iy + 1, ix] += (1 - fx) * fy
            kernel[iy + 1, ix + 1] += fx * fy
        elif ix < kernel_size and iy < kernel_size:
            kernel[min(iy, kernel_size - 1), min(ix, kernel_size - 1)] += 1.0

    total = kernel.sum()
    if total > 0:
        kernel /= total
    else:
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
    return kernel.astype(np.float32)


def generate_motion_blur_kernel(
    kernel_size: int = 15,
    intensity: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate a motion blur PSF kernel.

    Args:
        kernel_size: Spatial extent of the kernel (must be odd).
        intensity: Controls the trajectory length — higher = more blur.
        rng: Numpy RandomState for reproducibility.

    Returns:
        Kernel of shape (kernel_size, kernel_size), sums to 1.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    max_len = max(1.0, intensity * kernel_size * 0.35)
    trajectory = _random_trajectory(
        num_steps=max(16, int(64 * intensity)),
        max_total_length=max_len,
        rng=rng,
    )
    return _trajectory_to_kernel(trajectory, kernel_size)


# ---------------------------------------------------------------------------
# Heteroscedastic Poisson-Gaussian noise
# ---------------------------------------------------------------------------

def apply_poisson_gaussian_noise(
    image: torch.Tensor,
    poisson_scale: float = 0.02,
    gaussian_std: float = 0.01,
    rng_seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply heteroscedastic Poisson-Gaussian noise to an image tensor.

    The noise model follows:
        y = x + sqrt(poisson_scale * x) * n_p + gaussian_std * n_g

    where n_p, n_g ~ N(0, 1).  This is a first-order approximation of the
    Poisson-Gaussian camera sensor model (Foi et al., 2008).

    Args:
        image: Float tensor in [0, 1] of any shape (..., H, W) or (..., H, W, C).
        poisson_scale: Shot-noise gain (higher = noisier; maps to ISO).
        gaussian_std: Read-noise standard deviation (ISO-independent floor).
        rng_seed: Optional seed for the noise generator (for reproducibility).

    Returns:
        Noisy image, clamped to [0, 1].
    """
    if rng_seed is not None:
        gen = torch.Generator(device=image.device).manual_seed(rng_seed)
    else:
        gen = None

    # Signal-dependent (Poisson) component
    poisson_std = (poisson_scale * image.clamp(min=0)).sqrt()
    noise_p = torch.empty_like(image).normal_(0, 1, generator=gen) * poisson_std

    # Signal-independent (Gaussian / read-noise) component
    noise_g = torch.empty_like(image).normal_(0, gaussian_std, generator=gen)

    noisy = image + noise_p + noise_g
    return noisy.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main degradation class
# ---------------------------------------------------------------------------

class LowLightDegradation(nn.Module):
    """Physically motivated low-light degradation pipeline.

    Combines heteroscedastic Poisson-Gaussian noise with motion blur,
    applied consistently across a temporal sequence of frames.

    The ``lux_level`` parameter controls overall brightness reduction *before*
    noise injection.  Lower lux → darker image → higher effective noise/signal.

    ISO mapping heuristic:
        effective_poisson = poisson_scale / max(lux_level / 50, 0.01)
        effective_gaussian = gaussian_std / max(lux_level / 50, 0.01)

    This ensures that as lux drops, sensor gain (ISO) rises and noise increases.

    Args:
        lux_level: Simulated illuminance level (50 = normal; 5 = very low light).
        poisson_scale: Base Poisson noise scale at lux=50.
        gaussian_std: Base Gaussian read-noise std at lux=50.
        blur_kernel_size: Size of the motion-blur PSF kernel.
        blur_intensity: Strength of camera-shake motion.
        temporal_variance: Per-frame random variation in noise parameters
            (fraction of base value).  0 = identical noise per frame.
        apply_brightness_drop: Whether to reduce brightness to simulate low-light.
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
    ):
        super().__init__()
        self.lux_level = lux_level
        self.poisson_scale_base = poisson_scale
        self.gaussian_std_base = gaussian_std
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_intensity = blur_intensity
        self.temporal_variance = temporal_variance
        self.apply_brightness_drop = apply_brightness_drop

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

    def _apply_motion_blur(
        self, frames: torch.Tensor, kernel: np.ndarray
    ) -> torch.Tensor:
        """Apply motion blur to a batch of frames.

        Args:
            frames: (T, C, H, W) or (T, H, W, C) float tensor.
            kernel: (K, K) numpy PSF kernel.

        Returns:
            Blurred frames, same shape as input.
        """
        # Ensure (T, C, H, W)
        needs_permute = frames.ndim == 4 and frames.shape[-1] <= 4  # HWC layout heuristic
        if needs_permute:
            frames = frames.permute(0, 3, 1, 2)

        T, C, H, W = frames.shape
        k = kernel.shape[0]
        pad = k // 2

        # Build conv kernel: (C, 1, K, K)
        kern_tensor = torch.from_numpy(kernel).float().to(frames.device)
        kern_tensor = kern_tensor.unsqueeze(0).unsqueeze(0).expand(C, 1, k, k)

        # Apply as grouped (depthwise) convolution
        frames_padded = F.pad(frames, (pad, pad, pad, pad), mode="reflect")
        blurred = F.conv2d(frames_padded, kern_tensor, groups=C)

        if needs_permute:
            blurred = blurred.permute(0, 2, 3, 1)
        return blurred

    def forward(
        self,
        frames: torch.Tensor,
        rng: Optional[np.random.RandomState] = None,
    ) -> torch.Tensor:
        """Apply low-light degradation to a temporal sequence of frames.

        Args:
            frames: (T, H, W, C) float32 tensor in [0, 1]  (numpy-style layout)
                    OR (T, C, H, W) float32 tensor in [0, 1].
            rng: Optional numpy RNG for reproducible degradation.

        Returns:
            Degraded frames, same shape and range [0, 1].
        """
        if rng is None:
            rng = np.random.RandomState()

        is_hwc = frames.ndim == 4 and frames.shape[-1] <= 4
        if is_hwc:
            # Convert (T, H, W, C) → (T, C, H, W) for processing
            frames = frames.permute(0, 3, 1, 2).contiguous()

        T, C, H, W = frames.shape

        # --- Step 1: Brightness reduction (simulate low-light scene) ---
        if self.apply_brightness_drop:
            brightness_factor = max(self.lux_level / 50.0, 0.01)
            frames = frames * brightness_factor

        # --- Step 2: Motion blur (clip-level consistent kernel) ---
        if self.blur_intensity > 0 and self.blur_kernel_size > 1:
            kernel = generate_motion_blur_kernel(
                kernel_size=self.blur_kernel_size,
                intensity=self.blur_intensity,
                rng=rng,
            )
            frames = self._apply_motion_blur(frames, kernel)

        # --- Step 3: Heteroscedastic noise (per-frame with temporal variance) ---
        eff_ps = self.effective_poisson_scale
        eff_gs = self.effective_gaussian_std

        base_seed = rng.randint(0, 2**31)
        for t in range(T):
            # Per-frame jitter: slight variation to avoid perfectly identical noise
            if self.temporal_variance > 0:
                jitter = 1.0 + rng.uniform(
                    -self.temporal_variance, self.temporal_variance
                )
            else:
                jitter = 1.0

            frame_ps = eff_ps * jitter
            frame_gs = eff_gs * jitter

            frames[t] = apply_poisson_gaussian_noise(
                frames[t],
                poisson_scale=frame_ps,
                gaussian_std=frame_gs,
                rng_seed=base_seed + t,  # deterministic but varying per frame
            )

        # Clamp final output
        frames = frames.clamp(0.0, 1.0)

        if is_hwc:
            frames = frames.permute(0, 2, 3, 1).contiguous()

        return frames


# ---------------------------------------------------------------------------
# Convenience: create degradation from config dict
# ---------------------------------------------------------------------------

def build_degradation(cfg: dict) -> Optional[LowLightDegradation]:
    """Build a LowLightDegradation instance from a config dictionary.

    Expected keys under ``cfg['degradation']``:
        enabled: bool
        lux_level: float
        poisson_scale: float
        gaussian_std: float
        blur_kernel_size: int
        blur_intensity: float
        temporal_variance: float

    Returns None if degradation is disabled.
    """
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
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing LowLightDegradation...")

    T, H, W, C = 8, 64, 64, 3
    frames = torch.rand(T, H, W, C)  # Simulate a small focal sweep

    deg = LowLightDegradation(
        lux_level=5.0,
        poisson_scale=0.02,
        gaussian_std=0.01,
        blur_kernel_size=15,
        blur_intensity=1.0,
        temporal_variance=0.05,
    )

    rng = np.random.RandomState(42)
    degraded = deg(frames.clone(), rng=rng)

    assert degraded.shape == frames.shape, f"Shape mismatch: {degraded.shape} vs {frames.shape}"
    assert degraded.min() >= 0.0 and degraded.max() <= 1.0, "Output out of [0,1]"
    assert not torch.allclose(degraded, frames), "Degradation should change the frames"

    # Test with CHW layout
    frames_chw = torch.rand(T, C, H, W)
    degraded_chw = deg(frames_chw.clone(), rng=np.random.RandomState(42))
    assert degraded_chw.shape == frames_chw.shape
    assert degraded_chw.min() >= 0.0 and degraded_chw.max() <= 1.0

    print(f"  Input range:  [{frames.min():.3f}, {frames.max():.3f}]")
    print(f"  Output range: [{degraded.min():.3f}, {degraded.max():.3f}]")
    print(f"  Mean diff:    {(degraded - frames).abs().mean():.4f}")
    print("LowLightDegradation test passed!")
