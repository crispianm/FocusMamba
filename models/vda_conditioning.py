"""VDA-specific conditioning helpers for bug-hunt option surfaces."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


_SUMMARY_FEATURE_KEYS: tuple[str, ...] = (
    "policy_code",
    "severity_scale",
    "applied",
    "mean_exposure_ev",
    "mean_shot_scale",
    "mean_read_noise",
    "mean_quant_step",
    "mean_band_noise",
    "mean_blur_sigma1",
    "mean_blur_sigma2",
    "time_mode_code",
    "processing_applied",
    "sensor_applied",
    "blocky_artifact",
    "line_jitter",
    "hot_dead_pixels",
)

_PARAM_FALLBACK_KEYS: tuple[str, ...] = (
    "exposure_ev",
    "shot_scale",
    "read_noise",
    "quant_step",
    "band_noise",
    "blur_sigma1",
    "blur_sigma2",
    "band_direction",
)


def _reduce_tensor_to_batch(
    value: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)

    if tensor.ndim == 0:
        return tensor.expand(batch_size)
    if tensor.ndim == 1:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size)
        return tensor.mean().expand(batch_size)

    while tensor.ndim > 1:
        tensor = tensor.mean(dim=-1)

    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size)
    return tensor.mean().expand(batch_size)


def build_degradation_quality_features(
    degradation_params: Mapping[str, Any] | None,
    degradation_summary: Mapping[str, Any] | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build a fixed-size metadata vector from degradation outputs.

    The summary path is preferred because it already contains the clip-level
    scalars we want for option 6. If the summary is missing, fall back to the
    raw per-frame params where possible.
    """

    if degradation_params is None and degradation_summary is None:
        return None

    features: list[torch.Tensor] = []
    summary = degradation_summary or {}
    params = degradation_params or {}
    for key in _SUMMARY_FEATURE_KEYS:
        tensor = _reduce_tensor_to_batch(
            summary.get(key), batch_size=batch_size, device=device, dtype=dtype
        )
        if tensor is None and key in _PARAM_FALLBACK_KEYS:
            tensor = _reduce_tensor_to_batch(
                params.get(key), batch_size=batch_size, device=device, dtype=dtype
            )
        if tensor is None:
            tensor = torch.zeros(batch_size, device=device, dtype=dtype)
        features.append(tensor)
    return torch.stack(features, dim=-1)


class DegradationMetadataConditioner(nn.Module):
    """Tiny MLP that maps degradation metadata to a quality embedding."""

    def __init__(
        self, input_dim: int = 16, hidden_dim: int = 32, embedding_dim: int = 16
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        hidden_dim = max(int(hidden_dim), self.embedding_dim)
        self.norm = nn.LayerNorm(self.input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        if metadata.ndim != 2:
            raise ValueError(
                f"DegradationMetadataConditioner expects (B, K), got {tuple(metadata.shape)}"
            )
        return self.net(self.norm(metadata))


__all__ = [
    "DegradationMetadataConditioner",
    "build_degradation_quality_features",
]
