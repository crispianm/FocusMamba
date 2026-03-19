"""Lightweight RGB prefilters shared across video depth models."""

from __future__ import annotations

from typing import Any, Sequence

import torch.nn as nn

__all__ = [
    "FastClassicalPrefilter",
    "ChannelStatsAlignPrefilter",
    "StatsGuidedFrontAdapter",
    "TinyAffineNormalizer",
    "DepthwiseResidualNormalizer",
    "build_prefilter",
]


def __getattr__(name: str) -> Any:
    if name in {
        "FastClassicalPrefilter",
        "ChannelStatsAlignPrefilter",
        "StatsGuidedFrontAdapter",
        "TinyAffineNormalizer",
        "DepthwiseResidualNormalizer",
    }:
        from .video_depth_anything_model import (
            ChannelStatsAlignPrefilter,
            DepthwiseResidualNormalizer,
            FastClassicalPrefilter,
            StatsGuidedFrontAdapter,
            TinyAffineNormalizer,
        )

        exports = {
            "FastClassicalPrefilter": FastClassicalPrefilter,
            "ChannelStatsAlignPrefilter": ChannelStatsAlignPrefilter,
            "StatsGuidedFrontAdapter": StatsGuidedFrontAdapter,
            "TinyAffineNormalizer": TinyAffineNormalizer,
            "DepthwiseResidualNormalizer": DepthwiseResidualNormalizer,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_prefilter(
    *,
    enabled: bool,
    prefilter_type: str,
    target_mean: Sequence[float] | None = None,
    target_std: Sequence[float] | None = None,
    kernel_size: int = 5,
    sigma: float = 1.0,
    denoise_init: float = 0.20,
    sharpen_init: float = 0.10,
    learnable: bool = True,
    front_adapter_hidden: int = 16,
    front_adapter_blocks: int = 2,
    front_adapter_use_stats_align: bool = True,
    front_adapter_use_se: bool = True,
) -> nn.Module | None:
    if not enabled:
        return None

    from .video_depth_anything_model import (
        ChannelStatsAlignPrefilter,
        DepthwiseResidualNormalizer,
        FastClassicalPrefilter,
        StatsGuidedFrontAdapter,
        TinyAffineNormalizer,
    )

    kind = str(prefilter_type).strip().lower()
    mean = tuple(target_mean or (0.485, 0.456, 0.406))
    std = tuple(target_std or (0.229, 0.224, 0.225))
    if kind == "fast_classical":
        return FastClassicalPrefilter(
            kernel_size=int(kernel_size),
            sigma=float(sigma),
            denoise_init=float(denoise_init),
            sharpen_init=float(sharpen_init),
            learnable=bool(learnable),
        )
    if kind == "stats_align":
        return ChannelStatsAlignPrefilter(
            target_mean=mean, target_std=std, learnable=bool(learnable)
        )
    if kind == "learned_affine":
        return TinyAffineNormalizer()
    if kind == "depthwise":
        return DepthwiseResidualNormalizer()
    if kind == "stats_guided_front_adapter":
        return StatsGuidedFrontAdapter(
            hidden_channels=int(front_adapter_hidden),
            num_blocks=int(front_adapter_blocks),
            use_stats_align=bool(front_adapter_use_stats_align),
            use_se=bool(front_adapter_use_se),
            target_mean=mean,
            target_std=std,
            learnable_stats=bool(learnable),
        )

    valid = (
        "fast_classical",
        "stats_align",
        "learned_affine",
        "depthwise",
        "stats_guided_front_adapter",
    )
    raise ValueError(
        f"Unknown prefilter_type={prefilter_type!r}. Valid: {', '.join(valid)}"
    )
