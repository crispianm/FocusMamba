"""
Models package — Degradation-Robust Metric Video Depth Estimation
==================================================================

Provides:
    - FocusMamba:       Mamba-based depth model (primary)
    - FocusTransformer: Transformer-based ablation baseline
    - VideoDepthAnythingModel: Trainable VDA baseline
    - build_model:      Factory function from config dict
"""

from __future__ import annotations

from .video_depth_anything_model import VideoDepthAnythingModel

_FOCUS_MAMBA_IMPORT_ERROR: Exception | None = None
_FOCUS_TRANSFORMER_IMPORT_ERROR: Exception | None = None

try:
    from .focus_mamba import FocusMamba
except Exception as exc:  # pragma: no cover - optional dependency path
    FocusMamba = None  # type: ignore[assignment]
    _FOCUS_MAMBA_IMPORT_ERROR = exc

try:
    from .focus_transformer import FocusTransformer
except Exception as exc:  # pragma: no cover - optional dependency path
    FocusTransformer = None  # type: ignore[assignment]
    _FOCUS_TRANSFORMER_IMPORT_ERROR = exc


def build_model(cfg: dict) -> "torch.nn.Module":
    """Factory function: instantiate model from a config dict.

    Reads ``cfg['model']['type']`` (or legacy ``cfg['model_type']``) and
    returns the corresponding architecture. Both models share the same
    encoder/decoder channel hierarchy and tubelet embedding — only the
    sequence-modelling core differs.

    All models now output metric depth via exp(log_depth).

    Supported types:
        - ``'mamba'``          → :class:`FocusMamba`
        - ``'transformer'``    → :class:`FocusTransformer`
        - ``'conv_baseline'``  → :class:`FocusTransformer` (ablation alias)
        - ``'video_depth_anything'`` / ``'vda'`` → :class:`VideoDepthAnythingModel`
    """
    import torch  # local import to avoid circular at package level

    # Support both new nested config and legacy flat config
    model_cfg = cfg.get("model", cfg)
    model_type = model_cfg.get("type", cfg.get("model_type", "mamba")).lower()

    data_cfg = cfg.get("data", {})
    predict_uncertainty = cfg.get("loss", {}).get("uncertainty_nll_weight", 0) > 0
    train_frames = int(data_cfg.get("train_num_frames", data_cfg.get("num_frames", 8)))
    val_frames = int(data_cfg.get("val_num_frames", data_cfg.get("num_frames", train_frames)))
    model_num_frames = int(model_cfg.get("num_frames", max(train_frames, val_frames)))

    if model_type == "mamba":
        if FocusMamba is None:
            raise ImportError(
                "FocusMamba is unavailable in this environment. "
                f"Original import error: {_FOCUS_MAMBA_IMPORT_ERROR}"
            )
        model = FocusMamba(
            in_channels=3,
            variant=model_cfg.get("variant", "small"),
            embed_dim=model_cfg.get("embed_dim", cfg.get("embed_dim", None)),
            depths=model_cfg.get("encoder_depths", cfg.get("encoder_depths", None)),
            patch_size=model_cfg.get("patch_size", data_cfg.get("patch_size", 14)),
            t_patch=model_cfg.get("t_patch", data_cfg.get("t_patch", 1)),
            d_state=model_cfg.get("d_state", cfg.get("d_state", 16)),
            d_conv=model_cfg.get("d_conv", cfg.get("d_conv", 4)),
            expand=model_cfg.get("expand", cfg.get("expand", 2)),
            predict_uncertainty=predict_uncertainty,
            num_blocks=model_cfg.get("num_blocks", None),
            out_indices=model_cfg.get("out_indices", None),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            num_frames=model_num_frames,
            positional_encoding=model_cfg.get("positional_encoding", "ape"),
            checkpoint_path=model_cfg.get("checkpoint_path", None),
            strict_checkpoint=bool(model_cfg.get("strict_checkpoint", False)),
            require_mamba=bool(model_cfg.get("require_mamba", True)),
            output_activation=model_cfg.get("output_activation", "softplus"),
        )
    elif model_type in ("transformer", "conv_baseline"):
        if FocusTransformer is None:
            raise ImportError(
                "FocusTransformer is unavailable in this environment. "
                f"Original import error: {_FOCUS_TRANSFORMER_IMPORT_ERROR}"
            )
        common_kwargs = dict(
            in_channels=3,
            embed_dim=model_cfg.get("embed_dim", cfg.get("embed_dim", 96)),
            depths=model_cfg.get("encoder_depths", cfg.get("encoder_depths", [2, 2, 4, 2])),
            patch_size=data_cfg.get("patch_size", cfg.get("patch_size", 4)),
            t_patch=data_cfg.get("t_patch", cfg.get("t_patch", 2)),
            d_state=model_cfg.get("d_state", cfg.get("d_state", 16)),
            d_conv=model_cfg.get("d_conv", cfg.get("d_conv", 4)),
            expand=model_cfg.get("expand", cfg.get("expand", 2)),
            predict_uncertainty=predict_uncertainty,
        )
        # Use the tuned mlp_ratio for parameter matching (see utils/param_check.py)
        transformer_kwargs = {
            **common_kwargs,
            "mlp_ratio": model_cfg.get("transformer_mlp_ratio", cfg.get("transformer_mlp_ratio", 3.15)),
        }
        model = FocusTransformer(**transformer_kwargs)
    elif model_type in ("video_depth_anything", "vda"):
        model = VideoDepthAnythingModel(
            variant=model_cfg.get("variant", "small"),
            num_frames=model_num_frames,
            positional_encoding=model_cfg.get("positional_encoding", "ape"),
            checkpoint_path=model_cfg.get("checkpoint_path", None),
            strict_checkpoint=bool(model_cfg.get("strict_checkpoint", False)),
            mode=model_cfg.get("mode", model_cfg.get("streaming_mode", "offline")),
            stream_max_cache_len=model_cfg.get("stream_max_cache_len", None),
            stream_reset_interval=int(model_cfg.get("stream_reset_interval", 0)),
            state_gate_enabled=bool(model_cfg.get("state_gate_enabled", False)),
            state_gate_reduction=int(model_cfg.get("state_gate_reduction", 8)),
            state_gate_stage_mask=model_cfg.get("state_gate_stage_mask", None),
            prefilter_enabled=bool(model_cfg.get("prefilter_enabled", False)),
            prefilter_type=str(model_cfg.get("prefilter_type", "fast_classical")),
            prefilter_target_mean=model_cfg.get("prefilter_target_mean", None),
            prefilter_target_std=model_cfg.get("prefilter_target_std", None),
            prefilter_kernel_size=int(model_cfg.get("prefilter_kernel_size", 5)),
            prefilter_sigma=float(model_cfg.get("prefilter_sigma", 1.0)),
            prefilter_denoise_init=float(model_cfg.get("prefilter_denoise_init", 0.20)),
            prefilter_sharpen_init=float(model_cfg.get("prefilter_sharpen_init", 0.10)),
            prefilter_learnable=bool(model_cfg.get("prefilter_learnable", True)),
            front_adapter_hidden=int(model_cfg.get("front_adapter_hidden", 16)),
            front_adapter_blocks=int(model_cfg.get("front_adapter_blocks", 2)),
            front_adapter_use_stats_align=bool(model_cfg.get("front_adapter_use_stats_align", True)),
            front_adapter_use_se=bool(model_cfg.get("front_adapter_use_se", True)),
            pre_temporal_stage_adapter_enabled=bool(model_cfg.get("pre_temporal_stage_adapter_enabled", False)),
            pre_temporal_stage_adapter_stages=model_cfg.get("pre_temporal_stage_adapter_stages", None),
            pre_temporal_stage_adapter_bottleneck_ratio=int(
                model_cfg.get("pre_temporal_stage_adapter_bottleneck_ratio", 4)
            ),
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be 'mamba', 'transformer', 'conv_baseline', or 'video_depth_anything'."
        )

    return model

__all__ = ["FocusMamba", "FocusTransformer", "VideoDepthAnythingModel", "build_model"]
