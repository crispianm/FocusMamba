"""
Models package — Degradation-Robust Metric Video Depth Estimation
==================================================================

Provides:
    - VideoDepthAnythingModel: Trainable VDA-Small fine-tune wrapper
    - build_model:      Factory function from config dict
"""

from __future__ import annotations

from .video_depth_anything_model import VideoDepthAnythingModel


def build_model(cfg: dict) -> "torch.nn.Module":
    """Instantiate model from a config dict.

    Reads ``cfg['model']['type']`` (or legacy ``cfg['model_type']``).

    Supported types:
        - ``'video_depth_anything'`` / ``'vda'`` → :class:`VideoDepthAnythingModel`
    """
    import torch  # local import to avoid circular at package level

    # Support both new nested config and legacy flat config
    model_cfg = cfg.get("model", cfg)
    model_type = model_cfg.get("type", cfg.get("model_type", "video_depth_anything")).lower()

    data_cfg = cfg.get("data", {})
    predict_uncertainty = cfg.get("loss", {}).get("uncertainty_nll_weight", 0) > 0
    train_frames = int(data_cfg.get("train_num_frames", data_cfg.get("num_frames", 8)))
    val_frames = int(
        data_cfg.get("val_num_frames", data_cfg.get("num_frames", train_frames))
    )
    model_num_frames = int(model_cfg.get("num_frames", max(train_frames, val_frames)))

    if model_type in ("video_depth_anything", "vda"):
        quality_cfg = model_cfg.get("quality_conditioning", {}) or {}
        aux_recon_cfg = model_cfg.get("aux_reconstruction", {}) or {}
        degradation_conditioning_cfg = (
            model_cfg.get("degradation_conditioning", {}) or {}
        )
        model = VideoDepthAnythingModel(
            variant=model_cfg.get("variant", "small"),
            num_frames=model_num_frames,
            predict_uncertainty=predict_uncertainty,
            positional_encoding=model_cfg.get("positional_encoding", "ape"),
            checkpoint_path=model_cfg.get("checkpoint_path", None),
            strict_checkpoint=bool(model_cfg.get("strict_checkpoint", False)),
            mode=model_cfg.get("mode", model_cfg.get("streaming_mode", "offline")),
            stream_max_cache_len=model_cfg.get("stream_max_cache_len", None),
            stream_reset_interval=int(model_cfg.get("stream_reset_interval", 0)),
            stream_training_mask_probability=float(
                model_cfg.get("stream_training_mask_probability", 0.0)
            ),
            stream_training_mask_fill=str(
                model_cfg.get("stream_training_mask_fill", "previous")
            ),
            stream_training_random_reset_probability=float(
                model_cfg.get("stream_training_random_reset_probability", 0.0)
            ),
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
            front_adapter_use_stats_align=bool(
                model_cfg.get("front_adapter_use_stats_align", True)
            ),
            front_adapter_use_se=bool(model_cfg.get("front_adapter_use_se", True)),
            refiner_enabled=bool(model_cfg.get("refiner_enabled", False)),
            refiner_channels=int(model_cfg.get("refiner_channels", 32)),
            refiner_scale=float(model_cfg.get("refiner_scale", 0.5)),
            refiner_predict_edges=bool(model_cfg.get("refiner_predict_edges", False)),
            quality_conditioning_enabled=bool(
                quality_cfg.get(
                    "enabled", model_cfg.get("quality_conditioning_enabled", False)
                )
            ),
            quality_embedding_dim=int(
                quality_cfg.get("dim", model_cfg.get("quality_embedding_dim", 16))
            ),
            quality_hidden_channels=int(
                quality_cfg.get(
                    "hidden_channels", model_cfg.get("quality_hidden_channels", 16)
                )
            ),
            aux_reconstruction_enabled=bool(
                aux_recon_cfg.get(
                    "enabled", model_cfg.get("aux_reconstruction_enabled", False)
                )
            ),
            aux_reconstruction_channels=int(
                aux_recon_cfg.get(
                    "channels", model_cfg.get("aux_reconstruction_channels", 32)
                )
            ),
            degradation_conditioning_enabled=bool(
                degradation_conditioning_cfg.get(
                    "enabled", model_cfg.get("degradation_conditioning_enabled", False)
                )
            ),
            degradation_conditioning_hidden=int(
                degradation_conditioning_cfg.get(
                    "hidden",
                    degradation_conditioning_cfg.get(
                        "dim", model_cfg.get("degradation_conditioning_hidden", 32)
                    ),
                )
            ),
            pre_temporal_stage_adapter_enabled=bool(
                model_cfg.get("pre_temporal_stage_adapter_enabled", False)
            ),
            pre_temporal_stage_adapter_stages=model_cfg.get(
                "pre_temporal_stage_adapter_stages", None
            ),
            pre_temporal_stage_adapter_bottleneck_ratio=int(
                model_cfg.get("pre_temporal_stage_adapter_bottleneck_ratio", 4)
            ),
            temporal_module_type=str(
                model_cfg.get("temporal_module_type", "attention")
            ),
            temporal_mamba_d_state=int(model_cfg.get("temporal_mamba_d_state", 16)),
            temporal_mamba_d_conv=int(model_cfg.get("temporal_mamba_d_conv", 4)),
            temporal_mamba_expand=int(model_cfg.get("temporal_mamba_expand", 2)),
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Must be 'video_depth_anything' or 'vda'."
        )

    return model


__all__ = ["VideoDepthAnythingModel", "build_model"]
