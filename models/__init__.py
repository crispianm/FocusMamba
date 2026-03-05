"""
Models package — Degradation-Robust Metric Video Depth Estimation
==================================================================

Provides:
    - FocusMamba:       Mamba-based depth model (primary)
    - FocusTransformer: Transformer-based ablation baseline
    - VideoDepthAnythingModel: Trainable VDA baseline
    - build_model:      Factory function from config dict
"""

from .focus_mamba import FocusMamba
from .focus_transformer import FocusTransformer
from .video_depth_anything_model import VideoDepthAnythingModel


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

    if model_type == "mamba":
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
            num_frames=model_cfg.get("num_frames", data_cfg.get("num_frames", 8)),
            positional_encoding=model_cfg.get("positional_encoding", "ape"),
            checkpoint_path=model_cfg.get("checkpoint_path", None),
            strict_checkpoint=bool(model_cfg.get("strict_checkpoint", False)),
            require_mamba=bool(model_cfg.get("require_mamba", True)),
            output_activation=model_cfg.get("output_activation", "softplus"),
        )
    elif model_type in ("transformer", "conv_baseline"):
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
            num_frames=model_cfg.get("num_frames", data_cfg.get("num_frames", 8)),
            positional_encoding=model_cfg.get("positional_encoding", "ape"),
            checkpoint_path=model_cfg.get("checkpoint_path", None),
            strict_checkpoint=bool(model_cfg.get("strict_checkpoint", False)),
            tiny_arch=model_cfg.get("tiny_arch", "dual_dinov3"),
            tiny_dino_model_name=model_cfg.get("tiny_dino_model_name", "vit_small_patch16_dinov3"),
            tiny_pretrained=bool(model_cfg.get("tiny_pretrained", True)),
            tiny_relative_bounded=bool(model_cfg.get("tiny_relative_bounded", True)),
            tiny_relative_activation=model_cfg.get("tiny_relative_activation", "softplus"),
            tiny_metric_activation=model_cfg.get("tiny_metric_activation", "softplus"),
            tiny_out_indices=model_cfg.get("tiny_out_indices", None),
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be 'mamba', 'transformer', 'conv_baseline', or 'video_depth_anything'."
        )

    return model

__all__ = ["FocusMamba", "FocusTransformer", "VideoDepthAnythingModel", "build_model"]
