"""
Models package — Degradation-Robust Metric Video Depth Estimation
==================================================================

Provides:
    - FocusMamba:       Mamba-based depth model (primary)
    - FocusTransformer: Transformer-based ablation baseline
    - build_model:      Factory function from config dict
"""

from .focus_mamba import FocusMamba
from .focus_transformer import FocusTransformer


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
    """
    import torch  # local import to avoid circular at package level

    # Support both new nested config and legacy flat config
    model_cfg = cfg.get("model", cfg)
    model_type = model_cfg.get("type", cfg.get("model_type", "mamba")).lower()

    common_kwargs = dict(
        in_channels=3,
        embed_dim=model_cfg.get("embed_dim", cfg.get("embed_dim", 96)),
        depths=model_cfg.get("encoder_depths", cfg.get("encoder_depths", [2, 2, 4, 2])),
        patch_size=cfg.get("data", cfg).get("patch_size", cfg.get("patch_size", 4)),
        t_patch=cfg.get("data", cfg).get("t_patch", cfg.get("t_patch", 2)),
        d_state=model_cfg.get("d_state", cfg.get("d_state", 16)),
        d_conv=model_cfg.get("d_conv", cfg.get("d_conv", 4)),
        expand=model_cfg.get("expand", cfg.get("expand", 2)),
        predict_uncertainty=cfg.get("loss", {}).get("uncertainty_nll_weight", 0) > 0,
    )

    if model_type == "mamba":
        model = FocusMamba(**common_kwargs)
    elif model_type in ("transformer", "conv_baseline"):
        # Use the tuned mlp_ratio for parameter matching (see utils/param_check.py)
        transformer_kwargs = {
            **common_kwargs,
            "mlp_ratio": model_cfg.get("transformer_mlp_ratio", cfg.get("transformer_mlp_ratio", 3.15)),
        }
        model = FocusTransformer(**transformer_kwargs)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be 'mamba', 'transformer', or 'conv_baseline'."
        )

    return model

__all__ = ["FocusMamba", "FocusTransformer", "build_model"]
