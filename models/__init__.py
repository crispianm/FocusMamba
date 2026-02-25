from .focus_mamba import FocusMamba
from .focus_transformer import FocusTransformer


def build_model(cfg: dict) -> "torch.nn.Module":
    """Factory function: instantiate model from a config dict.

    Reads ``cfg['model_type']`` (default ``'mamba'``) and returns the
    corresponding architecture.  Both models share the same encoder/decoder
    channel hierarchy and tubelet embedding so only the sequence-modelling
    core differs.

    Supported types:
        - ``'mamba'``       → :class:`FocusMamba`
        - ``'transformer'`` → :class:`FocusTransformer`
    """
    import torch  # local import to avoid circular at package level

    model_type = cfg.get("model_type", "mamba").lower()

    common_kwargs = dict(
        in_channels=3,
        embed_dim=cfg.get("embed_dim", 96),
        depths=cfg.get("encoder_depths", [2, 2, 4, 2]),
        patch_size=cfg.get("patch_size", 4),
        t_patch=cfg.get("t_patch", 2),
        d_state=cfg.get("d_state", 16),
        d_conv=cfg.get("d_conv", 4),
        expand=cfg.get("expand", 2),
    )

    if model_type == "mamba":
        model = FocusMamba(**common_kwargs)
    elif model_type == "transformer":
        # Use the tuned mlp_ratio for parameter matching (see utils/param_check.py)
        transformer_kwargs = {
            **common_kwargs,
            "mlp_ratio": cfg.get("transformer_mlp_ratio", 3.15),
        }
        model = FocusTransformer(**transformer_kwargs)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Must be 'mamba' or 'transformer'."
        )

    return model
