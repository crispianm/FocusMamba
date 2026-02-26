"""
Pretrained Weight Initialization
==================================

Utilities for loading pretrained weights (ImageNet, VideoMAE, etc.)
into the Mamba or Transformer encoder.

Usage:
    from models.encoders.pretrained_init import load_pretrained_encoder

    model = FocusMamba(...)
    load_pretrained_encoder(model.encoder, "path/to/weights.pth", strict=False)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def load_pretrained_encoder(
    encoder: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    key_prefix: Optional[str] = None,
) -> list[str]:
    """Load pretrained weights into an encoder module.

    Handles mismatched keys gracefully when strict=False.

    Args:
        encoder: The encoder module to load weights into.
        checkpoint_path: Path to .pt / .pth checkpoint.
        strict: If True, require exact key matching.
        key_prefix: Optional prefix to strip from checkpoint keys
            (e.g. "encoder." if the checkpoint includes full model).

    Returns:
        List of keys that were successfully loaded.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Handle nested checkpoints (e.g. {"model": {...}, "optimizer": {...}})
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip prefix if needed
    if key_prefix:
        state_dict = {
            k[len(key_prefix):] if k.startswith(key_prefix) else k: v
            for k, v in state_dict.items()
        }

    # Load with matching
    result = encoder.load_state_dict(state_dict, strict=strict)

    loaded_keys = [
        k for k in state_dict.keys()
        if k not in (result.unexpected_keys if hasattr(result, 'unexpected_keys') else [])
    ]

    return loaded_keys
