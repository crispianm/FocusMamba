# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
#
# Vendored DepthAnything3Net for da3metric-large.
# Hardcoded architecture — no OmegaConf, no registry, no HuggingFace Hub.
# Only mono depth prediction (no cam_enc/dec, no GS adapter).

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .dinov2.dinov2 import DinoV2
from .dpt import DPT

logger = logging.getLogger("da3")

# ---------------------------------------------------------------------------
# da3metric-large config (from configs/da3metric-large.yaml), hardcoded.
# ---------------------------------------------------------------------------
_DA3_METRIC_LARGE = dict(
    backbone=dict(
        name="vitl",
        out_layers=[4, 11, 17, 23],
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        cat_token=False,
    ),
    head=dict(
        dim_in=1024,
        output_dim=1,
        features=256,
        out_channels=[256, 512, 1024, 1024],
    ),
)


class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 mono-depth network (da3metric-large).

    Forward input:  (B, N, 3, H, W) — ImageNet-normalised, N=1 for mono.
    Forward output: dict with:
        - ``depth``:  (B, N, H, W) positive metric depth (exp-activated)
        - ``sky``:    (B, N, 1, H, W) sky logit (relu-activated)
    """

    PATCH_SIZE = 14

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, N, 3, H, W) ImageNet-normalised images. N=1 for mono.

        Returns:
            dict with 'depth' (B, N, H, W) and 'sky' (B, N, 1, H, W).
        """
        feats, _aux = self.backbone(x)
        H, W = x.shape[-2], x.shape[-1]

        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self.head(feats, H, W, patch_start_idx=0)

        return output


def create_da3_metric_large() -> DepthAnything3Net:
    """Build a da3metric-large model (random weights)."""
    cfg = _DA3_METRIC_LARGE
    backbone = DinoV2(**cfg["backbone"])
    head = DPT(**cfg["head"])
    return DepthAnything3Net(backbone=backbone, head=head)


def load_da3_metric_large(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> DepthAnything3Net:
    """Build da3metric-large and load weights from a safetensors or .pth file.

    The checkpoint may have been saved by HuggingFace's ``PyTorchModelHubMixin``
    which wraps the net inside ``DepthAnything3.model``.  This function
    transparently strips the ``model.`` prefix from state-dict keys.

    Args:
        checkpoint_path: Path to ``*.safetensors`` or ``*.pth`` / ``*.pt``.
        device: Target device after loading.

    Returns:
        Frozen ``DepthAnything3Net`` in eval mode on *device*.
    """
    model = create_da3_metric_large()

    ckpt = str(checkpoint_path)
    if ckpt.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("Install safetensors: pip install safetensors") from e
        state = load_file(ckpt, device="cpu")
    else:
        raw = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = raw.get("model", raw)

    # Strip 'model.' prefix added by the HF wrapper (DepthAnything3.model = DepthAnything3Net).
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        key = k.removeprefix("model.")
        cleaned[key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("DA3 missing keys: %s", missing)
    if unexpected:
        logger.info("DA3 unexpected keys (ignored): %d entries", len(unexpected))

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
