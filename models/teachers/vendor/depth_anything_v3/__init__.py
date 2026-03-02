# Vendored Depth Anything V3 — da3metric-large architecture (hardcoded).
# Original: https://github.com/ByteDance-Seed/Depth-Anything-3
# License: Apache 2.0
#
# Eliminates runtime dependencies on: omegaconf, addict, huggingface_hub,
# pycolmap, gsplat, and the DA3 config/registry system.

from .da3 import DepthAnything3Net, create_da3_metric_large

__all__ = ["DepthAnything3Net", "create_da3_metric_large"]
