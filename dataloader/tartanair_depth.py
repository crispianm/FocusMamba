"""Helpers for TartanAir depth PNG encoding and decoding.

TartanAir stores each depth value as a little-endian ``float32`` packed
losslessly into a 4-channel PNG. The canonical decode path uses
``cv2.imread(..., cv2.IMREAD_UNCHANGED)`` and reinterprets the returned
BGRA bytes as ``<f4``. Reading the same file with PIL and reinterpreting
its RGBA bytes directly is not equivalent.
"""

from __future__ import annotations

from os import PathLike

import cv2
import numpy as np


def decode_tartanair_depth(path: str | PathLike[str]) -> np.ndarray:
    """Decode a TartanAir depth PNG to a ``float32`` metre map."""
    depth_bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_bgra is None:
        raise FileNotFoundError(f"Could not read TartanAir depth PNG: {path}")
    if depth_bgra.ndim != 3 or depth_bgra.shape[2] != 4 or depth_bgra.dtype != np.uint8:
        raise ValueError(
            "Expected a 4-channel uint8 depth PNG, "
            f"got shape={depth_bgra.shape!r} dtype={depth_bgra.dtype!r} for {path}"
        )
    depth_bgra = np.ascontiguousarray(depth_bgra)
    return np.squeeze(depth_bgra.view("<f4"), axis=-1)


def encode_tartanair_depth(depth: np.ndarray) -> np.ndarray:
    """Pack a ``float32`` depth image into TartanAir's 4-channel PNG byte layout."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth array, got shape={depth.shape!r}")
    depth = np.ascontiguousarray(depth[..., None])
    return depth.view("<u1")


__all__ = ["decode_tartanair_depth", "encode_tartanair_depth"]
