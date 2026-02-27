"""
Depth Anything V3 — Image Teacher Wrapper
==========================================

Wraps the Depth Anything 3 model for use as a frozen image-level teacher.
Produces metric depth predictions from individual clean frames.

Repository: https://github.com/ByteDance-Seed/Depth-Anything-3

Usage:
    # From a local weights directory (safetensors + config.json):
    teacher = DepthAnythingV3Teacher(checkpoint_path="/path/to/model_dir")

    # From a .pth file, specify model_name too:
    teacher = DepthAnythingV3Teacher(
        checkpoint_path="checkpoints/da3.pth",
        model_name="da3-large",
    )

    with torch.no_grad():
        depth = teacher.predict(clean_frames)  # (B, 1, T, H, W)
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Optional

import numpy as np
import torch

from .teacher_base import TeacherBase

# Path to Depth-Anything-3 Python package (src/ layout).
_DA3_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "teachers", "Depth-Anything-3", "src")
)


class DepthAnythingV3Teacher(TeacherBase):
    """Depth Anything 3 image teacher.

    Args:
        checkpoint_path: Either a local directory containing
            ``model.safetensors`` + ``config.json`` (for ``from_pretrained``),
            or a ``.pth`` / ``.pt`` file whose state-dict is loaded into the
            model specified by ``model_name``.
            If ``None``, the model is built from ``model_name`` only (weights
            left at random init — useful only for testing).
        model_name: Architecture preset, e.g. ``"da3-large"``,
            ``"da3-giant"``.  Ignored when ``checkpoint_path`` is a directory.
        device: Target device.
        input_size: Expected input resolution (H, W).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "./checkpoints/da3_metric.safetensors",
        model_name: str = "da3metric-large",
        device: str = "cuda",
        input_size: tuple[int, int] = (504, 504),
    ):
        super().__init__(checkpoint_path, device, input_size)
        self.model_name = model_name
        self.model = None

    def _load_model(self) -> None:
        """Load Depth Anything 3 from teachers/Depth-Anything-3."""
        if _DA3_SRC not in sys.path:
            sys.path.insert(0, _DA3_SRC)

        # DA3's export/__init__.py eagerly imports pycolmap and gsplat, which are
        # heavy C++ deps not needed for plain inference.  Stub them out so the
        # import chain succeeds without installing these packages.
        from unittest.mock import MagicMock
        for _stub in ("pycolmap", "gsplat"):
            if _stub not in sys.modules:
                sys.modules[_stub] = MagicMock()

        from depth_anything_3.api import DepthAnything3

        cp = self.checkpoint_path

        if cp is not None and os.path.isdir(cp):
            # Local directory with model.safetensors + config.json
            self.model = DepthAnything3.from_pretrained(cp)
        elif cp is not None and os.path.isfile(cp) and cp.endswith(".safetensors"):
            # Flat .safetensors file.  from_pretrained needs a directory with
            # a file named EXACTLY "model.safetensors" + "config.json".
            # Create a temp directory of symlinks so from_pretrained is happy.
            tmpdir = tempfile.mkdtemp(prefix="da3_ckpt_")
            os.symlink(os.path.abspath(cp), os.path.join(tmpdir, "model.safetensors"))
            config_candidate = os.path.join(os.path.dirname(os.path.abspath(cp)), "config.json")
            if os.path.exists(config_candidate):
                os.symlink(config_candidate, os.path.join(tmpdir, "config.json"))
            else:
                raise FileNotFoundError(
                    f"DA3 needs config.json alongside {cp}. "
                    "Place it at: " + config_candidate
                )
            self.model = DepthAnything3.from_pretrained(tmpdir)
        elif cp is not None and os.path.isfile(cp):
            # Raw torch state-dict (.pth / .pt)
            self.model = DepthAnything3(model_name=self.model_name)
            state = torch.load(cp, map_location="cpu")
            if "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
        else:
            # No checkpoint — build with default weights (HF Hub or random init)
            self.model = DepthAnything3(model_name=self.model_name)

        self.model = self.model.to(self.target_device).eval()

    def _predict_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict metric depth for a single frame.

        Args:
            frame: (1, 3, H, W) float32 in [0, 1].

        Returns:
            depth: (1, 1, H, W) float32 metric depth.
        """
        # Convert to HWC uint8 numpy array expected by inference().
        frame_np = (
            frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        ).astype(np.uint8)

        prediction = self.model.inference(
            [frame_np],
            process_res=self.input_size[0],
        )
        # prediction.depth: np.ndarray (N, H, W), N==1 here
        depth = torch.from_numpy(prediction.depth[0]).float()  # (H, W)
        return depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
