"""
Test Temporal Consistency
===========================

Feed identical frames → verify depth predictions don't drift.
"""

from __future__ import annotations

import torch
import pytest


class TestTemporalConsistency:
    """Tests that temporal processing is stable on constant input."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_identical_frames_stable(self, device):
        """Identical input frames should produce near-identical depth per frame."""
        from models import build_model

        cfg = {
            "model": {
                "type": "mamba",
                "in_channels": 3,
                "base_channels": 16,
                "num_frames": 8,
                "img_size": 64,
            },
            "training": {"loss": {}},
        }
        model = build_model(cfg).to(device).eval()

        # Create a clip of identical frames
        single_frame = torch.randn(1, 3, 1, 64, 64, device=device)
        clip = single_frame.expand(1, 3, 8, 64, 64).contiguous()

        with torch.no_grad():
            out = model(clip)
        depth = out["depth"]  # (1, 1, 8, H, W)

        # Max deviation across frames should be small
        frame_depths = [depth[0, 0, t] for t in range(8)]
        for t in range(1, 8):
            rel_diff = (frame_depths[t] - frame_depths[0]).abs() / (frame_depths[0].abs() + 1e-6)
            assert rel_diff.mean() < 0.1, (
                f"Frame {t} differs from frame 0 by {rel_diff.mean():.4f} (mean relative)"
            )
