"""
Test Realtime Engine
=====================

Stateful inference: state persists across frames, reset works correctly.
"""

from __future__ import annotations

import torch
import pytest


class TestRealtimeEngine:
    """Tests for the RealtimeDepthEngine."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_single_frame(self, device):
        """Engine should process a single frame and return depth."""
        from models import build_model
        from inference.realtime_engine import RealtimeDepthEngine

        cfg = {
            "model": {
                "type": "mamba",
                "in_channels": 3,
                "base_channels": 16,
                "num_frames": 1,
                "img_size": 64,
            },
            "training": {"loss": {}},
        }
        model = build_model(cfg)
        engine = RealtimeDepthEngine(model, device=device)

        frame = torch.rand(3, 64, 64)
        depth = engine.process_frame(frame)

        assert depth.shape == (1, 64, 64)
        assert (depth > 0).all()

    def test_reset_clears_state(self, device):
        """Reset should clear hidden state and frame counter."""
        from models import build_model
        from inference.realtime_engine import RealtimeDepthEngine

        cfg = {
            "model": {
                "type": "mamba",
                "in_channels": 3,
                "base_channels": 16,
                "num_frames": 1,
                "img_size": 64,
            },
            "training": {"loss": {}},
        }
        model = build_model(cfg)
        engine = RealtimeDepthEngine(model, device=device)

        # Process some frames
        for _ in range(5):
            engine.process_frame(torch.rand(3, 64, 64))

        assert engine._frame_count == 5
        assert engine._prev_frame is not None

        engine.reset()

        assert engine._frame_count == 0
        assert engine._prev_frame is None
        assert engine._hidden_state is None

    def test_process_clip(self, device):
        """process_clip should handle multi-frame input."""
        from models import build_model
        from inference.realtime_engine import RealtimeDepthEngine

        cfg = {
            "model": {
                "type": "mamba",
                "in_channels": 3,
                "base_channels": 16,
                "num_frames": 4,
                "img_size": 64,
            },
            "training": {"loss": {}},
        }
        model = build_model(cfg)
        engine = RealtimeDepthEngine(model, device=device)

        clip = torch.rand(4, 3, 64, 64)  # (T, C, H, W)
        depth = engine.process_clip(clip)

        assert depth.shape == (4, 1, 64, 64)
        assert (depth > 0).all()
