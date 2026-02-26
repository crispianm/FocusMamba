"""
Test Model Forward Pass
=========================

Forward pass on dummy input: check output shape, dict keys, no NaN.
"""

from __future__ import annotations

import torch
import pytest


class TestModelForward:
    """Tests for FocusMamba forward pass."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_output_is_dict(self, device):
        """Model should return dict with 'depth' key."""
        from models import build_model

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
        model = build_model(cfg).to(device)
        x = torch.randn(1, 3, 4, 64, 64, device=device)
        out = model(x)

        assert isinstance(out, dict)
        assert "depth" in out

    def test_output_shape(self, device):
        """Output depth should be (B, 1, T, H, W)."""
        from models import build_model

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
        model = build_model(cfg).to(device)
        x = torch.randn(2, 3, 4, 64, 64, device=device)
        out = model(x)

        assert out["depth"].shape == (2, 1, 4, 64, 64)

    def test_no_nan(self, device):
        """Output should not contain NaN."""
        from models import build_model

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
        model = build_model(cfg).to(device)
        x = torch.randn(1, 3, 4, 64, 64, device=device)
        out = model(x)

        assert torch.isfinite(out["depth"]).all()

    def test_positive_depth(self, device):
        """exp(log_depth) should always be positive."""
        from models import build_model

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
        model = build_model(cfg).to(device)
        x = torch.randn(1, 3, 4, 64, 64, device=device)
        out = model(x)

        assert (out["depth"] > 0).all()

    def test_uncertainty_head(self, device):
        """If predict_uncertainty=True, output should have 'uncertainty' key."""
        from models import build_model

        cfg = {
            "model": {
                "type": "mamba",
                "in_channels": 3,
                "base_channels": 16,
                "num_frames": 4,
                "img_size": 64,
            },
            "training": {"loss": {"uncertainty_nll_weight": 0.1}},
        }
        model = build_model(cfg).to(device)
        x = torch.randn(1, 3, 4, 64, 64, device=device)
        out = model(x)

        assert "uncertainty" in out
        assert out["uncertainty"].shape == out["depth"].shape
