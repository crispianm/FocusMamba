"""
Exponential Moving Average (EMA) of Model Weights
===================================================

Maintains a shadow copy of model weights that is an exponential moving
average of the training weights. The EMA model typically produces better
predictions (smoother, less noisy) than the training model.

Usage:
    ema = EMAModel(model, decay=0.999)
    for batch in dataloader:
        train_step(model, batch)
        ema.update()
    # Evaluate with EMA weights:
    ema.apply_shadow()
    evaluate(model)
    ema.restore()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor (0.999 is typical).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        # Initialize shadow with current parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self) -> None:
        """Replace model parameters with EMA shadow parameters.
        Call restore() to undo."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original model parameters (undo apply_shadow)."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        """Load EMA state from checkpoint."""
        self.shadow = state["shadow"]
        self.decay = state.get("decay", self.decay)
