"""
Encoder sub-package.

The primary encoder lives in models/encoder.py (FocusMambaEncoder).
The Transformer encoder lives in models/focus_transformer.py (FocusTransformerEncoder).
This sub-package provides additional utilities:
    - pretrained_init.py: Load pretrained weights into encoders.
"""

from .pretrained_init import load_pretrained_encoder

__all__ = ["load_pretrained_encoder"]
