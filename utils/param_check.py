#!/usr/bin/env python3
"""
Parameter Matching Check: FocusMamba vs FocusTransformer
=========================================================

Instantiates both models with identical config and verifies their trainable
parameter counts are within ~5% of each other.  If they are not, the script
iteratively tunes the Transformer's ``mlp_ratio`` to close the gap.

Usage:
    python utils/param_check.py
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_and_tune(
    embed_dim: int = 96,
    depths: list[int] | None = None,
    patch_size: int = 4,
    t_patch: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    tolerance: float = 0.05,
) -> dict:
    """Instantiate both models and tune Transformer mlp_ratio to match params.

    Returns a dict with param counts and the final mlp_ratio.
    """
    if depths is None:
        depths = [2, 2, 4, 2]

    from models.focus_mamba import FocusMamba
    from models.focus_transformer import FocusTransformer

    # ----- Mamba model -----
    mamba = FocusMamba(
        in_channels=3,
        embed_dim=embed_dim,
        depths=depths,
        patch_size=patch_size,
        t_patch=t_patch,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    mamba_params = count_params(mamba)
    print(f"FocusMamba parameters:        {mamba_params:>12,}")

    # ----- Transformer model — sweep mlp_ratio to match -----
    best_ratio = 2.0
    best_diff = float("inf")
    best_transformer_params = 0

    # Coarse sweep
    for ratio_x10 in range(10, 61, 1):  # 1.0 to 6.0 in steps of 0.1
        ratio = ratio_x10 / 10.0
        transformer = FocusTransformer(
            in_channels=3,
            embed_dim=embed_dim,
            depths=depths,
            patch_size=patch_size,
            t_patch=t_patch,
            mlp_ratio=ratio,
        )
        t_params = count_params(transformer)
        diff = abs(t_params - mamba_params) / mamba_params
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
            best_transformer_params = t_params
        del transformer

    # Fine sweep around the best
    for ratio_x100 in range(int(best_ratio * 100) - 15, int(best_ratio * 100) + 16):
        ratio = ratio_x100 / 100.0
        if ratio < 0.5:
            continue
        transformer = FocusTransformer(
            in_channels=3,
            embed_dim=embed_dim,
            depths=depths,
            patch_size=patch_size,
            t_patch=t_patch,
            mlp_ratio=ratio,
        )
        t_params = count_params(transformer)
        diff = abs(t_params - mamba_params) / mamba_params
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
            best_transformer_params = t_params
        del transformer

    print(f"FocusTransformer parameters:  {best_transformer_params:>12,}")
    print(f"  mlp_ratio = {best_ratio:.2f}")
    print(f"  Relative difference: {best_diff * 100:.2f}%")

    passed = best_diff <= tolerance
    status = "PASS" if passed else "FAIL"
    print(f"\nParameter matching ({tolerance*100:.0f}% tolerance): {status}")

    if not passed:
        print(
            f"  WARNING: Could not match parameters within {tolerance*100:.0f}%.\n"
            f"  Mamba: {mamba_params:,}  Transformer: {best_transformer_params:,}\n"
            f"  Delta: {best_diff*100:.2f}%\n"
            f"  Consider adjusting depths, embed_dim, or Transformer mlp_ratio."
        )

    # Quick forward-pass sanity check
    print("\nRunning forward-pass smoke test...")
    transformer = FocusTransformer(
        in_channels=3,
        embed_dim=embed_dim,
        depths=depths,
        patch_size=patch_size,
        t_patch=t_patch,
        mlp_ratio=best_ratio,
    )
    x = torch.randn(1, 3, 8, 64, 64)
    with torch.no_grad():
        out_mamba = mamba(x)
        out_transformer = transformer(x)
    assert out_mamba.shape == out_transformer.shape, (
        f"Shape mismatch: Mamba {out_mamba.shape} vs Transformer {out_transformer.shape}"
    )
    assert out_mamba.shape == (1, 1, 8, 64, 64), f"Unexpected shape: {out_mamba.shape}"
    print(f"  Mamba output:       {out_mamba.shape}  range=[{out_mamba.min():.3f}, {out_mamba.max():.3f}]")
    print(f"  Transformer output: {out_transformer.shape}  range=[{out_transformer.min():.3f}, {out_transformer.max():.3f}]")
    print("  Smoke test passed!")

    return {
        "mamba_params": mamba_params,
        "transformer_params": best_transformer_params,
        "mlp_ratio": best_ratio,
        "relative_diff": best_diff,
        "passed": passed,
    }


if __name__ == "__main__":
    result = check_and_tune()
    sys.exit(0 if result["passed"] else 1)
