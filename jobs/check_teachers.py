"""
Pre-flight teacher import checker
==================================

Verifies that all three teacher repos can be imported and (optionally) that
their checkpoints exist on disk.  Run this *before* submitting to the cluster
to catch import errors early.

Usage:
    # Import check only (fast, no GPU needed):
    python jobs/check_teachers.py --dry-run

    # Full check including forward pass on a random tensor (requires GPU/CPU):
    python jobs/check_teachers.py --device cpu

    # From the SLURM script or interactive session:
    uv run python jobs/check_teachers.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

# Make sure the project root is on the path regardless of cwd.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_file(path: str, label: str) -> bool:
    full = os.path.join(_ROOT, path)
    if os.path.isfile(full):
        size_mb = os.path.getsize(full) / 1e6
        _ok(f"{label}: {path}  ({size_mb:.0f} MB)")
        return True
    else:
        _fail(f"{label}: {path}  — NOT FOUND")
        return False


def check_da3_sys_path() -> bool:
    da3_src = os.path.join(_ROOT, "teachers", "Depth-Anything-3", "src")
    if not os.path.isdir(da3_src):
        _fail(f"DA3 src not found at {da3_src}")
        return False
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)
    # DA3 eagerly imports pycolmap and gsplat (heavy C++ deps used only for 3D
    # export, not inference).  Stub them out so the import chain succeeds.
    from unittest.mock import MagicMock
    for _stub in ("pycolmap", "gsplat"):
        if _stub not in sys.modules:
            sys.modules[_stub] = MagicMock()
    try:
        from depth_anything_3.api import DepthAnything3  # noqa: F401
        _ok("depth_anything_3.api importable")
        return True
    except ImportError as e:
        _fail(f"Cannot import depth_anything_3.api — {e}")
        return False


def check_vda_sys_path() -> bool:
    vda_root = os.path.join(_ROOT, "teachers", "Video-Depth-Anything")
    if not os.path.isdir(vda_root):
        _fail(f"Video-Depth-Anything root not found at {vda_root}")
        return False
    if vda_root not in sys.path:
        sys.path.insert(0, vda_root)
    # VDA's video_depth.py does `from utils.util import ...` which collides with
    # FocusMamba's own utils/ package.  Temporarily remove it from the module
    # cache so Python searches sys.path (where VDA's utils/ comes first).
    # Also evict any previously cached (error-state) VDA module entries so
    # Python re-executes the import rather than replaying the cached failure.
    _stash_keys = ["utils", "utils.util"]
    _evict_keys = [k for k in sys.modules if k.startswith("video_depth_anything")]
    _stash = {k: sys.modules.pop(k) for k in _stash_keys if k in sys.modules}
    for k in _evict_keys:
        sys.modules.pop(k, None)
    try:
        from video_depth_anything.video_depth import VideoDepthAnything  # noqa: F401
        _ok("VideoDepthAnything importable")
        return True
    except ImportError as e:
        _fail(f"Cannot import VideoDepthAnything — {e}")
        return False
    finally:
        sys.modules.update(_stash)


def check_depth_pro_sys_path() -> bool:
    dp_src = os.path.join(_ROOT, "teachers", "ml-depth-pro", "src")
    if not os.path.isdir(dp_src):
        _fail(f"ml-depth-pro src not found at {dp_src}")
        return False
    if dp_src not in sys.path:
        sys.path.insert(0, dp_src)
    try:
        from depth_pro import create_model_and_transforms  # noqa: F401
        _ok("depth_pro.create_model_and_transforms importable")
        return True
    except ImportError as e:
        _fail(f"Cannot import depth_pro — {e}  (did you run: uv pip install -e teachers/ml-depth-pro?)")
        return False


def run_checks(dry_run: bool, device: str) -> int:
    """Return number of failures."""
    failures = 0

    # ------------------------------------------------------------------
    print("\n=== Checkpoint files ===")
    checks = [
        ("checkpoints/da3_metric.safetensors", "DA3 weights"),
        ("checkpoints/config.json",            "DA3 config.json"),
        ("checkpoints/depth_pro.pt",           "Depth Pro weights"),
        ("checkpoints/metric_video_depth_anything_vitl.pth", "VideoDepthAnything weights"),
    ]
    for path, label in checks:
        if not check_file(path, label):
            failures += 1

    # ------------------------------------------------------------------
    print("\n=== Teacher module imports ===")
    for fn in (check_da3_sys_path, check_vda_sys_path, check_depth_pro_sys_path):
        if not fn():
            failures += 1

    # ------------------------------------------------------------------
    if dry_run:
        print("\n  (skipping forward-pass test — dry-run mode)")
        return failures

    print(f"\n=== Forward-pass test (device={device}) ===")
    import torch

    # -- Depth Anything V3 --
    try:
        from models.teachers.depth_anything_v3 import DepthAnythingV3Teacher
        t = DepthAnythingV3Teacher(
            checkpoint_path=os.path.join(_ROOT, "checkpoints/da3_metric.safetensors"),
            device=device,
        )
        dummy = torch.rand(1, 3, 4, 64, 64)  # (B, C, T, H, W)
        with torch.no_grad():
            out = t.predict(dummy)
        _ok(f"DA3 predict → {tuple(out.shape)}")
    except Exception as e:
        _fail(f"DA3 forward pass — {e}")
        failures += 1

    # -- Video Depth Anything --
    try:
        from models.teachers.video_teacher import VideoDepthAnythingTeacher
        t = VideoDepthAnythingTeacher(
            checkpoint_path=os.path.join(_ROOT, "checkpoints/metric_video_depth_anything_vitl.pth"),
            device=device,
        )
        dummy = torch.rand(1, 3, 4, 64, 64)
        with torch.no_grad():
            out = t.predict(dummy)
        _ok(f"VideoDepthAnything predict → {tuple(out.shape)}")
    except Exception as e:
        _fail(f"VideoDepthAnything forward pass — {e}")
        failures += 1

    # -- Depth Pro --
    try:
        from models.teachers.depth_pro import DepthProTeacher
        t = DepthProTeacher(
            checkpoint_path=os.path.join(_ROOT, "checkpoints/depth_pro.pt"),
            device=device,
        )
        dummy = torch.rand(1, 3, 4, 64, 64)
        with torch.no_grad():
            out = t.predict(dummy)
        _ok(f"DepthPro predict → {tuple(out.shape)}")
    except Exception as e:
        _fail(f"DepthPro forward pass — {e}")
        failures += 1

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-flight teacher check")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check imports and checkpoint files — no model loading or inference.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for the optional forward-pass test (default: cpu).",
    )
    args = parser.parse_args()

    print("FocusMamba — teacher pre-flight check")
    print(f"Project root : {_ROOT}")
    print(f"Mode         : {'dry-run (imports + files only)' if args.dry_run else f'full (device={args.device})'}")

    failures = run_checks(dry_run=args.dry_run, device=args.device)

    print(f"\n{'='*48}")
    if failures == 0:
        print(f"  All checks passed.")
    else:
        print(f"  {failures} check(s) FAILED — fix before submitting.")
    print("="*48)
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
