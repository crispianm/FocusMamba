"""
WildDataset — Custom Smartphone Focal Sweep Loader
====================================================

Parses uncalibrated or manually captured focal sweeps from smartphone video
for zero-shot real-world evaluation.

Expected directory structure::

    wild_data/
        scene_001/
            video.mp4           # OR a folder of frames (frame_0000.png ...)
            labels.json         # ground-truth focus distance per frame
            metadata.json       # optional: camera info, focal length, etc.
        scene_002/
            ...

``labels.json`` format::

    {
        "frames": [
            {"index": 0, "focus_distance": 1.2},
            {"index": 1, "focus_distance": 1.3},
            ...
        ],
        "unit": "meters"
    }

If the scene directory contains a ``video.mp4`` (or ``.mov``, ``.avi``),
frames are extracted on-the-fly via OpenCV.  If a ``frames/`` or
``frames_sharp/`` sub-directory exists, PNG frames are loaded directly.

Frames are resized to match the training resolution (default 256×256) and
returned as (C, T, H, W) tensors in [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _find_video_file(scene_dir: Path) -> Optional[Path]:
    """Return the first video file found in *scene_dir*, or None."""
    for f in sorted(scene_dir.iterdir()):
        if f.suffix.lower() in _VIDEO_EXTENSIONS:
            return f
    return None


def _extract_frames_from_video(
    video_path: Path,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """Extract frames from a video file using OpenCV.

    Returns a list of (H, W, 3) uint8 numpy arrays (RGB).
    """
    if not _HAS_CV2:
        raise RuntimeError(
            "OpenCV (cv2) is required to load video files. "
            "Install with: pip install opencv-python"
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads BGR; convert to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _load_frames_from_dir(frame_dir: Path) -> List[np.ndarray]:
    """Load PNG/JPG frames from a directory, sorted by name."""
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        [p for p in frame_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]


def _resize_frame(frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a (H, W, 3) frame to (target_H, target_W, 3) using PIL."""
    h, w = target_hw
    img = Image.fromarray(frame).resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WildDataset(Dataset):
    """Dataset for uncalibrated smartphone focal sweeps.

    Each sample is a full-scene clip (or a sliding window of ``window_length``
    frames, depending on the scene length and stride).

    Args:
        data_root: Root directory containing scene sub-directories.
        window_length: Number of frames per clip.
        stride: Sliding window stride.
        resolution: (H, W) to resize frames to.
        max_scenes: Optional limit on number of scenes to load.
    """

    def __init__(
        self,
        data_root: str | Path,
        window_length: int = 49,
        stride: int = 16,
        resolution: Tuple[int, int] = (256, 256),
        max_scenes: Optional[int] = None,
    ):
        super().__init__()
        self.root = Path(data_root)
        self.window_length = window_length
        self.stride = stride
        self.resolution = resolution

        # Discover scenes
        self._scenes: List[Dict[str, Any]] = []
        self._clips: List[Tuple[int, int]] = []  # (scene_idx, start_frame)

        scene_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if max_scenes is not None:
            scene_dirs = scene_dirs[:max_scenes]

        for scene_dir in scene_dirs:
            scene_record = self._load_scene(scene_dir)
            if scene_record is None:
                continue
            scene_idx = len(self._scenes)
            self._scenes.append(scene_record)

            n = scene_record["n_frames"]
            for start in range(0, max(1, n - window_length + 1), stride):
                self._clips.append((scene_idx, start))

    def _load_scene(self, scene_dir: Path) -> Optional[Dict[str, Any]]:
        """Load a single scene's frames and labels."""
        # Try loading frames from directory first
        frames = None
        for subdir_name in ("frames_sharp", "frames"):
            subdir = scene_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                frames = _load_frames_from_dir(subdir)
                break

        # Fall back to video extraction
        if frames is None or len(frames) == 0:
            video_path = _find_video_file(scene_dir)
            if video_path is not None:
                try:
                    frames = _extract_frames_from_video(video_path)
                except Exception as e:
                    print(f"WARNING: Could not extract frames from {video_path}: {e}")
                    return None
            else:
                return None

        if len(frames) < self.window_length:
            return None

        # Resize all frames
        frames = [_resize_frame(f, self.resolution) for f in frames]

        # Load labels (focus distances)
        labels_path = scene_dir / "labels.json"
        focus_distances = None
        if labels_path.exists():
            try:
                with open(labels_path) as f:
                    labels_data = json.load(f)
                focus_distances = [
                    entry.get("focus_distance", None)
                    for entry in labels_data.get("frames", [])
                ]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: Could not parse labels for {scene_dir.name}: {e}")

        # Load optional metadata
        meta_path = scene_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "name": scene_dir.name,
            "frames": frames,  # list of (H, W, 3) uint8 arrays
            "n_frames": len(frames),
            "focus_distances": focus_distances,
            "metadata": metadata,
        }

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_idx, start = self._clips[idx]
        scene = self._scenes[scene_idx]
        T = self.window_length
        end = min(start + T, scene["n_frames"])
        actual_T = end - start

        # Stack frames
        frames_np = np.stack(scene["frames"][start:end], axis=0)  # (T, H, W, 3)

        # Convert to tensor (C, T, H, W)
        frames_t = (
            torch.from_numpy(frames_np.copy())
            .permute(3, 0, 1, 2)
            .float()
            .div(255.0)
        )

        # Focus distances for this clip (if available)
        focus_dists = None
        if scene["focus_distances"] is not None:
            focus_dists = scene["focus_distances"][start:end]

        return {
            "frames": frames_t,
            "metadata": {
                "scene": scene["name"],
                "start_frame": start,
                "n_frames": actual_T,
                "focus_distances": focus_dists,
            },
        }


def wild_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for WildDataset — stacks frames, collects metadata."""
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }


def build_wild_dataloader(
    data_root: str | Path,
    batch_size: int = 1,
    window_length: int = 49,
    stride: int = 16,
    resolution: Tuple[int, int] = (256, 256),
    num_workers: int = 0,
    max_scenes: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader for wild smartphone focal sweeps."""
    ds = WildDataset(
        data_root=data_root,
        window_length=window_length,
        stride=stride,
        resolution=resolution,
        max_scenes=max_scenes,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=wild_collate_fn,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("Testing WildDataset with synthetic scene...")

    # Create a temporary scene with fake frames
    with tempfile.TemporaryDirectory() as tmpdir:
        scene_dir = Path(tmpdir) / "test_scene"
        frames_dir = scene_dir / "frames"
        frames_dir.mkdir(parents=True)

        T = 16
        H, W = 128, 128
        for i in range(T):
            img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
            Image.fromarray(img).save(frames_dir / f"frame_{i:04d}.png")

        # Create labels
        labels = {
            "frames": [{"index": i, "focus_distance": 1.0 + 0.1 * i} for i in range(T)],
            "unit": "meters",
        }
        with open(scene_dir / "labels.json", "w") as f:
            json.dump(labels, f)

        ds = WildDataset(
            data_root=tmpdir,
            window_length=8,
            stride=4,
            resolution=(64, 64),
        )

        print(f"  Scenes: {len(ds._scenes)}")
        print(f"  Clips:  {len(ds)}")

        sample = ds[0]
        print(f"  Frame tensor shape: {sample['frames'].shape}")
        print(f"  Scene: {sample['metadata']['scene']}")
        print(f"  Focus distances: {sample['metadata']['focus_distances']}")

        assert sample["frames"].shape == (3, 8, 64, 64)
        assert sample["frames"].min() >= 0 and sample["frames"].max() <= 1
        print("WildDataset test passed!")
