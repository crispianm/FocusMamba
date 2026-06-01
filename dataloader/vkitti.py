"""
Virtual KITTI 2 (VKITTI 2.0.3) Clip Dataset
============================================

Loads short RGB + dense depth clips from Virtual KITTI 2 for supervised
fine-tuning. VKITTI provides KITTI-like driving scenes (far-range outdoor
depth) with dense ground-truth depth and RGB at native KITTI resolution
(1242x375), which TartanAir's indoor/harbour scenes lack. Adding it closes the
domain gap to the real-KITTI evaluation set.

Dataset structure (after extracting vkitti_2.0.3_rgb.tar / _depth.tar)::

    <root>/
      vkitti_2.0.3_rgb/   (or rgb/ or directly Scene*/)
        Scene01/<variant>/frames/rgb/Camera_0/rgb_00000.jpg
      vkitti_2.0.3_depth/
        Scene01/<variant>/frames/depth/Camera_0/depth_00000.png

Scenes:   Scene01, Scene02, Scene06, Scene18, Scene20
Variants: clone, morning, overcast, sunset, fog, rain, 15-deg-*, 30-deg-*
Cameras:  Camera_0 (left), Camera_1 (right)

Depth encoding: 16-bit PNG, depth in centimetres → metres = value / 100.0.
Sky / far pixels are stored at 65535 (655.35 m) and excluded by the max_depth
validity mask, mirroring TartanAir's sky handling.

Wide VKITTI frames (1242x375) do not fit in 24 GB at 518x518 temporal
attention, so each clip is reduced to a random (train) / centre (val) square
crop of the full image height and resized to ``image_size``. A 375→518 crop is
~1.38x isotropic scaling, matching the 518x1792 evaluation scale (~1.4x), so
per-object depth statistics transfer to the wide-aspect eval.

Returns the same dict contract as :class:`TartanAirV2Dataset` so the two can be
concatenated into a single training set.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from dataloader.degradation import ConfigurableDegradation, LowLightDegradation

try:  # OpenCV gives the correct 16-bit unchanged read for VKITTI depth PNGs.
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


def _decode_vkitti_depth(path: str) -> np.ndarray:
    """Decode a VKITTI 2 depth PNG to float32 metres (uint16 cm / 100)."""
    if _HAS_CV2:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Could not read VKITTI depth: {path}")
    else:  # PIL fallback — 16-bit grayscale PNG
        arr = np.array(Image.open(path), dtype=np.uint16)
    return arr.astype(np.float32) / 100.0


class VKITTIDataset(Dataset):
    """RGB + dense depth clip dataset from Virtual KITTI 2.

    Args mirror :class:`TartanAirV2Dataset` where they overlap. VKITTI-specific:
        scenes:   list of scene names to include (default: all found).
        variants: list of weather/lighting variants (default: ``["clone"]`` —
                  clean daytime; exclude fog/rain for the clean baseline).
        cameras:  list of camera names (default: ``["Camera_0"]``).
    """

    def __init__(
        self,
        root: str,
        num_frames: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        clip_stride: int = 8,
        frame_stride: int = 1,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        max_depth: float = 80.0,
        scenes: Optional[List[str]] = None,
        variants: Optional[List[str]] = None,
        cameras: Optional[List[str]] = None,
        degradation: Optional[LowLightDegradation | ConfigurableDegradation] = None,
        return_clean_and_degraded: bool = False,
        return_clean_reference: bool = True,
        return_degradation_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_frames = num_frames
        self.image_size = tuple(image_size)
        self.clip_stride = clip_stride
        self.frame_stride = frame_stride
        self.max_depth = max_depth
        self.split = split
        self.seed = int(seed)
        self.variants = variants if variants is not None else ["clone"]
        self.cameras = cameras if cameras is not None else ["Camera_0"]
        self.degradation = degradation
        self.return_clean_and_degraded = bool(
            return_clean_and_degraded and degradation is not None
        )
        self.return_clean_reference = bool(
            return_clean_reference and self.return_clean_and_degraded
        )
        self.return_degradation_metadata = bool(
            return_degradation_metadata and self.return_clean_and_degraded
        )

        # --- Locate the rgb / depth roots (handle a few extraction layouts) --
        rgb_root = self._find_modality_root("rgb")
        depth_root = self._find_modality_root("depth")
        if rgb_root is None or depth_root is None:
            raise FileNotFoundError(
                f"Could not locate VKITTI rgb/depth under {root}. "
                f"Expected <root>/vkitti_2.0.3_rgb/Scene*/<variant>/frames/rgb/Camera_*/"
            )
        self.rgb_root = rgb_root
        self.depth_root = depth_root

        # --- Discover sequences: (scene, variant, camera) ---------------------
        all_scenes = sorted(
            p.name for p in rgb_root.iterdir() if p.is_dir() and p.name.startswith("Scene")
        )
        if scenes is not None:
            all_scenes = [s for s in all_scenes if s in scenes]

        sequences: List[Tuple[List[Path], List[Path], str]] = []
        for scene in all_scenes:
            for variant in self.variants:
                for cam in self.cameras:
                    rgb_dir = rgb_root / scene / variant / "frames" / "rgb" / cam
                    depth_dir = depth_root / scene / variant / "frames" / "depth" / cam
                    if not (rgb_dir.is_dir() and depth_dir.is_dir()):
                        continue
                    rgb_files = sorted(rgb_dir.glob("rgb_*.jpg"))
                    depth_files = sorted(depth_dir.glob("depth_*.png"))
                    n = min(len(rgb_files), len(depth_files))
                    if n < (num_frames - 1) * frame_stride + 1:
                        continue
                    seq_id = f"{scene}/{variant}/{cam}"
                    sequences.append((rgb_files[:n], depth_files[:n], seq_id))

        if not sequences:
            raise FileNotFoundError(
                f"No VKITTI sequences found under {root} for scenes={scenes} "
                f"variants={self.variants} cameras={self.cameras}."
            )

        # --- Deterministic train/val split by sequence ------------------------
        rng = random.Random(seed)
        shuffled = sequences[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        chosen = shuffled[:n_val] if split == "val" else shuffled[n_val:]

        # --- Build clip index -------------------------------------------------
        self.clips: List[Tuple[List[Path], List[Path], str, int]] = []
        clip_len = (num_frames - 1) * frame_stride + 1
        for rgb_files, depth_files, seq_id in chosen:
            n = len(rgb_files)
            for start in range(0, n - clip_len + 1, clip_stride):
                self.clips.append((rgb_files, depth_files, seq_id, start))

        self._to_tensor = transforms.ToTensor()

    def _find_modality_root(self, modality: str) -> Optional[Path]:
        """Find the directory that contains Scene*/ for a given modality."""
        candidates = [
            self.root / f"vkitti_2.0.3_{modality}",
            self.root / modality,
            self.root,
        ]
        for c in candidates:
            if c.is_dir() and any(
                p.name.startswith("Scene") for p in c.iterdir() if p.is_dir()
            ):
                # Verify it is the right modality by checking the frames path.
                for scene in c.iterdir():
                    if not scene.is_dir() or not scene.name.startswith("Scene"):
                        continue
                    if list(scene.glob(f"*/frames/{modality}/Camera_*")):
                        return c
                    break
        return None

    def __len__(self) -> int:
        return len(self.clips)

    def _square_crop_params(self, h: int, w: int, idx: int) -> int:
        """Return horizontal offset for an h×h square crop."""
        max_off = max(0, w - h)
        if max_off == 0:
            return 0
        if self.split == "train":
            return random.randint(0, max_off)
        # Deterministic centre crop for val.
        return max_off // 2

    def __getitem__(self, idx: int) -> dict:
        rgb_files, depth_files, seq_id, start = self.clips[idx]
        indices = [start + t * self.frame_stride for t in range(self.num_frames)]

        imgs, depths, masks = [], [], []
        x_off = None
        for i in indices:
            img = Image.open(rgb_files[i]).convert("RGB")
            w, h = img.size  # PIL: (W, H)
            if x_off is None:  # shared crop offset across the whole clip
                x_off = self._square_crop_params(h, w, idx)
            t = self._to_tensor(img)  # (3, H, W)

            depth_raw = _decode_vkitti_depth(str(depth_files[i]))  # (H, W) metres
            valid = np.isfinite(depth_raw) & (depth_raw > 0) & (depth_raw < self.max_depth)
            depth = np.clip(depth_raw, 0.0, self.max_depth)
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()
            mask_t = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)

            # Square crop (full height, shared horizontal offset).
            side = h
            t = t[:, :, x_off : x_off + side]
            depth_t = depth_t[:, :, x_off : x_off + side]
            mask_t = mask_t[:, :, x_off : x_off + side]
            imgs.append(t)
            depths.append(depth_t)
            masks.append(mask_t)

        frames = torch.stack(imgs, dim=1)   # (3, T, H, W)
        depth = torch.stack(depths, dim=1)  # (1, T, H, W)
        mask = torch.stack(masks, dim=1)    # (1, T, H, W)

        H, W = self.image_size
        if frames.shape[-2:] != (H, W):
            frames = F.interpolate(
                frames.permute(1, 0, 2, 3), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(1, 0, 2, 3)
            depth = F.interpolate(
                depth.permute(1, 0, 2, 3), size=(H, W), mode="nearest",
            ).permute(1, 0, 2, 3)
            mask = F.interpolate(
                mask.permute(1, 0, 2, 3), size=(H, W), mode="nearest",
            ).permute(1, 0, 2, 3)

        result: Dict = {
            "depth": depth,
            "mask": mask,
            "video_id": f"vkitti/{seq_id}",
            "start_frame": start,
        }
        if self.return_clean_and_degraded:
            clip_tchw = frames.permute(1, 0, 2, 3).contiguous()
            rng = None
            if self.split != "train":
                rng = np.random.RandomState(self.seed + idx)
            degradation_output = self.degradation(
                clip_tchw, rng=rng, return_metadata=self.return_degradation_metadata,
            )
            degradation_params = None
            degradation_summary = None
            if self.return_degradation_metadata:
                degraded_clip, degradation_params, degradation_summary = degradation_output
            else:
                degraded_clip = degradation_output
            degraded_frames = degraded_clip.permute(1, 0, 2, 3).contiguous()
            if self.return_clean_reference:
                result["clean_frames"] = frames
            result["degraded_frames"] = degraded_frames
            if degradation_params is not None:
                result["degradation_params"] = degradation_params.as_dict()
            if degradation_summary is not None:
                result["degradation_summary"] = degradation_summary
        else:
            result["frames"] = frames

        return result
