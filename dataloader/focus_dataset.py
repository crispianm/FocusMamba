"""
FocusMamba Dataset & DataLoader
================================

Expected directory structure under `data_root`:
    data/
        bear/
            split.txt                    # train|val|test (optional)
            metadata.json                # optional, includes lens presets
            frames_sharp/frame_0000.png
            depth_maps/depth_0000.npz    # key: depth
            focus_maps/
                set_00/focus_0000.npz    # optional, key: focus_map
                set_01/focus_0000.npz
        bear1/
            ...

If precomputed focus maps are unavailable, focus maps are synthesized on-the-fly
from depth via the CoC model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataloader.degradation import LowLightDegradation


# ---------------------------------------------------------------------------
# Utility: deterministic scene-level split
# ---------------------------------------------------------------------------

def _split_scenes(
    scene_dirs: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Split scene directories deterministically into train / val / test."""
    rng = np.random.RandomState(seed)
    scenes = sorted(scene_dirs, key=lambda p: p.name)
    indices = rng.permutation(len(scenes))
    n_train = int(len(scenes) * train_ratio)
    n_val = int(len(scenes) * val_ratio)
    return {
        "train": [scenes[i] for i in indices[:n_train]],
        "val": [scenes[i] for i in indices[n_train : n_train + n_val]],
        "test": [scenes[i] for i in indices[n_train + n_val :]],
    }


# ---------------------------------------------------------------------------
# Circle-of-Confusion helpers
# ---------------------------------------------------------------------------

def _sample_lens_params(
    rng: np.random.RandomState,
    focal_length_range: Tuple[float, float] = (24.0, 85.0),
    aperture_range: Tuple[float, float] = (1.4, 8.0),
) -> Tuple[float, float]:
    """Sample focal length (uniform) and aperture (log-uniform)."""
    f = rng.uniform(*focal_length_range)
    log_lo, log_hi = np.log(aperture_range[0]), np.log(aperture_range[1])
    N = float(np.exp(rng.uniform(log_lo, log_hi)))
    return f, N


def _sample_focus_distance(
    depth_map: np.ndarray,
    rng: np.random.RandomState,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> float:
    """Sample a focus distance from a Beta(alpha, beta) mapped to [d_min, d_max]."""
    valid = depth_map[depth_map > 0]
    if len(valid) == 0:
        return 1.0  # fallback
    d_min, d_max = float(valid.min()), float(valid.max())
    if d_max - d_min < 1e-6:
        return d_min
    u = rng.beta(alpha, beta)
    return d_min + u * (d_max - d_min)


def compute_focus_map(
    depth: np.ndarray,
    focal_length: float,
    aperture: float,
    s_focus: float,
    max_coc: float,
) -> np.ndarray:
    """Compute a soft focus map using a physically-based Circle of Confusion model.

    CoC(d) = (f^2 / (N * S_focus)) * |d - S_focus| / d

    Returns:
        focus_map in [0, 1] where 1 = perfectly in-focus.
    """
    d = depth.astype(np.float64).copy()
    d[d < 1e-8] = 1e-8  # avoid division by zero
    coc = (focal_length ** 2 / (aperture * s_focus)) * np.abs(d - s_focus) / d
    coc_norm = coc / max_coc
    focus_map = 1.0 - np.clip(coc_norm, 0.0, 1.0)
    return focus_map.astype(np.float32)


def _read_frame_png(path: Path) -> np.ndarray:
    """Load an RGB frame PNG as uint8 ndarray of shape (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_depth_npz(path: Path) -> np.ndarray:
    """Load depth from an NPZ file and return float32 ndarray (H, W)."""
    with np.load(str(path)) as npz:
        if "depth" in npz:
            depth = npz["depth"]
        elif len(npz.files) > 0:
            depth = npz[npz.files[0]]
        else:
            raise ValueError(f"Empty depth npz file: {path}")
    return depth.astype(np.float32)


def _read_focus_npz(path: Path) -> np.ndarray:
    """Load focus map from an NPZ file and return float32 ndarray (H, W)."""
    with np.load(str(path)) as npz:
        if "focus_map" in npz:
            focus = npz["focus_map"]
        elif len(npz.files) > 0:
            focus = npz[npz.files[0]]
        else:
            raise ValueError(f"Empty focus npz file: {path}")
    return focus.astype(np.float32)



# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

class _TrainAugmentation:
    """Random horizontal flip, random crop, temporal jitter (±1 frame)."""

    def __init__(self, crop_h: int, crop_w: int, rng: np.random.RandomState):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.rng = rng

    def __call__(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        focus_maps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        T, H, W, C = frames.shape

        # --- random horizontal flip ---
        if self.rng.rand() > 0.5:
            frames = frames[:, :, ::-1, :].copy()
            depth_maps = depth_maps[:, :, ::-1].copy()
            focus_maps = focus_maps[:, :, ::-1].copy()

        # --- random crop ---
        if H > self.crop_h:
            y0 = self.rng.randint(0, H - self.crop_h + 1)
        else:
            y0 = 0
        if W > self.crop_w:
            x0 = self.rng.randint(0, W - self.crop_w + 1)
        else:
            x0 = 0
        frames = frames[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w, :]
        depth_maps = depth_maps[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w]
        focus_maps = focus_maps[:, y0 : y0 + self.crop_h, x0 : x0 + self.crop_w]

        return frames, depth_maps, focus_maps


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FocusDataset(Dataset):
    """Sliding-window video focus-map dataset with depth-guided CoC generation.

    Each sample is a clip of ``window_length`` consecutive frames.
    """

    def __init__(
        self,
        scene_dirs: List[Path],
        window_length: int = 8,
        stride: int = 4,
        crop_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        # CoC parameters
        max_coc: float = 0.05,
        focal_length_range: Tuple[float, float] = (24.0, 85.0),
        aperture_range: Tuple[float, float] = (1.4, 8.0),
        beta_alpha: float = 2.0,
        beta_beta: float = 2.0,
        seed: int = 42,
        # Low-light degradation
        apply_lowlight: bool = False,
        lowlight_degradation: Optional[LowLightDegradation] = None,
    ):
        super().__init__()
        self.window_length = window_length
        self.stride = stride
        self.crop_size = crop_size
        self.augment = augment
        self.max_coc = max_coc
        self.focal_length_range = focal_length_range
        self.aperture_range = aperture_range
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        # Low-light degradation pipeline
        self.apply_lowlight = apply_lowlight
        self.lowlight_degradation = lowlight_degradation

        # Build scene records and clip index.
        self._scene_records: Dict[str, Dict[str, Any]] = {}
        self._clips: List[Tuple[str, int]] = []
        for sd in sorted(scene_dirs, key=lambda p: p.name):
            frames_dir = sd / "frames_sharp"
            depth_dir = sd / "depth_maps"
            if not frames_dir.exists() or not depth_dir.exists():
                continue

            frame_paths = sorted(frames_dir.glob("frame_*.png"))
            depth_paths = sorted(depth_dir.glob("depth_*.npz"))
            n_frames = min(len(frame_paths), len(depth_paths))
            if n_frames < window_length:
                continue

            frame_paths = frame_paths[:n_frames]
            depth_paths = depth_paths[:n_frames]

            focus_sets: Dict[str, List[Path]] = {}
            focus_root = sd / "focus_maps"
            if focus_root.exists() and focus_root.is_dir():
                for set_dir in sorted([d for d in focus_root.iterdir() if d.is_dir()]):
                    focus_paths = sorted(set_dir.glob("focus_*.npz"))
                    if len(focus_paths) >= n_frames:
                        focus_sets[set_dir.name] = focus_paths[:n_frames]

            scene_meta: Dict[str, Any] = {}
            metadata_path = sd / "metadata.json"
            if metadata_path.exists():
                try:
                    scene_meta = json.loads(metadata_path.read_text())
                except (json.JSONDecodeError, OSError):
                    scene_meta = {}

            self._scene_records[sd.name] = {
                "scene_dir": sd,
                "n_frames": n_frames,
                "frames": frame_paths,
                "depths": depth_paths,
                "focus_sets": focus_sets,
                "metadata": scene_meta,
            }

            for start in range(0, n_frames - window_length + 1, stride):
                self._clips.append((sd.name, start))

    # ---- helpers ----
    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_name, start = self._clips[idx]
        scene_record = self._scene_records[scene_name]
        T = self.window_length

        # Temporal jitter for training (±1)
        if self.augment:
            jitter = int(self.rng.randint(-1, 2))  # -1, 0, or 1
        else:
            jitter = 0

        n_frames_total = int(scene_record["n_frames"])

        s = max(0, min(start + jitter, n_frames_total - T))
        e = s + T

        frame_paths: List[Path] = scene_record["frames"][s:e]
        depth_paths: List[Path] = scene_record["depths"][s:e]
        frames = np.stack([_read_frame_png(p) for p in frame_paths], axis=0)
        depth_maps = np.stack([_read_depth_npz(p) for p in depth_paths], axis=0)

        focus_sets: Dict[str, List[Path]] = scene_record["focus_sets"]
        selected_focus_set = None
        f_len = None
        aperture = None
        s_focus = None
        max_coc_used = None

        if focus_sets:
            eligible_sets = [
                set_name for set_name, paths in focus_sets.items() if len(paths) >= e
            ]
            if eligible_sets:
                selected_focus_set = str(eligible_sets[int(self.rng.randint(0, len(eligible_sets)))])
                focus_paths = focus_sets[selected_focus_set][s:e]
                focus_maps = np.stack([_read_focus_npz(p) for p in focus_paths], axis=0)

                scene_meta = scene_record.get("metadata", {})
                if isinstance(scene_meta, dict) and isinstance(scene_meta.get("sets"), list):
                    try:
                        set_idx = int(selected_focus_set.split("_")[-1])
                        if 0 <= set_idx < len(scene_meta["sets"]):
                            preset = scene_meta["sets"][set_idx]
                            f_len = preset.get("f")
                            aperture = preset.get("N")
                            s_focus = preset.get("S_focus")
                            max_coc_used = preset.get("max_coc")
                    except (ValueError, AttributeError):
                        pass
            else:
                focus_maps = None
        else:
            focus_maps = None

        if focus_maps is None:
            f_len, aperture = _sample_lens_params(
                self.rng, self.focal_length_range, self.aperture_range
            )
            s_focus = _sample_focus_distance(
                depth_maps.mean(axis=0), self.rng, self.beta_alpha, self.beta_beta
            )
            max_coc_used = self.max_coc
            focus_maps = np.stack(
                [
                    compute_focus_map(depth_maps[t], f_len, aperture, s_focus, self.max_coc)
                    for t in range(T)
                ],
                axis=0,
            )

        # --- Augmentation ---
        if self.augment:
            aug = _TrainAugmentation(self.crop_size[0], self.crop_size[1], self.rng)
            frames, depth_maps, focus_maps = aug(frames, depth_maps, focus_maps)
        else:
            # Centre crop for val/test
            H, W = frames.shape[1], frames.shape[2]
            ch, cw = self.crop_size
            y0 = max(0, (H - ch) // 2)
            x0 = max(0, (W - cw) // 2)
            frames = frames[:, y0 : y0 + ch, x0 : x0 + cw, :]
            depth_maps = depth_maps[:, y0 : y0 + ch, x0 : x0 + cw]
            focus_maps = focus_maps[:, y0 : y0 + ch, x0 : x0 + cw]


        # --- Convert to tensors (C, T, H, W) ---
        frames_t = (
            torch.from_numpy(frames.copy())
            .permute(3, 0, 1, 2)
            .float()
            .div(255.0)
        )  # (3, T, H, W)

        # --- Low-light degradation (applied to normalised [0,1] frames) ---
        if self.apply_lowlight and self.lowlight_degradation is not None:
            # Convert (C, T, H, W) -> (T, C, H, W) for degradation module
            frames_for_deg = frames_t.permute(1, 0, 2, 3).contiguous()  # (T, C, H, W)
            frames_for_deg = self.lowlight_degradation(frames_for_deg, rng=self.rng)
            frames_t = frames_for_deg.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

        focus_t = torch.from_numpy(focus_maps.copy()).unsqueeze(0).float()   # (1, T, H, W)
        depth_t = torch.from_numpy(depth_maps.copy()).unsqueeze(0).float()   # (1, T, H, W)

        return {
            "frames": frames_t,
            "focus_maps": focus_t,
            "depth_maps": depth_t,
            "metadata": {
                "scene": scene_name,
                "start_frame": s,
                "focal_length": f_len,
                "aperture": aperture,
                "s_focus": s_focus,
                "focus_set": selected_focus_set,
                "max_coc": max_coc_used,
            },
        }


def focus_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that stacks tensors and collects metadata as a list."""
    return {
        "frames": torch.stack([b["frames"] for b in batch]),        # (B,C,T,H,W)
        "focus_maps": torch.stack([b["focus_maps"] for b in batch]),# (B,1,T,H,W)
        "depth_maps": torch.stack([b["depth_maps"] for b in batch]),# (B,1,T,H,W)
        "metadata": [b["metadata"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker's numpy RNG independently.

    When PyTorch forks a worker process the dataset's ``self.rng`` is a copy
    of the parent's state, so ALL workers would produce identical random
    samples (same augmentation, same lens params).  We re-seed each worker
    with a unique value derived from the process-level base seed to avoid this.
    """
    import numpy as np
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    # The dataset stores ``seed`` at construction time; offset by worker_id.
    base_seed = getattr(dataset, "seed", 42)
    new_seed = (base_seed + worker_id * 1000) % (2 ** 31)
    dataset.rng = np.random.RandomState(new_seed)


def build_dataloaders(
    data_root: str,
    batch_size: int = 4,
    window_length: int = 8,
    stride: int = 4,
    crop_size: Tuple[int, int] = (256, 256),
    max_coc: float = 0.05,
    focal_length_range: Tuple[float, float] = (24.0, 85.0),
    aperture_range: Tuple[float, float] = (1.4, 8.0),
    beta_alpha: float = 2.0,
    beta_beta: float = 2.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 4,
    # Low-light degradation
    lowlight_degradation: Optional[LowLightDegradation] = None,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders from a data root directory.

    If *lowlight_degradation* is provided, it is applied to training frames
    during ``__getitem__``.  Validation and test sets receive the *same*
    degradation so that metrics reflect noisy-input performance.
    """
    root = Path(data_root)
    scene_dirs = sorted(
        [
            d
            for d in root.iterdir()
            if d.is_dir() and (d / "frames_sharp").exists() and (d / "depth_maps").exists()
        ]
    )

    split_from_file: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    all_have_valid_split = True
    for scene_dir in scene_dirs:
        split_path = scene_dir / "split.txt"
        if not split_path.exists():
            all_have_valid_split = False
            break
        split_name = split_path.read_text().strip().lower()
        if split_name not in split_from_file:
            all_have_valid_split = False
            break
        split_from_file[split_name].append(scene_dir)

    if all_have_valid_split:
        splits = split_from_file
    else:
        splits = _split_scenes(scene_dirs, train_ratio, val_ratio, split_seed)

    apply_ll = lowlight_degradation is not None

    loaders = {}
    for split_name, dirs in splits.items():
        ds = FocusDataset(
            scene_dirs=dirs,
            window_length=window_length,
            stride=stride,
            crop_size=crop_size,
            augment=(split_name == "train"),
            max_coc=max_coc,
            focal_length_range=tuple(focal_length_range),
            aperture_range=tuple(aperture_range),
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            seed=split_seed,
            apply_lowlight=apply_ll,
            lowlight_degradation=lowlight_degradation,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            collate_fn=focus_collate_fn,
            pin_memory=True,
            drop_last=(split_name == "train"),
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            persistent_workers=True
        )
    return loaders



class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        
        with torch.cuda.stream(self.stream):
            for k in ['frames', 'focus_maps', 'depth_maps']:
                self.next_batch[k] = self.next_batch[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch