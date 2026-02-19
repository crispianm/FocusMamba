#!/usr/bin/env python3
"""Inspect Circle-of-Confusion distribution across a sample of scenes.

Usage:
    python -m dataloader.inspect_coc_distribution --data_root ./data --n_scenes 20

Produces a histogram of raw CoC values to help calibrate ``max_coc`` in
config.yaml (set it to approximately the 95th percentile of the distribution).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataloader.focus_dataset import _sample_focus_distance, _sample_lens_params


def collect_coc_values(
    data_root: Path,
    n_scenes: int = 20,
    frames_per_scene: int = 5,
    focal_length_range: tuple = (24.0, 85.0),
    aperture_range: tuple = (1.4, 8.0),
    seed: int = 42,
) -> np.ndarray:
    """Collect raw CoC values from a subset of scenes."""
    rng = np.random.RandomState(seed)
    scene_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if len(scene_dirs) == 0:
        raise FileNotFoundError(f"No scene directories found in {data_root}")

    chosen = rng.choice(len(scene_dirs), size=min(n_scenes, len(scene_dirs)), replace=False)
    all_coc: list[np.ndarray] = []

    for idx in chosen:
        sd = scene_dirs[idx]
        depth_path = sd / "depth_maps.npy"
        if not depth_path.exists():
            continue
        depth_all = np.load(str(depth_path), mmap_mode="r")
        n_frames = depth_all.shape[0]
        frame_indices = rng.choice(n_frames, size=min(frames_per_scene, n_frames), replace=False)

        for fi in frame_indices:
            depth = np.array(depth_all[fi], dtype=np.float64)
            depth[depth < 1e-8] = 1e-8

            f_len, aperture = _sample_lens_params(rng, focal_length_range, aperture_range)
            s_focus = _sample_focus_distance(depth, rng)

            coc = (f_len ** 2 / (aperture * s_focus)) * np.abs(depth - s_focus) / depth
            all_coc.append(coc.ravel())

    return np.concatenate(all_coc)


def main():
    parser = argparse.ArgumentParser(description="Inspect CoC distribution")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--n_scenes", type=int, default=20)
    parser.add_argument("--frames_per_scene", type=int, default=5)
    parser.add_argument("--output", type=str, default="coc_histogram.png")
    args = parser.parse_args()

    coc_values = collect_coc_values(
        Path(args.data_root),
        n_scenes=args.n_scenes,
        frames_per_scene=args.frames_per_scene,
    )

    p50 = np.percentile(coc_values, 50)
    p95 = np.percentile(coc_values, 95)
    p99 = np.percentile(coc_values, 99)

    print(f"CoC statistics (n={len(coc_values):,} pixels):")
    print(f"  Mean:   {coc_values.mean():.6f}")
    print(f"  Median: {p50:.6f}")
    print(f"  P95:    {p95:.6f}  <-- recommended max_coc")
    print(f"  P99:    {p99:.6f}")
    print(f"  Max:    {coc_values.max():.6f}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(coc_values, bins=200, range=(0, p99 * 1.2), density=True, alpha=0.7)
    ax.axvline(p95, color="r", linestyle="--", label=f"P95 = {p95:.4f}")
    ax.axvline(p99, color="orange", linestyle="--", label=f"P99 = {p99:.4f}")
    ax.set_xlabel("Raw CoC value")
    ax.set_ylabel("Density")
    ax.set_title("Circle of Confusion Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Histogram saved to {args.output}")


if __name__ == "__main__":
    main()
