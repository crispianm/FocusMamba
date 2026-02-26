"""
Visualise Degradation Types
==============================

Generate a paper figure showing all degradation types at multiple severities.

Usage:
    python tools/visualise_degradation.py --input sample_frame.png --output figs/degradation_gallery.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path


def generate_degradation_gallery(
    input_image: Path,
    output_path: Path,
    degradation_types: list[str] = None,
    severities: list[float] = None,
) -> None:
    """Generate a grid figure showing degradation types × severity levels.

    Args:
        input_image: Path to a clean sample frame.
        output_path: Path to save the figure.
        degradation_types: List of degradation types to show.
        severities: List of severity values [0, 1] to demonstrate.
    """
    raise NotImplementedError(
        "Implement: load image, apply each degradation type at each severity, "
        "arrange in matplotlib grid, save figure."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="figs/degradation_gallery.pdf")
    args = parser.parse_args()

    generate_degradation_gallery(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
