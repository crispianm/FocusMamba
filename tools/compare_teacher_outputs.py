"""
Compare Teacher Outputs
=========================

Side-by-side teacher disagreement analysis.
Useful for understanding where teachers agree (high confidence)
vs. disagree (potential training signal issues).

Usage:
    python tools/compare_teacher_outputs.py --config configs/data/teachers.yaml --input video.mp4
"""

from __future__ import annotations

import argparse


def compare_teachers(config_path: str, input_path: str, output_dir: str = "teacher_comparison") -> None:
    """Load all teachers and compare their depth outputs on sample frames.

    Generates:
        - Per-teacher colourised depth maps
        - Ensemble mean + variance maps
        - Disagreement analysis (where variance is highest)
    """
    raise NotImplementedError(
        "Implement: load teachers, run on sample frames, compute variance, "
        "generate side-by-side visualisation."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="teacher_comparison")
    args = parser.parse_args()

    compare_teachers(args.config, args.input, args.output_dir)


if __name__ == "__main__":
    main()
