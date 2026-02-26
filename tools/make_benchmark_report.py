"""
Make Benchmark Report
========================

Compile evaluation JSONs into a LaTeX table for paper.

Usage:
    python tools/make_benchmark_report.py --results evaluation_report.json --output tables/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def json_to_latex_table(results_path: Path, output_dir: Path) -> str:
    """Convert evaluation JSON to LaTeX table.

    Args:
        results_path: Path to evaluation_report.json.
        output_dir: Directory to save .tex files.

    Returns:
        LaTeX table string.
    """
    raise NotImplementedError(
        "Implement: parse JSON results, format as LaTeX tabular, "
        "highlight best results with \\textbf{}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="tables/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_to_latex_table(Path(args.results), output_dir)


if __name__ == "__main__":
    main()
