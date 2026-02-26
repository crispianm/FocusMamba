"""
Degradation Sweep Plot
========================

Generate performance-vs-severity curves for the paper (Figure 3 or 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def plot_degradation_sweep(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric_name: str = "abs_rel",
    output_path: Optional[Path] = None,
) -> None:
    """Plot metric vs. degradation severity for each degradation type.

    Args:
        results: Nested dict from per_degradation protocol:
                 {degradation_type: {severity: {metric: value}}}.
        metric_name: Which metric to plot.
        output_path: If provided, save the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(8, 5))

    for deg_type, sev_dict in results.items():
        severities = sorted(sev_dict.keys())
        values = [sev_dict[s].get(metric_name, float("nan")) for s in severities]
        ax.plot(severities, values, marker="o", label=deg_type)

    ax.set_xlabel("Severity Level")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Degradation Robustness: {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
