"""
Benchmark Callback — Runs Eval Benchmark at Intervals
=======================================================

Runs the full evaluation benchmark every N epochs and logs metrics
(AbsRel, δ1, RMSE, etc.) to TensorBoard.
"""

from __future__ import annotations


class BenchmarkCallback:
    """Runs evaluation benchmark periodically during training.

    Args:
        eval_every_n_epochs: Run benchmark every N epochs.
        benchmark_config: Config for the benchmark suite.
    """

    def __init__(self, eval_every_n_epochs: int = 20, benchmark_config: dict = None):
        self.eval_every_n_epochs = eval_every_n_epochs
        self.benchmark_config = benchmark_config or {}

    def should_run(self, epoch: int) -> bool:
        return (epoch + 1) % self.eval_every_n_epochs == 0

    def run(self, model, writer, epoch: int) -> dict:
        """Run benchmark and return results dict."""
        from pathlib import Path

        import torch

        from evaluation.benchmark import run_benchmark

        device = (
            next(model.parameters()).device
            if any(True for _ in model.parameters())
            else torch.device("cpu")
        )
        output_dir = Path(self.benchmark_config.get("output_dir", "benchmark_reports"))
        output_path = output_dir / f"epoch_{epoch + 1:04d}.json"
        results = run_benchmark(
            model=model,
            config=self.benchmark_config,
            device=device,
            output_path=output_path,
        )
        if writer is not None:
            for protocol_name, protocol_result in results.get("protocols", {}).items():
                if isinstance(protocol_result, dict):
                    for key, value in protocol_result.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                f"benchmark/{protocol_name}/{key}", value, epoch
                            )
        return results
