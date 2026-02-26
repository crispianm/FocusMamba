"""
Benchmark Callback — Runs Eval Benchmark at Intervals
=======================================================

Runs the full evaluation benchmark every N epochs and logs metrics
(AbsRel, δ1, RMSE, etc.) to TensorBoard.
"""

from __future__ import annotations


class BenchmarkCallback:
    """Runs evaluation benchmark periodically during training.

    TODO: Implement by connecting to evaluation/benchmark.py.

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
        """Run benchmark and return results dict.

        TODO: Implement by calling evaluation/benchmark.py.
        """
        raise NotImplementedError("Connect to evaluation/benchmark.py")
