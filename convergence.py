"""Window comparisons for detecting apparent stability of sampled observables."""

import math
from dataclasses import dataclass
from operator import index
from statistics import fmean
from typing import Callable

from model import IsingModel
from sampler import Sampler


@dataclass(frozen=True)
class ConvergenceCriterion:
    """Compare an observable's window means using tolerances or a custom rule.

    A custom rule takes (previous_mean, recent_mean) and returns a boolean.
    The default uses abs_tol + rel_tol * abs(previous_mean) as the margin.
    """

    name: str
    observable: Callable[[IsingModel, Sampler], float]
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    custom_check: Callable[[float, float], bool] | None = None

    def __post_init__(self):
        if any(not math.isfinite(tol) or tol < 0 for tol in (self.abs_tol, self.rel_tol)):
            raise ValueError("tolerances must be finite and nonnegative")

    def is_stable(self, previous_mean, recent_mean):
        """Return whether the two window means meet this criterion."""
        if self.custom_check is not None:
            return bool(self.custom_check(previous_mean, recent_mean))
        margin = self.abs_tol + self.rel_tol * abs(previous_mean)
        return abs(recent_mean - previous_mean) <= margin


class ConvergenceMonitor:
    """Check consecutive windows of non-overlapping batches.

    Each batch contains batch_size observations, and each window contains
    window_size batches. At each new batch, all criteria must pass together
    for stable_checks_required consecutive checks.

    This is a heuristic, with no mathematical guarantee of equilibrium.
    At low temperature a metastable configuration can pass the checks.
    """

    def __init__(
        self, criteria, *, batch_size=10, window_size=3,
        stable_checks_required=3, min_batches=6,
    ):
        self.criteria = tuple(criteria)
        names = {criterion.name for criterion in self.criteria}
        if not names or len(names) != len(self.criteria):
            raise ValueError("provide at least one criterion and use distinct names")

        self.batch_size = index(batch_size)
        self.window_size = index(window_size)
        self.stable_checks_required = index(stable_checks_required)
        self.min_batches = index(min_batches)
        if min(self.batch_size, self.window_size, self.stable_checks_required, self.min_batches) < 1:
            raise ValueError("batch and check settings must be positive integers")
        self.reset()

    def reset(self):
        """Clear observations, batch means, and consecutive checks."""
        self.batch_history = {criterion.name: [] for criterion in self.criteria}
        self._pending = {criterion.name: [] for criterion in self.criteria}
        self._pending_count = 0
        self._batch_count = 0
        self.stable_checks = 0

    @property
    def converged(self):
        return self.stable_checks >= self.stable_checks_required

    def observe(self, model: IsingModel, sampler: Sampler):
        """Record one sample and check for stability when a batch is complete."""
        values = {}
        for criterion in self.criteria:
            value = float(criterion.observable(model, sampler))
            if not math.isfinite(value):
                raise ValueError(f"observable {criterion.name!r} must be finite")
            values[criterion.name] = value

        for name, value in values.items():
            self._pending[name].append(value)
        self._pending_count += 1
        if self._pending_count < self.batch_size:
            return self.converged

        for name, batch in self._pending.items():
            self.batch_history[name].append(fmean(batch))
            batch.clear()
        self._pending_count = 0
        self._batch_count += 1

        width = self.window_size
        if self._batch_count < max(self.min_batches, 2 * width):
            return False

        stable = True
        for criterion in self.criteria:
            batches = self.batch_history[criterion.name]
            previous_mean = fmean(batches[-2 * width:-width])
            recent_mean = fmean(batches[-width:])
            if not criterion.is_stable(previous_mean, recent_mean):
                stable = False
        self.stable_checks = self.stable_checks + 1 if stable else 0
        return self.converged
