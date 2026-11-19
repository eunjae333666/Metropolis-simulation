"""Thermalization and measurement scheduling for Ising simulations."""

from dataclasses import dataclass, field
from operator import index

from convergence import ConvergenceMonitor
from sampler import Sampler


@dataclass
class SimulationHistory:
    """Measurements after burn-in, together with the run's stopping status."""

    sweeps: list[int] = field(default_factory=list)
    energy_per_spin: list[float] = field(default_factory=list)
    magnetization_per_spin: list[float] = field(default_factory=list)
    abs_magnetization_per_spin: list[float] = field(default_factory=list)
    acceptance_rate: list[float] = field(default_factory=list)
    sweeps_completed: int = 0
    converged: bool = False

    def record(self, sweep, sampler: Sampler):
        """Record observables and the cumulative measurement acceptance rate."""
        model = sampler.model
        self.sweeps.append(sweep)
        self.energy_per_spin.append(model.energy_per_spin)
        self.magnetization_per_spin.append(model.magnetization_per_spin)
        self.abs_magnetization_per_spin.append(model.abs_magnetization_per_spin)
        self.acceptance_rate.append(sampler.acceptance_rate)


class SimulationRunner:
    """Advance a sampler in batches and record regularly spaced observations."""

    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def run(
        self, *, burn_in_sweeps=0, max_sweeps=1_000, sample_interval=1,
        monitor: ConvergenceMonitor | None = None,
    ):
        """Run up to max_sweeps after burn-in, sampling on interval boundaries."""
        burn_in_sweeps = index(burn_in_sweeps)
        max_sweeps = index(max_sweeps)
        sample_interval = index(sample_interval)
        if min(burn_in_sweeps, max_sweeps) < 0 or sample_interval < 1:
            raise ValueError("sweep limits must be nonnegative and sample_interval positive")

        self.sampler.sweeps(burn_in_sweeps)
        self.sampler.reset_statistics()
        if monitor is not None:
            monitor.reset()
        history = SimulationHistory(sweeps_completed=burn_in_sweeps)

        # Monte Carlo sweeps measure algorithmic pseudodynamics, not physical time.
        for elapsed in range(sample_interval, max_sweeps + 1, sample_interval):
            self.sampler.sweeps(sample_interval)
            history.sweeps_completed = burn_in_sweeps + elapsed
            history.record(history.sweeps_completed, self.sampler)
            if monitor is not None and monitor.observe(self.sampler.model, self.sampler):
                history.converged = True
                return history

        self.sampler.sweeps(max_sweeps % sample_interval)
        history.sweeps_completed = burn_in_sweeps + max_sweeps
        return history
