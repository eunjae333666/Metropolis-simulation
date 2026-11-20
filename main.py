"""Run and plot an example Metropolis simulation."""

from statistics import fmean

import matplotlib.pyplot as plt
from numpy.random import SeedSequence

from convergence import ConvergenceCriterion, ConvergenceMonitor
from model import IsingModel
from plotting import IsingPlotter
from runner import SimulationRunner
from sampler import MetropolisSampler

L = 400
J = 1.0
T = 2.27
INITIALIZATION = "random"
SEED = 42
BURN_IN_SWEEPS = 50000
MAX_SWEEPS = 2_00000
SAMPLE_INTERVAL = 5


def main():
    """Run the examples with the settings above."""
    model_seed, sampler_seed = SeedSequence(SEED).spawn(2)
    model = IsingModel(L, J, INITIALIZATION, seed=model_seed)
    sampler = MetropolisSampler(model, T, seed=sampler_seed)
    monitor = ConvergenceMonitor(
        [
            ConvergenceCriterion(
                "E/N", lambda model, sampler: model.energy_per_spin, rel_tol=0.025,
            ),
            ConvergenceCriterion(
                "|M|/N", lambda model, sampler: model.abs_magnetization_per_spin, abs_tol=0.05,
            ),
        ],
        batch_size=10,
        window_size=3,
        stable_checks_required=3,
        min_batches=8,
    )
    history = SimulationRunner(sampler).run(
        burn_in_sweeps=BURN_IN_SWEEPS, max_sweeps=MAX_SWEEPS,
        sample_interval=SAMPLE_INTERVAL, monitor=monitor,
    )
    stop = "convergence heuristic passed" if history.converged else "sweep budget reached"
    print(f"Completed {history.sweeps_completed} sweeps including burn-in ({stop}).")
    print(f"Final E/N = {model.energy_per_spin:.4f}")
    print(f"Final M/N = {model.magnetization_per_spin:.4f}")
    if history.sweeps:
        print(f"Mean E/N = {fmean(history.energy_per_spin):.4f}")
        print(f"Mean |M|/N = {fmean(history.abs_magnetization_per_spin):.4f}")
    print(f"Measurement acceptance rate = {sampler.acceptance_rate:.2%}")

    plotter = IsingPlotter()
    plotter.plot_lattice(model)
    plotter.plot_energy(history)
    plotter.plot_magnetization(history)
    plt.show()


if __name__ == "__main__":
    main()
