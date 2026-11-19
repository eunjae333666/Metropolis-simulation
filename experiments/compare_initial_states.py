"""Compare ordered and random starts using independent Metropolis runs."""

import argparse
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from operator import index
from pathlib import Path

import numpy as np

from model import IsingModel
from runner import SimulationRunner
from sampler import MetropolisSampler

DEFAULT_TEMPERATURES = (1.0, 1.5, 2.0, 2.27, 2.5, 3.0, 5.0)
DEFAULT_MAX_WORKERS = min(10, os.cpu_count() or 1)
INITIAL_STATES = ("random", "ordered")


def _energy_trajectory(job, *, L, J, max_sweeps, sample_interval):
    """Return one trajectory with its sweep-zero and terminal measurements."""
    temperature, state, seed = job
    model_seed, sampler_seed = seed.spawn(2)
    initialization = "up" if state == "ordered" else "random"
    model = IsingModel(L, J, initialization, seed=model_seed)
    sampler = MetropolisSampler(model, temperature, seed=sampler_seed)
    initial_energy = model.energy_per_spin
    history = SimulationRunner(sampler).run(
        max_sweeps=max_sweeps, sample_interval=sample_interval,
    )
    energies = [initial_energy, *history.energy_per_spin]
    if max_sweeps % sample_interval:
        energies.append(model.energy_per_spin)
    return energies


def compare_initial_states(
    temperatures=DEFAULT_TEMPERATURES, *, L=40, J=1.0, runs=8,
    max_sweeps=2_000, sample_interval=1, max_workers=DEFAULT_MAX_WORKERS, seed=42,
):
    """Return sweep numbers and {T: {state: energy array (runs, samples)}}.

    Every run starts measuring at sweep zero and runs through max_sweeps.
    SeedSequence creates independent streams reproducible across worker counts.
    """
    temperatures = tuple(float(T) for T in temperatures)
    if not temperatures or not all(math.isfinite(T) and T >= 0 for T in temperatures):
        raise ValueError("provide at least one finite, nonnegative temperature")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("temperatures must be distinct")
    L, runs, max_sweeps, sample_interval, max_workers = map(
        index, (L, runs, max_sweeps, sample_interval, max_workers),
    )
    if L < 2 or min(runs, max_sweeps, sample_interval, max_workers) < 1:
        raise ValueError("L must be at least 2 and run, sweep, interval, and worker counts positive")
    J = float(J)
    if not math.isfinite(J):
        raise ValueError("J must be finite")

    conditions = [
        (T, state) for T in temperatures for state in INITIAL_STATES for _ in range(runs)
    ]
    seeds = np.random.SeedSequence(seed).spawn(len(conditions))
    jobs = [(T, state, child_seed) for (T, state), child_seed in zip(conditions, seeds)]
    run = partial(
        _energy_trajectory, L=L, J=J, max_sweeps=max_sweeps, sample_interval=sample_interval,
    )

    if max_workers == 1:
        trajectories = list(map(run, jobs))
    else:
        # Populate the Numba cache once before spawned workers load the kernel.
        MetropolisSampler(IsingModel(2, initialization="up"), 1.0, seed=0).step()
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            trajectories = list(pool.map(run, jobs))

    sweeps = np.arange(0, max_sweeps + 1, sample_interval)
    if sweeps[-1] != max_sweeps:
        sweeps = np.append(sweeps, max_sweeps)
    energy_arrays = np.asarray(trajectories).reshape(len(temperatures), len(INITIAL_STATES), runs, -1)
    energies = {
        T: {state: energy_arrays[i, j] for j, state in enumerate(INITIAL_STATES)}
        for i, T in enumerate(temperatures)
    }
    return sweeps, energies


def _relaxation_sweep(sweeps, mean_difference, tolerance, window):
    """Return the start of the first two consecutive windows within tolerance."""
    # Average the signed difference before taking its absolute value.
    prefix = np.r_[0.0, np.cumsum(mean_difference)]
    rolling_gap = np.abs((prefix[window:] - prefix[:-window]) / window)
    within_tolerance = rolling_gap <= tolerance
    # Starts separated by window samples select two disjoint, adjacent windows.
    matching_starts = np.flatnonzero(within_tolerance[:-window] & within_tolerance[window:])
    if matching_starts.size:
        return int(sweeps[matching_starts[0]])
    return None


def main():
    """Run the comparison, print relaxation estimates, and render its plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--L", type=int, default=40, help="lattice side length")
    parser.add_argument("--J", type=float, default=1.0, help="coupling")
    parser.add_argument("--runs", type=int, default=8, help="runs per temperature and start")
    parser.add_argument("--max-sweeps", type=int, default=2_000)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tail-fraction", type=float, default=0.25, help="late-run sweep fraction")
    parser.add_argument(
        "--energy-tolerance", type=float, default=0.02, help="energy gap threshold per spin",
    )
    parser.add_argument(
        "--relaxation-window", type=int, default=10, help="samples per rolling energy average",
    )
    parser.add_argument("--output-dir", type=Path, help="save the three plots as PNG files")
    parser.add_argument("--no-show", action="store_true", help="skip interactive display")
    args = parser.parse_args()
    if not 0 < args.tail_fraction <= 1:
        parser.error("--tail-fraction must be in (0, 1]")
    if not math.isfinite(args.energy_tolerance) or args.energy_tolerance < 0:
        parser.error("--energy-tolerance must be finite and nonnegative")
    if args.relaxation_window < 1:
        parser.error("--relaxation-window must be a positive sample count")

    try:
        sweeps, energies = compare_initial_states(
            args.temperatures, L=args.L, J=args.J, runs=args.runs,
            max_sweeps=args.max_sweeps, sample_interval=args.sample_interval,
            max_workers=args.max_workers, seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    means = {
        T: {state: values.mean(axis=0) for state, values in states.items()}
        for T, states in energies.items()
    }
    deviations = {
        T: {state: values.std(axis=0) for state, values in states.items()}
        for T, states in energies.items()
    }
    differences = {T: states["random"] - states["ordered"] for T, states in means.items()}
    gaps = {T: np.abs(difference) for T, difference in differences.items()}
    tail_index = int(np.searchsorted(sweeps, args.max_sweeps * (1 - args.tail_fraction)))
    first_tail_sweep = sweeps[tail_index]
    late_energies = {
        T: {state: values[:, tail_index:] for state, values in states.items()}
        for T, states in energies.items()
    }

    print(
        f"Relaxation estimate (first two adjacent {args.relaxation_window}-sample "
        f"windows with mean energy gap <= {args.energy_tolerance:g})"
    )
    print(f"Late-run energy samples start at sweep {first_tail_sweep}.")
    for T, difference in differences.items():
        estimate = _relaxation_sweep(sweeps, difference, args.energy_tolerance, args.relaxation_window)
        label = f"sweep {estimate}" if estimate is not None else "threshold not sustained"
        print(f"T = {T:g}: {label}")

    import matplotlib.pyplot as plt

    from plotting import plot_energy_gap, plot_energy_relaxation, plot_late_energy_distribution

    figures = {
        "energy_relaxation": plot_energy_relaxation(sweeps, means, deviations),
        "energy_gap": plot_energy_gap(sweeps, gaps),
        "late_energy_distribution": plot_late_energy_distribution(late_energies, first_tail_sweep),
    }
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(args.output_dir / f"{name}.png", dpi=150)
        print(f"Plots saved to {args.output_dir}")
    if not args.no_show:
        plt.show()
    for figure in figures.values():
        plt.close(figure)


if __name__ == "__main__":
    main()
