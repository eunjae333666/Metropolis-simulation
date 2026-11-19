"""Matplotlib views of simulation results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class IsingPlotter:
    """Draw snapshots and diagnostics from existing measurements."""

    def plot_lattice(self, model, ax=None):
        """Draw +1 spins in black and -1 spins in white."""
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(
            model.spins, cmap=ListedColormap(["white", "black"]),
            vmin=-1, vmax=1, interpolation="nearest", aspect="equal",
        )
        ax.set_axis_off()
        return ax

    def _plot_history(self, history, values, ylabel, ax):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(history.sweeps, values)
        ax.set(xlabel="Monte Carlo sweeps", ylabel=ylabel)
        return ax

    def plot_energy(self, history, ax=None):
        """Plot energy per spin against sweep number."""
        return self._plot_history(history, history.energy_per_spin, r"$E/N$", ax)

    def plot_magnetization(self, history, ax=None):
        """Plot signed magnetization per spin."""
        return self._plot_history(history, history.magnetization_per_spin, r"$M/N$", ax)

    def plot_abs_magnetization(self, history, ax=None):
        """Plot absolute magnetization per spin."""
        return self._plot_history(
            history, history.abs_magnetization_per_spin, r"$|M|/N$", ax,
        )

    def plot_acceptance_rate(self, history, ax=None):
        """Plot the cumulative acceptance rate since burn-in."""
        ax = self._plot_history(history, history.acceptance_rate, "Acceptance rate", ax)
        ax.set_ylim(0, 1.05)
        return ax

    def plot_convergence(self, monitor, ax=None):
        """Plot the completed batch means for each monitored observable."""
        if ax is None:
            _, ax = plt.subplots()
        for name, batch_means in monitor.batch_history.items():
            ax.plot(range(1, len(batch_means) + 1), batch_means, label=name)
        ax.set(xlabel="Batch number", ylabel="Batch mean")
        ax.legend()
        return ax


def plot_energy_relaxation(sweeps, means, deviations):
    """Plot ensemble mean energy and one standard deviation at each temperature."""
    figure, axes = _temperature_panels(means, "Mean energy and one standard deviation")
    for (temperature, states), ax in zip(means.items(), axes):
        for state, mean in states.items():
            std = deviations[temperature][state]
            line, = ax.plot(sweeps, mean, label=state.capitalize())
            ax.fill_between(sweeps, mean - std, mean + std, color=line.get_color(), alpha=0.18)
        ax.set(xlabel="Monte Carlo sweeps", ylabel=r"$E/N$")
    axes[0].legend()
    return figure


def plot_energy_gap(sweeps, gaps):
    """Plot the absolute energy gap between ensemble means."""
    figure, ax = plt.subplots(layout="constrained")
    for temperature, gap in gaps.items():
        ax.plot(sweeps, gap, label=f"T = {temperature:g}")
    ax.set(xlabel="Monte Carlo sweeps", ylabel=r"$D_E$", ylim=(0, None))
    ax.legend()
    return figure


def plot_late_energy_distribution(energies, first_sweep):
    """Plot empirical CDFs of pooled late-run energies for each start."""
    figure, axes = _temperature_panels(
        energies, f"Late-run energy distributions from sweep {first_sweep:g}",
    )
    for (_, states), ax in zip(energies.items(), axes):
        pooled = np.concatenate([samples.ravel() for samples in states.values()])
        low, high = pooled.min(), pooled.max()
        margin = max(0.05 * (high - low), 0.005 * max(abs(low), abs(high), 1.0))
        for state, samples in states.items():
            values, counts = np.unique(samples, return_counts=True)
            cdf = counts.cumsum() / counts.sum()
            ax.step(
                np.r_[low - margin, values, high + margin], np.r_[0.0, cdf, 1.0],
                where="post", label=state.capitalize(),
            )
        ax.set(xlabel=r"$E/N$", ylabel="Empirical CDF", ylim=(-0.03, 1.03))
    axes[0].legend()
    return figure


def _temperature_panels(temperatures, title):
    columns = min(3, len(temperatures))
    rows = (len(temperatures) + columns - 1) // columns
    figure, grid = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.1 * rows),
        squeeze=False, layout="constrained",
    )
    figure.suptitle(title)
    axes = grid.ravel()
    for ax, temperature in zip(axes, temperatures):
        ax.set_title(f"T = {temperature:g}")
    for ax in axes[len(temperatures):]:
        ax.remove()
    return figure, axes[:len(temperatures)]
