"""Matplotlib views of simulation results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class IsingPlotter:
    """Draw snapshots and diagnostics from existing measurements."""

    def plot_lattice(self, model, ax=None):
        """Draw the spin configuration of a model."""
        return self.plot_spins(model.spins, ax)

    def plot_spins(self, spins, ax=None):
        """Draw a lattice array with +1 in black and -1 in white."""
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(
            spins, cmap=ListedColormap(["white", "black"]),
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


def plot_temperature_correlations(r, results, *, raw=False):
    """Overlay temperature curves and shade one chain-based standard error."""
    figure, ax = plt.subplots(layout="constrained")
    for result in results:
        values = result.raw if raw else result.connected
        sem = result.raw_sem if raw else result.connected_sem
        line, = ax.plot(r, values, label=f"T = {result.temperature:g}")
        if np.any(np.isfinite(sem)):
            ax.fill_between(r, values - sem, values + sem, color=line.get_color(), alpha=0.15)
    ax.axhline(0, color="0.55", linewidth=0.8)
    ax.set(
        xlabel="Distance r (lattice spacings)", ylabel=r"$C(r)$" if raw else r"$c(r)$",
        title="Raw axis correlations" if raw else "Connected axis correlations",
    )
    ax.legend(ncol=2 if len(results) > 5 else 1)
    return figure


def plot_correlation_fits(r, results):
    """Show the positive connected signal and the points in accepted fits."""
    figure, ax = plt.subplots(layout="constrained")
    for result in results:
        positive = (r > 0) & (result.connected > 0)
        log_values = np.full(r.shape, np.nan)
        log_values[positive] = np.log(result.connected[positive])
        line, = ax.plot(r, log_values, label=f"T = {result.temperature:g}", alpha=0.65)
        fit = result.fit
        if fit.status == "ok":
            x = r[fit.used]
            color = line.get_color()
            ax.scatter(x, log_values[fit.used], color=color, s=14)
            ax.plot(x, np.log(fit.amplitude) - x / fit.xi, "--", color=color)
    ax.set(
        xlabel="Distance r (lattice spacings)", ylabel=r"$\ln c(r)$",
        title="Connected correlations · points and dashed lines mark accepted fits",
    )
    ax.legend(ncol=2 if len(results) > 5 else 1)
    return figure


def plot_correlation_lengths(results, critical_temperature):
    """Plot pooled and independent-chain exponential length estimates."""
    figure, ax = plt.subplots(layout="constrained")
    ordered = sorted(results, key=lambda result: result.temperature)
    temperatures = np.array([result.temperature for result in ordered])
    pooled = np.array([result.fit.xi for result in ordered])
    means = np.array([result.xi_mean for result in ordered])
    stds = np.array([result.xi_std for result in ordered])
    ax.plot(temperatures, pooled, "o-", label="Pooled estimate")
    valid_means = np.isfinite(means)
    if valid_means.any():
        line, = ax.plot(temperatures[valid_means], means[valid_means], "s", label="Chain mean ± SD")
        with_spread = valid_means & np.isfinite(stds)
        if with_spread.any():
            ax.errorbar(
                temperatures[with_spread], means[with_spread], yerr=stds[with_spread],
                fmt="none", capsize=3, color=line.get_color(),
            )
    unavailable = temperatures[~np.isfinite(pooled)]
    if unavailable.size:
        ax.plot(
            unavailable, np.full_like(unavailable, 0.04), marker="x", linestyle="none",
            transform=ax.get_xaxis_transform(), color="0.45", label="No reported fit",
        )
    ax.axvline(critical_temperature, color="0.35", linestyle=":", label=r"$T_c$")
    ax.set(
        xlabel="Temperature T", ylabel=r"$\xi$ (lattice spacings)",
        title="Exponential correlation length on a finite lattice",
    )
    ax.legend()
    return figure


def plot_snapshot_correlations(r, sweeps, results):
    """Plot single-configuration connected correlations at sampled sweeps."""
    figure, axes = plt.subplots(
        len(results), len(sweeps), sharex=True, sharey="row", squeeze=False,
        figsize=(3.4 * len(sweeps), 2.3 * len(results)), layout="constrained",
    )
    for row, result in enumerate(results):
        for column, sweep in enumerate(sweeps):
            ax = axes[row, column]
            values = result.chains[0].snapshot_correlations[column]
            ax.bar(r[1:], values[1:], width=0.85)
            ax.axhline(0, color="0.55", linewidth=0.8)
            if row == 0:
                ax.set_title(f"Sweep {sweep}")
            if column == 0:
                ax.set_ylabel(f"T = {result.temperature:g}\n" + r"$c_{\rm snapshot}(r)$")
    figure.supxlabel("Distance r (lattice spacings)")
    figure.suptitle("Single-configuration correlations")
    return figure


def plot_lattice_snapshots(results):
    """Show representative final sampled configurations using IsingPlotter."""
    plotter = IsingPlotter()
    figure, axes = plt.subplots(
        1, len(results), figsize=(3 * len(results), 3.2), squeeze=False, layout="constrained",
    )
    for ax, result in zip(axes[0], results):
        plotter.plot_spins(result.chains[0].spins, ax=ax)
        ax.set_title(f"T = {result.temperature:g}")
    return figure
