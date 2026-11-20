"""Measure spatial correlations and their decay in the 2D Ising model."""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from multiprocessing import get_context
from operator import index
from pathlib import Path

import numpy as np

from analysis.correlation import (
    CorrelationFit, EmpiricalCorrelationLength, axis_pair_correlation, ensemble_correlations,
    estimate_empirical_correlation_length, fit_correlation_length, radial_pair_correlation,
)
from model import IsingModel
from sampler import MetropolisSampler

TEMPERATURES = (1.5, 2.0, 2.15, 2.25, 2.269, 2.35, 2.5, 3.0)
MAX_WORKERS = min(10, os.cpu_count() or 1)
CRITICAL_T_OVER_J = 2 / np.log1p(np.sqrt(2))


@dataclass
class ChainMeasurements:
    """Time averages and optional illustrations from one independent run."""

    pairs: np.ndarray
    signed_m: float
    absolute_m: float
    snapshot_correlations: np.ndarray | None
    spins: np.ndarray | None


@dataclass
class TemperatureResult:
    """Empirical lengths and exponential fit diagnostics at one temperature."""

    temperature: float
    sector: str
    chains: list[ChainMeasurements]
    raw: np.ndarray
    connected: np.ndarray
    raw_sem: np.ndarray
    connected_sem: np.ndarray
    empirical: EmpiricalCorrelationLength
    fit: CorrelationFit
    chain_fits: list[CorrelationFit]
    xi_exp_mean: float
    xi_exp_std: float
    valid_fits: int


def _critical_point(temperature, J):
    # Accept a rounded specification such as 2.269 when J=1.
    return abs(temperature / J - CRITICAL_T_OVER_J) <= 0.0005


def _measure_chain(
    job, *, L, J, burn_in, samples, sample_spacing, axis_r,
    snapshot_indices, lattice_snapshots, geometries,
):
    temperature, chain_index, seed = job
    model_seed, sampler_seed = seed.spawn(2)
    # Start in an ordered sector below Tc to avoid long-lived stripe states.
    ordered = temperature < J * CRITICAL_T_OVER_J and not _critical_point(temperature, J)
    model = IsingModel(L, J, "up" if ordered else "random", seed=model_seed)
    sampler = MetropolisSampler(model, temperature, seed=sampler_seed)
    sampler.sweeps(burn_in)
    distances = {"axis": axis_r}
    pair_sums = {}
    signed_m = 0.0
    absolute_m = 0.0
    snapshots = {mode: [] for mode in geometries}
    for sample in range(samples):
        sampler.sweeps(sample_spacing)
        spins = model.spins
        sample_pairs = {}
        if "axis" in geometries:
            sample_pairs["axis"] = axis_pair_correlation(spins, axis_r)
        if "radial" in geometries:
            distances["radial"], sample_pairs["radial"] = radial_pair_correlation(spins)

        m = model.magnetization_per_spin
        signed_m += m
        absolute_m += abs(m)
        for mode, pairs in sample_pairs.items():
            if sample == 0:
                pair_sums[mode] = pairs
            else:
                pair_sums[mode] += pairs
            if chain_index == 0 and sample in snapshot_indices:
                snapshots[mode].append(pairs - m ** 2)

    final_spins = model.spins.copy() if chain_index == 0 and lattice_snapshots else None
    return {
        mode: (
            distances[mode],
            ChainMeasurements(
                pair_sums[mode] / samples, signed_m / samples, absolute_m / samples,
                np.stack(snapshots[mode]) if snapshots[mode] else None, final_spins,
            ),
        )
        for mode in geometries
    }


def _length_fit(r, values, L, *, r_min, r_max, standard_error=None):
    if r_max is not None and r_max < r_min:
        return CorrelationFit(np.zeros(len(r), dtype=bool), "no distances inside fit bounds")
    fit = fit_correlation_length(r, values, r_min, r_max, standard_error=standard_error)
    if fit.status == "ok" and fit.xi >= L / 4:
        return replace(fit, xi=np.nan, status="decay length reaches L/4")
    return fit


def _summarize_temperature(
    temperature, chains, r, *, L, J, sector, fit_r_min, fit_r_max,
):
    critical = _critical_point(temperature, J)
    if sector == "auto":
        sector = "aligned" if temperature < J * CRITICAL_T_OVER_J and not critical else "signed"

    pair_means = np.stack([chain.pairs for chain in chains])
    # Global flips preserve s_i*s_j. Aligning configurations maps m to |m|,
    # so the ordered-phase subtraction estimates fluctuations in one sector.
    m = np.array([
        chain.absolute_m if sector == "aligned" else chain.signed_m for chain in chains
    ])
    raw, connected, raw_sem, connected_sem = ensemble_correlations(pair_means, m)
    empirical = estimate_empirical_correlation_length(r, connected)

    chain_fits = []
    if critical:
        fit = CorrelationFit(np.zeros(len(r), dtype=bool), "critical point: power-law decay")
    else:
        fit = _length_fit(
            r, connected, L, r_min=fit_r_min, r_max=fit_r_max, standard_error=connected_sem,
        )
        if fit.status == "ok":
            # Use the same signal window across chains. A chain with a shorter
            # positive range has no directly comparable estimate in that window.
            first, last = r[fit.used][[0, -1]]
            for chain_pairs, chain_m in zip(pair_means, m):
                chain_fit = _length_fit(
                    r, chain_pairs - chain_m ** 2, L, r_min=first, r_max=last,
                )
                if not np.array_equal(chain_fit.used, fit.used):
                    chain_fit = replace(chain_fit, xi=np.nan, status="signal window incomplete")
                chain_fits.append(chain_fit)

    accepted = np.array([item.xi for item in chain_fits if item.status == "ok"])
    return TemperatureResult(
        temperature, sector, chains, raw, connected, raw_sem, connected_sem,
        empirical, fit, chain_fits,
        float(accepted.mean()) if accepted.size else np.nan,
        float(accepted.std(ddof=1)) if accepted.size > 1 else np.nan,
        accepted.size,
    )


def run_temperature_sweep(
    temperatures=TEMPERATURES, *, L=40, J=1.0, burn_in=3000, samples=500,
    sample_spacing=10, runs=4, max_workers=MAX_WORKERS, seed=42, sector="auto",
    fit_r_min=2, fit_r_max=None, lattice_snapshots=False, geometry="axis",
):
    """Return distances, preview sweeps, and results per temperature.

    With geometry="both", return these triples in a dict keyed by geometry.
    Both geometries measure the same configurations in each chain.
    """
    L, burn_in, samples, sample_spacing, runs, max_workers = map(
        index, (L, burn_in, samples, sample_spacing, runs, max_workers),
    )
    if L < 2 or burn_in < 0 or min(samples, sample_spacing, runs, max_workers) < 1:
        raise ValueError("use L >= 2, nonnegative burn-in, and positive sample, run and worker counts")
    if not np.isfinite(J) or J <= 0:
        raise ValueError("J must be positive and finite for the ferromagnetic experiment")
    temperatures = tuple(float(T) for T in temperatures)
    if not temperatures or not all(np.isfinite(T) and T >= 0 for T in temperatures):
        raise ValueError("provide at least one finite, nonnegative temperature")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("temperatures must be distinct")
    if sector not in ("auto", "aligned", "signed"):
        raise ValueError("sector must be auto, aligned, or signed")
    if not np.isfinite(fit_r_min) or fit_r_min < 1 or (
        fit_r_max is not None
        and (not np.isfinite(fit_r_max) or not fit_r_min <= fit_r_max <= L // 2)
    ):
        raise ValueError("explicit fit bounds must satisfy 1 <= fit-r-min <= fit-r-max <= L//2")

    if geometry not in ("axis", "radial", "both"):
        raise ValueError("geometry must be axis, radial, or both")
    geometries = ("axis", "radial") if geometry == "both" else (geometry,)
    axis_r = np.arange(L // 2 + 1)
    snapshot_indices = np.linspace(0, samples - 1, min(3, samples), dtype=int)
    snapshot_sweeps = burn_in + (snapshot_indices + 1) * sample_spacing
    conditions = [(T, chain) for T in temperatures for chain in range(runs)]
    seeds = np.random.SeedSequence(seed).spawn(len(conditions))
    jobs = [(*condition, child_seed) for condition, child_seed in zip(conditions, seeds)]
    measure = partial(
        _measure_chain, L=L, J=J, burn_in=burn_in, samples=samples,
        sample_spacing=sample_spacing, axis_r=axis_r, snapshot_indices=frozenset(snapshot_indices),
        lattice_snapshots=lattice_snapshots, geometries=geometries,
    )
    workers = min(max_workers, len(jobs))
    if workers == 1:
        measurements = list(map(measure, jobs))
    else:
        # Match the existing comparison experiment: warm the Numba cache and
        # let each spawned process advance its own Markov chain.
        MetropolisSampler(IsingModel(2, initialization="up"), 1.0, seed=0).step()
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            measurements = list(pool.map(measure, jobs))

    sweeps = {}
    for mode in geometries:
        r = measurements[0][mode][0]
        chains = [measurement[mode][1] for measurement in measurements]
        # A radial grid extends past L/2. Keep the default fit cutoff at L//4.
        r_max = L // 4 if mode == "radial" and fit_r_max is None else fit_r_max
        results = [
            _summarize_temperature(
                T, chains[i * runs:(i + 1) * runs], r, L=L, J=J, sector=sector,
                fit_r_min=fit_r_min, fit_r_max=r_max,
            )
            for i, T in enumerate(temperatures)
        ]
        sweeps[mode] = (r, snapshot_sweeps, results)
    return sweeps if geometry == "both" else sweeps[geometry]


def _save_data(output_dir, r, snapshot_sweeps, results, args, geometry):
    runs = args.runs
    chain_xi = np.full((len(results), runs), np.nan)
    chain_r_squared = np.full_like(chain_xi, np.nan)
    for i, result in enumerate(results):
        if result.chain_fits:
            chain_xi[i] = [fit.xi for fit in result.chain_fits]
            chain_r_squared[i] = [fit.r_squared for fit in result.chain_fits]

    arrays = {
        "r": r,
        "temperature": [result.temperature for result in results],
        "sector": [result.sector for result in results],
        "L": args.L,
        "J": args.J,
        "seed": args.seed,
        "burn_in": args.burn_in,
        "samples": args.samples,
        "sample_spacing": args.sample_spacing,
        "raw": [result.raw for result in results],
        "raw_sem": [result.raw_sem for result in results],
        "connected": [result.connected for result in results],
        "connected_sem": [result.connected_sem for result in results],
        "connected_normalized": [result.empirical.normalized for result in results],
        "xi_empirical": [result.empirical.xi for result in results],
        "xi_empirical_status": [result.empirical.status for result in results],
        "xi_empirical_lower_bound": [result.empirical.lower_bound for result in results],
        "raw_by_run": [[chain.pairs for chain in result.chains] for result in results],
        "signed_m_by_run": [[chain.signed_m for chain in result.chains] for result in results],
        "absolute_m_by_run": [[chain.absolute_m for chain in result.chains] for result in results],
        "fit_used": [result.fit.used for result in results],
        "fit_status": [result.fit.status for result in results],
        "xi_exp_pooled": [result.fit.xi for result in results],
        "fit_quality": [result.fit.r_squared for result in results],
        "fit_amplitude": [result.fit.amplitude for result in results],
        "xi_exp_by_run": chain_xi,
        "fit_quality_by_run": chain_r_squared,
        "snapshot_sweeps": snapshot_sweeps,
        "snapshot_connected": [result.chains[0].snapshot_correlations for result in results],
    }
    if geometry == "radial":
        arrays["geometry"] = geometry
    if args.lattice_snapshots:
        arrays["final_spins"] = [result.chains[0].spins for result in results]
    np.savez_compressed(output_dir / "correlation_data.npz", **arrays)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "temperature", "sector", "xi_empirical", "xi_empirical_status",
            "xi_empirical_lower_bound", "xi_exp_mean", "xi_exp_std", "xi_exp_pooled",
            "fit_quality", "fit_r_min", "fit_r_max", "valid_runs", "total_runs", "exp_fit_status",
        ])
        for result in results:
            fit = result.fit
            distances = r[fit.used]
            writer.writerow([
                result.temperature, result.sector, result.empirical.xi, result.empirical.status,
                result.empirical.lower_bound, result.xi_exp_mean, result.xi_exp_std,
                fit.xi, fit.r_squared, distances[0] if len(distances) else "",
                distances[-1] if len(distances) else "", result.valid_fits, runs, fit.status,
            ])


def _save_results(output_dir, r, snapshot_sweeps, results, args, geometry):
    """Save the plots and data for one geometry."""
    from plotting import (
        plot_correlation_fits, plot_correlation_lengths, plot_lattice_snapshots,
        plot_normalized_correlations, plot_snapshot_correlations, plot_temperature_correlations,
    )

    critical_temperature = args.J * CRITICAL_T_OVER_J
    representative = {
        min(results, key=lambda result: result.temperature).temperature,
        min(results, key=lambda result: abs(result.temperature - critical_temperature)).temperature,
        max(results, key=lambda result: result.temperature).temperature,
    }
    examples = sorted(
        (result for result in results if result.temperature in representative),
        key=lambda result: result.temperature,
    )
    figures = {
        "correlation_vs_r": plot_temperature_correlations(
            r, results, raw=args.correlation == "raw", geometry=geometry,
        ),
        "correlation_semilog": plot_correlation_fits(r, results),
        "normalized_correlation_vs_r": plot_normalized_correlations(r, results),
        "correlation_length_vs_temperature": plot_correlation_lengths(results, critical_temperature),
        "snapshot_correlations": plot_snapshot_correlations(r, snapshot_sweeps, examples),
    }
    if args.correlation == "both":
        figures["raw_correlation_vs_r"] = plot_temperature_correlations(
            r, results, raw=True, geometry=geometry,
        )
    if args.lattice_snapshots and (args.geometry != "both" or geometry == "axis"):
        figures["lattice_snapshots"] = plot_lattice_snapshots(examples)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=150)
    _save_data(output_dir, r, snapshot_sweeps, results, args, geometry)
    return list(figures.values())


def main():
    """Run the temperature sweep and save data, fit diagnostics, and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", nargs="+", type=float, default=TEMPERATURES)
    parser.add_argument("--L", type=int, default=40, help="lattice side length")
    parser.add_argument("--J", type=float, default=1.0, help="ferromagnetic coupling")
    parser.add_argument("--burn-in", type=int, default=3000, help="initial sweeps to discard")
    parser.add_argument("--samples", type=int, default=500, help="measurements per chain")
    parser.add_argument("--sample-spacing", type=int, default=10, help="sweeps between measurements")
    parser.add_argument("--runs", type=int, default=4, help="independent chains per temperature")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sector", choices=("auto", "aligned", "signed"), default="auto")
    parser.add_argument("--fit-r-min", type=int, default=2, help="exponential fit minimum distance")
    parser.add_argument("--fit-r-max", type=int, help="exponential fit maximum distance (default: L//4)")
    parser.add_argument(
        "--geometry", choices=("axis", "radial", "both"), default="axis",
        help="pair displacements to average (default: axis)",
    )
    parser.add_argument("--correlation", choices=("connected", "raw", "both"), default="connected")
    parser.add_argument("--lattice-snapshots", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="save without opening plot windows")
    parser.add_argument("--output-dir", type=Path, default=Path("results/problem_8_29"))
    args = parser.parse_args()
    try:
        sweep = run_temperature_sweep(
            args.temperatures, L=args.L, J=args.J, burn_in=args.burn_in, samples=args.samples,
            sample_spacing=args.sample_spacing, runs=args.runs, max_workers=args.max_workers,
            seed=args.seed, sector=args.sector, fit_r_min=args.fit_r_min, fit_r_max=args.fit_r_max,
            lattice_snapshots=args.lattice_snapshots, geometry=args.geometry,
        )
    except ValueError as error:
        parser.error(str(error))

    import matplotlib.pyplot as plt

    from plotting import plot_geometry_comparison

    geometries = sweep if args.geometry == "both" else {args.geometry: sweep}
    figures = []
    critical_temperature = args.J * CRITICAL_T_OVER_J
    print(f"L = {args.L}, Tc = {critical_temperature:.5f}. Lengths are finite-lattice estimates.")
    for mode, (r, snapshot_sweeps, results) in geometries.items():
        output_dir = args.output_dir
        if args.geometry == "both" and mode == "radial":
            output_dir = output_dir / "radial"
        figures.extend(_save_results(output_dir, r, snapshot_sweeps, results, args, mode))
        if args.geometry != "axis":
            print(f"{mode.capitalize()} correlations")
        for result in results:
            empirical = result.empirical
            if empirical.lower_bound:
                measured = f"xi_empirical > {empirical.xi:.3g} ({empirical.status})"
            elif np.isfinite(empirical.xi):
                measured = f"xi_empirical = {empirical.xi:.3g}"
            else:
                measured = f"xi_empirical unavailable ({empirical.status})"
            fit = result.fit
            exponential = (
                f"xi_exp_pooled = {fit.xi:.3g}, R² = {fit.r_squared:.3f}, "
                f"{result.valid_fits}/{args.runs} valid chain fits"
                if fit.status == "ok" else f"exponential fit unavailable ({fit.status})"
            )
            print(f"T = {result.temperature:g} [{result.sector}]  {measured} | {exponential}")

    if args.geometry == "both":
        axis_r, _, axis_results = geometries["axis"]
        radial_r, _, radial_results = geometries["radial"]
        comparison = plot_geometry_comparison(
            axis_r, axis_results, radial_r, radial_results, raw=args.correlation == "raw",
        )
        comparison.savefig(args.output_dir / "geometry_comparison.png", dpi=150)
        figures.append(comparison)
        if args.correlation == "both":
            raw_comparison = plot_geometry_comparison(
                axis_r, axis_results, radial_r, radial_results, raw=True,
            )
            raw_comparison.savefig(args.output_dir / "raw_geometry_comparison.png", dpi=150)
            figures.append(raw_comparison)

    print(f"Saved plots and data to {args.output_dir}")
    if not args.no_show:
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
