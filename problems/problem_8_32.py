"""Schroeder 8.32: visualize 3×3 block spins and estimate a matching temperature.

T' is a best match on a finite reference grid of measured observables.
An exact renormalized temperature requires analysis of the transformed interactions.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from operator import index
from pathlib import Path

import numpy as np

from analysis.block_spin import FeatureMatches, block_spin, match_reference_features, patch_features
from model import IsingModel
from sampler import MetropolisSampler

TEMPERATURES = (1.8, 2.0, 2.15, 2.269, 2.4, 2.6, 3.0)
REFERENCE_TEMPERATURES = (
    1.0, 1.3, 1.6, 1.9, 2.1, 2.2, 2.25, 2.269, 2.32, 2.4, 2.5, 2.65,
    2.8, 3.0, 3.4, 4.0, 5.0, 8.0, 12.0,
)
MAX_WORKERS = min(10, os.cpu_count() or 1)
CRITICAL_T_OVER_J = 2 / np.log1p(np.sqrt(2))


@dataclass
class _ChainSamples:
    patches: np.ndarray
    blocked: np.ndarray | None
    previews: np.ndarray | None


@dataclass
class BlockSpinExperiment:
    """Independent-run measurements and the empirical temperature mapping."""

    temperatures: np.ndarray
    reference_temperatures: np.ndarray
    patch_features: np.ndarray
    blocked_features: np.ndarray
    reference_features: np.ndarray
    snapshots: np.ndarray
    snapshot_sweeps: np.ndarray
    reference_patches: np.ndarray
    matches: FeatureMatches

    @property
    def T_prime(self):
        """Empirical best match, whose meaning is limited to the chosen observables."""
        return self.reference_temperatures[self.matches.best_index]

    @property
    def delta_T(self):
        return self.T_prime - self.temperatures


def _sample_chain(job, *, L, J, burn_in, samples, spacing, max_distance, snapshot_indices):
    temperature, is_reference, run, seed = job
    model_seed, sampler_seed, patch_seed = seed.spawn(3)
    patch_rng = np.random.default_rng(patch_seed)
    # Start below Tc in one ordered sector to avoid long-lived stripe starts.
    ordered = temperature / J < CRITICAL_T_OVER_J - 0.0005
    model = IsingModel(L, J, "up" if ordered else "random", seed=model_seed)
    sampler = MetropolisSampler(model, temperature, seed=sampler_seed)
    sampler.sweeps(burn_in)

    offsets = np.arange(L // 3)
    patches = np.empty((samples, max_distance + 1))
    blocked = None if is_reference else np.empty_like(patches)
    previews = []
    for sample in range(samples):
        sampler.sweeps(spacing)
        spins = model.spins
        # Uniform patch origins, including those that wrap on the parent torus.
        i, j = patch_rng.integers(0, L, size=2)
        patch = spins[np.ix_((i + offsets) % L, (j + offsets) % L)]
        patches[sample] = patch_features(patch, max_distance)
        if is_reference:
            if sample == samples // 2:
                previews.append(patch)
        else:
            blocked[sample] = patch_features(block_spin(spins), max_distance)
            if run == 0 and sample in snapshot_indices:
                previews.append(spins.copy())
    return _ChainSamples(patches, blocked, np.stack(previews) if previews else None)


def run_experiment(
    temperatures=TEMPERATURES, reference_temperatures=REFERENCE_TEMPERATURES, *,
    L=90, J=1.0, burn_in=3000, samples=80, sample_spacing=10, runs=3,
    max_workers=MAX_WORKERS, seed=42, snapshots=1,
):
    """Sample disjoint source/reference chains and match ensemble means."""
    L, burn_in, samples, sample_spacing, runs, max_workers, snapshots = map(
        index, (L, burn_in, samples, sample_spacing, runs, max_workers, snapshots),
    )
    temperatures = np.asarray(temperatures, dtype=float)
    reference_temperatures = np.asarray(reference_temperatures, dtype=float)
    for values, minimum in ((temperatures, 1), (reference_temperatures, 2)):
        if (
            values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all()
            or np.any(values < 0) or len(np.unique(values)) != len(values)
        ):
            raise ValueError("provide distinct, finite, nonnegative temperatures and at least two references")
    if L < 6 or L % 3:
        raise ValueError("L must be a multiple of three and at least six")
    if not np.isfinite(J) or J <= 0:
        raise ValueError("J must be positive and finite")
    if burn_in < 0 or min(samples, sample_spacing, runs, max_workers) < 1:
        raise ValueError("burn-in must be nonnegative and sample, interval, run and worker counts positive")
    if not 1 <= snapshots <= samples:
        raise ValueError("snapshots must be between one and the sample count")

    max_distance = min(5, L // 3 - 1)
    snapshot_indices = (
        np.array([samples // 2]) if snapshots == 1
        else np.linspace(0, samples - 1, snapshots, dtype=int)
    )
    jobs = [
        (T, is_reference, run)
        for grid, is_reference in ((temperatures, False), (reference_temperatures, True))
        for T in grid for run in range(runs)
    ]
    seeds = np.random.SeedSequence(seed).spawn(len(jobs))
    jobs = [(*job, child_seed) for job, child_seed in zip(jobs, seeds, strict=True)]
    sample_chain = partial(
        _sample_chain, L=L, J=J, burn_in=burn_in, samples=samples, spacing=sample_spacing,
        max_distance=max_distance, snapshot_indices=frozenset(snapshot_indices),
    )
    if max_workers == 1:
        all_chains = list(map(sample_chain, jobs))
    else:
        # Load the same JIT cache as the existing multiprocessing experiments.
        MetropolisSampler(IsingModel(2, initialization="up"), 1.0, seed=0).step()
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(jobs)), mp_context=get_context("spawn"),
        ) as pool:
            all_chains = list(pool.map(sample_chain, jobs))

    split = len(temperatures) * runs
    sources, references = all_chains[:split], all_chains[split:]
    shape = (len(temperatures), runs, samples, max_distance + 1)
    patches = np.stack([chain.patches for chain in sources]).reshape(shape)
    blocked = np.stack([chain.blocked for chain in sources]).reshape(shape)
    reference_shape = (len(reference_temperatures), runs, samples, max_distance + 1)
    reference_features = np.stack([chain.patches for chain in references]).reshape(reference_shape)
    matches = match_reference_features(
        blocked.mean(axis=(1, 2)),
        reference_features.reshape(len(reference_temperatures), runs * samples, max_distance + 1),
    )

    # Choose among saved crops from independent runs by distance to their own
    # reference mean, so the illustration shows a typical reference sample.
    reference_previews = np.stack([chain.previews[0] for chain in references])
    previews = reference_previews.reshape(len(reference_temperatures), runs, L // 3, L // 3)
    preview_features = reference_features[:, :, samples // 2]
    active = matches.scale > 0
    preview_offsets = preview_features[:, :, active] - matches.reference_means[:, None, active]
    typical = np.linalg.norm(preview_offsets / matches.scale[active], axis=-1).argmin(axis=1)
    reference_patches = previews[np.arange(len(reference_temperatures)), typical]

    return BlockSpinExperiment(
        temperatures, reference_temperatures, patches, blocked, reference_features,
        np.stack([chain.previews for chain in sources[::runs]]),
        burn_in + (snapshot_indices + 1) * sample_spacing, reference_patches, matches,
    )


def _save_data(output, result, args, tolerance):
    T = result.temperatures
    T_prime = result.T_prime
    delta = result.delta_T
    direction = np.where(delta < 0, "lower", np.where(delta > 0, "higher", "equal"))
    fixed = np.abs(delta) <= tolerance
    matched = result.matches.best_index
    score = result.matches.scores[np.arange(len(T)), matched]
    at_edge = (T_prime == result.reference_temperatures.min()) | (
        T_prime == result.reference_temperatures.max()
    )
    patch_m = result.patch_features[..., 0].mean(axis=(1, 2))
    blocked_m = result.blocked_features[..., 0].mean(axis=(1, 2))

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "T", "T_prime", "delta_T", "matching_score", "flow",
            "candidate_fixed_point", "reference_at_grid_edge", "patch_abs_m", "blocked_abs_m",
        ])
        writer.writerows(zip(T, T_prime, delta, score, direction, fixed, at_edge, patch_m, blocked_m))

    names = ["abs_m", *[f"C{r}" for r in range(1, len(result.matches.scale))]]
    np.savez_compressed(
        output / "block_spin_data.npz",
        T=T, T_prime=T_prime, delta_T=delta, matching_score=score, flow=direction,
        candidate_fixed_point=fixed, reference_at_grid_edge=at_edge,
        reference_temperatures=result.reference_temperatures, matched_reference_index=matched,
        feature_names=names, patch_features=result.patch_features, blocked_features=result.blocked_features,
        reference_features=result.reference_features, reference_means=result.matches.reference_means,
        feature_scale=result.matches.scale, scores_by_reference=result.matches.scores,
        original_snapshots=result.snapshots, snapshot_sweeps=result.snapshot_sweeps,
        reference_patches=result.reference_patches, L=args.L, J=args.J, seed=args.seed,
        burn_in=args.burn_in, sample_spacing=args.sample_spacing, block_levels=args.block_levels,
        fixed_point_tolerance=tolerance,
    )

    print("T' is an empirical best match defined by these observables and the finite reference grid.")
    for i, temperature in enumerate(T):
        tags = []
        if fixed[i]:
            tags.append("candidate fixed point")
        if at_edge[i]:
            tags.append("reference grid edge")
        detail = f" ({', '.join(tags)})" if tags else ""
        print(
            f"T={temperature:g} → T'={T_prime[i]:g}  ΔT={delta[i]:+.3f}  "
            f"score={score[i]:.3f}  |m| patch→block: {patch_m[i]:.3f}→{blocked_m[i]:.3f}{detail}"
        )

    critical = args.J * CRITICAL_T_OVER_J
    near = np.abs(T - critical) <= 0.0005 * args.J
    for label, selected, expected in (
        ("Below Tc", (T < critical) & ~near, delta < 0),
        ("Above Tc", (T > critical) & ~near, delta > 0),
    ):
        if selected.any():
            print(f"{label}: expected flow direction in {(selected & expected).sum()}/{selected.sum()} cases.")
    nearest = np.argmin(np.abs(T - critical))
    print(f"Closest input to Tc: T={T[nearest]:g}, ΔT={delta[nearest]:+.3f}.")
    print("Textbook flow: toward zero below Tc and infinity above Tc, with Tc separating them.")


def _plot_lattices(rows, labels, title):
    import matplotlib.pyplot as plt

    from plotting import IsingPlotter

    figure, axes = plt.subplots(
        len(rows), len(rows[0]), squeeze=False, layout="constrained",
        figsize=(2.6 * len(rows[0]), 2.3 * len(rows)),
    )
    plotter = IsingPlotter()
    for row, titles, row_axes in zip(rows, labels, axes, strict=True):
        for spins, name, ax in zip(row, titles, row_axes, strict=True):
            plotter.plot_spins(spins, ax=ax)
            ax.set_title(name)
    figure.suptitle(title)
    return figure


def _save_figures(output, result, args, tolerance):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def save(figure, filename):
        figure.savefig(output / filename, dpi=150)
        if args.no_show:
            plt.close(figure)

    for i, sweep in enumerate(result.snapshot_sweeps):
        rows = []
        labels = []
        for T, originals in zip(result.temperatures, result.snapshots, strict=True):
            sizes = [originals[i]]
            for _ in range(args.block_levels):
                sizes.append(block_spin(sizes[-1]))
            rows.append(sizes)
            labels.append([
                f"T = {T:g} · original ({args.L}×{args.L})",
                *[f"Block {n} ({len(spins)}×{len(spins)})" for n, spins in enumerate(sizes[1:], 1)],
            ])
        name = "block_spin_snapshots.png" if i == 0 else f"block_spin_snapshots_{i + 1:02}.png"
        save(_plot_lattices(rows, labels, f"Majority-rule block spins · sweep {sweep}"), name)

    rows = [
        (block_spin(snapshots[0]), result.reference_patches[j])
        for snapshots, j in zip(result.snapshots, result.matches.best_index, strict=True)
    ]
    labels = [
        (f"Blocked lattice · T = {T:g}", f"Reference patch · T' = {T_prime:g}")
        for T, T_prime in zip(result.temperatures, result.T_prime, strict=True)
    ]
    save(_plot_lattices(rows, labels, "Empirical temperature matches"), "reference_patch_comparison.png")

    order = np.argsort(result.temperatures)
    T, T_prime, delta = result.temperatures[order], result.T_prime[order], result.delta_T[order]
    critical = args.J * CRITICAL_T_OVER_J
    values = np.r_[T, T_prime, critical]
    low, high = values.min(), values.max()

    figure, ax = plt.subplots(layout="constrained")
    ax.plot([low, high], [low, high], "--", color="0.5", label=r"$T'=T$")
    ax.plot(T, T_prime, "o-", label="Best reference match")
    ax.axvline(critical, color="0.3", linestyle=":", label=r"$T_c$")
    ax.set(xlabel=r"Original temperature $T$", ylabel=r"Matching temperature $T'$")
    ax.legend()
    save(figure, "transformed_temperature.png")

    figure, ax = plt.subplots(layout="constrained")
    ax.axhline(0, color="0.5", linestyle="--")
    ax.axvline(critical, color="0.3", linestyle=":", label=r"$T_c$")
    ax.plot(T, delta, "o-")
    ax.set(xlabel=r"Original temperature $T$", ylabel=r"$\Delta T=T'-T$")
    ax.legend()
    save(figure, "delta_temperature.png")

    figure, ax = plt.subplots(layout="constrained", figsize=(7, max(3, 0.42 * len(T) + 1.3)))
    levels = np.arange(len(T))
    ax.scatter(T, levels, s=22, color="0.2", zorder=3)
    direction_colors = {"Lower T'": "tab:blue", "Higher T'": "tab:red"}
    directions = []
    for initial, destination, level in zip(T, T_prime, levels, strict=True):
        if destination != initial:
            label = "Lower T'" if destination < initial else "Higher T'"
            directions.append(label)
            ax.annotate(
                "", xy=(destination, level), xytext=(initial, level),
                arrowprops={"arrowstyle": "->", "color": direction_colors[label], "lw": 1.8},
            )
    critical_line = ax.axvline(critical, color="0.3", linestyle=":", label=r"$T_c$")
    legend = [
        Line2D([], [], color=color, label=label)
        for label, color in direction_colors.items() if label in directions
    ]
    fixed = np.abs(delta) <= tolerance
    if fixed.any():
        legend.append(ax.scatter(
            T[fixed], levels[fixed], s=75, facecolors="none", edgecolors="tab:green",
            label="Candidate fixed point", zorder=4,
        ))
    margin = 0.05 * max(high - low, 1)
    ax.set_xlim(low - margin, high + margin)
    ax.set_yticks(levels, [f"{t:g}" for t in T])
    ax.set(
        xlabel="Temperature along each arrow",
        ylabel=r"Original temperature $T$",
        title="Empirical block-spin flow",
    )
    ax.legend(handles=[*legend, critical_line], loc="lower center",
              bbox_to_anchor=(0.5, 1.12), ncol=2)
    save(figure, "rg_flow.png")
    if not args.no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=90, help="parent lattice side, divisible by three")
    parser.add_argument("--J", type=float, default=1.0, help="ferromagnetic coupling")
    parser.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    parser.add_argument("--reference-temperatures", type=float, nargs="+", default=REFERENCE_TEMPERATURES)
    parser.add_argument("--burn-in", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=80, help="measurements per chain")
    parser.add_argument("--sample-spacing", type=int, default=10, help="sweeps between measurements")
    parser.add_argument("--runs", type=int, default=3, help="independent chains per temperature and ensemble")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshots", type=int, default=1, help="recorded times in the original-lattice plots")
    parser.add_argument("--block-levels", type=int, default=1, help="successive reductions to visualize")
    parser.add_argument("--fixed-point-tolerance", type=float, help="default: 0.05 times J")
    parser.add_argument("--output-dir", type=Path, default=Path("results/problem_8_32"))
    parser.add_argument("--no-show", action="store_true", help="save without opening plot windows")
    args = parser.parse_args()

    tolerance = args.fixed_point_tolerance
    if tolerance is None:
        tolerance = 0.05 * args.J
    if not np.isfinite(tolerance) or tolerance < 0:
        parser.error("fixed-point tolerance must be finite and nonnegative")
    if args.block_levels < 1 or args.L < 6:
        parser.error("use at least one block level and L >= 6")
    side = args.L
    for _ in range(args.block_levels):
        if side % 3:
            parser.error("L must be divisible by three at each requested block level")
        side //= 3

    try:
        result = run_experiment(
            args.temperatures, args.reference_temperatures, L=args.L, J=args.J, burn_in=args.burn_in,
            samples=args.samples, sample_spacing=args.sample_spacing, runs=args.runs,
            max_workers=args.max_workers, seed=args.seed, snapshots=args.snapshots,
        )
    except ValueError as error:
        parser.error(str(error))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_data(args.output_dir, result, args, tolerance)
    _save_figures(args.output_dir, result, args, tolerance)
    print(f"Saved Problem 8.32 results to {args.output_dir}")


if __name__ == "__main__":
    main()
