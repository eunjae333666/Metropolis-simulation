"""Check periodic correlations, ensemble subtraction, and length estimates."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.correlation import (
    ONE_OVER_E, axis_pair_correlation, ensemble_correlations, estimate_empirical_correlation_length,
    fit_correlation_length, radial_pair_correlation,
)
from problems.problem_8_29 import ChainMeasurements, _save_data, _summarize_temperature
from problems.problem_8_29 import run_temperature_sweep


def test_ordered_lattice_has_unit_raw_and_zero_connected_correlation():
    pairs = axis_pair_correlation(np.ones((10, 10), dtype=np.int8))
    raw, connected, raw_sem, connected_sem = ensemble_correlations([pairs], [1.0])
    np.testing.assert_array_equal(raw, np.ones(6))
    np.testing.assert_array_equal(connected, np.zeros(6))
    assert np.isnan(raw_sem).all() and np.isnan(connected_sem).all()


def test_independent_spins_have_small_correlations_away_from_zero():
    rng = np.random.default_rng(7)
    spins = rng.choice([-1, 1], size=(160, 160))
    raw = axis_pair_correlation(spins)
    assert raw[0] == 1
    assert np.max(np.abs(raw[1:])) < 0.035


def test_periodic_axis_average_wraps_and_is_invariant_under_global_flip():
    spins = np.array([[1, 1, -1], [-1, 1, -1], [1, -1, -1]], dtype=np.int8)
    L = len(spins)
    raw = axis_pair_correlation(spins, [0, 1, L - 1, L])
    expected = sum(
        int(spins[i, j]) * (
            int(spins[(i + 1) % L, j]) + int(spins[i, (j + 1) % L])
        )
        for i in range(L) for j in range(L)
    ) / (2 * L ** 2)
    np.testing.assert_array_equal(raw, [1, expected, expected, 1])
    np.testing.assert_array_equal(raw, axis_pair_correlation(-spins, [0, 1, L - 1, L]))


def test_ensemble_subtracts_squared_mean_and_alignment_selects_one_sector():
    pairs = np.ones((2, 4))
    magnetizations = np.array([-1, 1])
    _, signed, _, _ = ensemble_correlations(pairs, magnetizations)
    _, aligned, _, _ = ensemble_correlations(pairs, np.abs(magnetizations))
    np.testing.assert_array_equal(signed, np.ones(4))
    np.testing.assert_array_equal(aligned, np.zeros(4))


def test_connected_error_accounts_for_covariance_across_chains():
    pairs = np.array([[1, 0.68], [1, 0.28], [1, 0.45]])
    magnetizations = np.array([0.78, 0.42, 0.64])
    _, _, _, connected_sem = ensemble_correlations(pairs, magnetizations)
    omitted = np.array([
        np.delete(pairs, i, axis=0).mean(axis=0) - np.delete(magnetizations, i).mean() ** 2
        for i in range(3)
    ])
    np.testing.assert_allclose(connected_sem, np.sqrt(2 / 3 * ((omitted - omitted.mean(0)) ** 2).sum(0)))


def test_length_fit_recovers_a_known_exponential_on_requested_distances():
    r = np.arange(11)
    fit = fit_correlation_length(r, 0.8 * np.exp(-r / 3.7), r_min=1, r_max=6)
    assert fit.status == "ok"
    assert fit.xi == pytest.approx(3.7)
    assert fit.amplitude == pytest.approx(0.8)
    assert fit.r_squared == pytest.approx(1)
    np.testing.assert_array_equal(r[fit.used], np.arange(1, 7))


def test_length_fit_stops_at_first_noisy_point_and_rejects_flat_curve():
    r = np.arange(11)
    exponential = np.exp(-r / 4)
    sem = np.zeros_like(exponential)
    sem[4] = 1
    no_signal = fit_correlation_length(r, exponential, standard_error=sem)
    np.testing.assert_array_equal(r[no_signal.used], [2, 3])
    assert np.isnan(no_signal.xi)
    assert np.isnan(fit_correlation_length(r, np.ones_like(r)).xi)


def test_short_sweep_samples_expected_distances_and_skips_critical_fit():
    r, sweeps, results = run_temperature_sweep(
        [0, 2.269], L=6, burn_in=2, samples=5, sample_spacing=3, runs=2,
        max_workers=1, lattice_snapshots=True,
    )
    np.testing.assert_array_equal(r, [0, 1, 2, 3])
    np.testing.assert_array_equal(sweeps, [5, 11, 17])
    ordered, critical = results
    assert ordered.sector == "aligned"
    np.testing.assert_array_equal(ordered.raw, np.ones(4))
    np.testing.assert_array_equal(ordered.connected, np.zeros(4))
    np.testing.assert_array_equal(ordered.chains[0].spins, np.ones((6, 6)))
    assert critical.sector == "signed"
    assert np.isnan(critical.fit.xi)
    assert "power-law" in critical.fit.status
    assert np.isfinite(critical.empirical.xi)
    assert np.isnan(ordered.empirical.xi)


@pytest.mark.parametrize("L", [4, 11])
def test_radial_shell_means_match_all_ordered_pairs(L):
    spins = np.random.default_rng(17).choice([-1, 1], size=(L, L))
    sums = {}
    counts = {}
    for i, j in np.ndindex(spins.shape):
        for a, b in np.ndindex(spins.shape):
            dx, dy = abs(i - a), abs(j - b)
            squared_distance = min(dx, L - dx) ** 2 + min(dy, L - dy) ** 2
            sums[squared_distance] = sums.get(squared_distance, 0) + int(spins[i, j] * spins[a, b])
            counts[squared_distance] = counts.get(squared_distance, 0) + 1

    distances, raw = radial_pair_correlation(spins)
    shells = sorted(sums)
    np.testing.assert_array_equal(distances, np.sqrt(shells))
    np.testing.assert_array_equal(raw, [sums[shell] / counts[shell] for shell in shells])


def test_radial_includes_diagonal_and_removes_fft_roundoff():
    checkerboard = (np.indices((6, 6)).sum(axis=0) % 2) * 2 - 1
    distances, raw = radial_pair_correlation(checkerboard)
    np.testing.assert_array_equal(distances[:3], [0, 1, np.sqrt(2)])
    np.testing.assert_array_equal(raw[:3], [1, -1, 1])
    _, ordered = radial_pair_correlation(np.ones((6, 6), dtype=np.int8))
    np.testing.assert_array_equal(ordered, np.ones_like(ordered))


def test_both_geometries_use_the_same_configurations_as_separate_modes():
    params = dict(
        L=8, burn_in=5, samples=5, sample_spacing=2, runs=2, max_workers=1, lattice_snapshots=True,
    )
    paired = run_temperature_sweep([2.6], **params, geometry="both")
    for geometry, (r, sweeps, results) in paired.items():
        separate_r, separate_sweeps, separate_results = run_temperature_sweep(
            [2.6], **params, geometry=geometry,
        )
        np.testing.assert_array_equal(r, separate_r)
        np.testing.assert_array_equal(sweeps, separate_sweeps)
        np.testing.assert_array_equal(results[0].raw, separate_results[0].raw)
        np.testing.assert_array_equal(results[0].connected, separate_results[0].connected)
        np.testing.assert_array_equal(
            results[0].chains[0].snapshot_correlations, separate_results[0].chains[0].snapshot_correlations,
        )

    axis = paired["axis"][2][0]
    r_radial, _, radial_results = paired["radial"]
    radial = radial_results[0]
    np.testing.assert_array_equal(axis.chains[0].spins, radial.chains[0].spins)
    np.testing.assert_array_equal(axis.raw[:2], radial.raw[:2])
    assert np.all(r_radial[radial.fit.used] <= params["L"] // 4)


def test_empirical_length_recovers_an_exponential():
    r = np.linspace(0, 12, 121)
    xi = 3.73
    result = estimate_empirical_correlation_length(r, 0.7 * np.exp(-r / xi))
    assert result.status == "ok"
    assert not result.lower_bound
    assert result.xi == pytest.approx(xi, abs=0.001)
    assert result.normalized[0] == 1


def test_empirical_length_interpolates_first_crossing_on_fractional_grid():
    r = [0, 1, np.sqrt(2), 2, np.sqrt(5)]
    result = estimate_empirical_correlation_length(r, [1, 0.6, 0.2, 0.5, 0.1])
    assert result.xi == pytest.approx(1 + (0.6 - ONE_OVER_E) / 0.4 * (np.sqrt(2) - 1))
    exact = estimate_empirical_correlation_length([0, 1, 2], [1, 0.8, ONE_OVER_E])
    assert exact.xi == pytest.approx(2)
    assert not exact.lower_bound


def test_empirical_length_reports_last_distance_as_lower_bound():
    r = [0, 1, np.sqrt(2)]
    result = estimate_empirical_correlation_length(r, [1, 0.6, 0.4])
    assert result.lower_bound
    assert result.xi == r[-1]
    assert result.status == "lower bound: no 1/e crossing within measured range"


@pytest.mark.parametrize("c0", [0, -0.1, np.nan, np.inf])
def test_empirical_length_handles_invalid_normalization(c0):
    result = estimate_empirical_correlation_length([0, 1], [c0, 0.1])
    assert np.isnan(result.xi)
    assert result.status.startswith("invalid c(0)")
    assert not result.lower_bound
    assert np.isnan(result.normalized).all()


def test_empirical_length_does_not_interpolate_across_nonfinite_measurement():
    r = np.arange(4)
    result = estimate_empirical_correlation_length(r, [1, np.nan, 0.2, 0.1])
    assert np.isnan(result.xi)
    assert "nonfinite" in result.status
    assert not result.lower_bound
    earlier_crossing = estimate_empirical_correlation_length(r, [1, 0.2, np.nan, 0.1])
    assert np.isfinite(earlier_crossing.xi)


def test_empirical_and_exponential_lengths_are_saved_independently(tmp_path):
    r = np.arange(9)
    curves = [np.r_[1.0, 0.8 * np.exp(-r[1:] / 2)], 1 / (1 + r / 8)]
    results = []
    for temperature, curve in zip([2.0, 2.269], curves, strict=True):
        chain = ChainMeasurements(
            pairs=curve, signed_m=0, absolute_m=0, snapshot_correlations=curve[None, :],
            spins=None,
        )
        results.append(_summarize_temperature(
            temperature, [chain], r, L=24, J=1, sector="signed", fit_r_min=2, fit_r_max=6,
        ))
    args = SimpleNamespace(runs=1, L=24, J=1, seed=42, burn_in=0, samples=1,
                           sample_spacing=1, lattice_snapshots=False)
    _save_data(tmp_path, r, [1], results, args, "axis")

    with (tmp_path / "summary.csv").open(newline="") as file:
        measured, critical = list(csv.DictReader(file))
    assert list(measured)[2:5] == ["xi_empirical", "xi_empirical_status", "xi_empirical_lower_bound"]
    assert float(measured["xi_empirical"]) < float(measured["xi_exp_pooled"])
    assert float(measured["xi_exp_pooled"]) == pytest.approx(2)
    assert critical["xi_empirical_lower_bound"] == "True"
    assert float(critical["xi_empirical"]) == r[-1]
    assert np.isnan(float(critical["xi_exp_pooled"]))
    assert "power-law" in critical["exp_fit_status"]

    with np.load(tmp_path / "correlation_data.npz") as data:
        np.testing.assert_array_equal(data["connected_normalized"], curves)
        np.testing.assert_array_equal(data["xi_empirical_lower_bound"], [False, True])
        assert data["xi_empirical"][1] == r[-1]
        assert np.isnan(data["xi_exp_pooled"][1])
        assert data["xi_exp_by_run"][0, 0] == pytest.approx(2)
