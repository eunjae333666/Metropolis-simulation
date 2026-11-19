"""Check observables, periodic bonds, and measurement scheduling."""

import numpy as np
import pytest

from convergence import ConvergenceCriterion, ConvergenceMonitor
from model import IsingModel
from runner import SimulationRunner
from sampler import MetropolisSampler


@pytest.mark.parametrize("L", [2, 3, 9])
def test_incremental_energy_matches_full_recomputation(L):
    model = IsingModel(L, J=0.37, seed=42)
    sites = np.random.default_rng(7).integers(0, L, size=(200, 2))
    for i, j in sites:
        model.flip_spin(i, j)
        assert model.energy == model.calculate_energy()


def test_incremental_magnetization_matches_spin_sum():
    model = IsingModel(11, seed=2)
    assert model.spins.dtype == np.int8
    for i, j in np.random.default_rng(3).integers(0, model.L, size=(200, 2)):
        model.flip_spin(i, j)
        assert model.magnetization == model.spins.sum()


@pytest.mark.parametrize(("initialization", "sign"), [("up", 1), ("down", -1)])
def test_ordered_ground_state(initialization, sign):
    model = IsingModel(7, J=1.25, initialization=initialization)
    assert model.energy_per_spin == -2 * model.J
    assert model.magnetization_per_spin == sign
    assert model.abs_magnetization_per_spin == 1


def test_delta_energy_matches_the_full_energy_difference():
    model = IsingModel(7, J=0.8, initialization="up")
    model.flip_spin(3, 2)
    before = model.calculate_energy()
    delta = model.delta_energy(3, 3)
    model.flip_spin(3, 3)
    assert delta == pytest.approx(model.calculate_energy() - before)


def test_periodic_corner_has_neighbors_across_both_edges():
    model = IsingModel(5, J=1.5, initialization="up")
    assert model.delta_energy(0, 0) == 8 * model.J
    model.flip_spin(model.L - 1, 0)
    assert model.delta_energy(0, 0) == 4 * model.J
    model.flip_spin(0, model.L - 1)
    assert model.delta_energy(0, 0) == 0


def test_compiled_sweeps_match_reference_metropolis():
    model = IsingModel(7, J=0.37, seed=5)
    reference = IsingModel(7, J=0.37, seed=5)
    sampler = MetropolisSampler(model, T=1.3, seed=12)
    rng = np.random.default_rng(12)
    expected_acceptances = 0
    for _ in range(20 * model.N):
        i, j = rng.integers(0, model.L), rng.integers(0, model.L)
        delta = reference.delta_energy(i, j)
        if delta <= 0 or rng.random() < np.exp(-delta / sampler.T):
            reference.flip_spin(i, j)
            expected_acceptances += 1

    assert sampler.sweeps(20) == expected_acceptances
    assert sampler.accepted_flips == expected_acceptances
    assert sampler.attempted_flips == 20 * model.N
    assert sampler.acceptance_rate == expected_acceptances / sampler.attempted_flips
    np.testing.assert_array_equal(model.spins, reference.spins)
    assert model.energy == model.calculate_energy()
    assert model.magnetization == model.calculate_magnetization()


def test_runner_counts_only_measurement_attempts_and_finishes_partial_interval():
    model = IsingModel(5, initialization="up")
    sampler = MetropolisSampler(model, T=0, seed=8)
    assert sampler.step() is False
    assert sampler.sweep() == 0
    history = SimulationRunner(sampler).run(
        burn_in_sweeps=3, max_sweeps=5, sample_interval=2,
    )
    assert history.sweeps == [5, 7]
    assert history.sweeps_completed == 8
    assert history.energy_per_spin == [-2, -2]
    assert history.acceptance_rate == [0, 0]
    assert sampler.attempted_flips == 5 * model.N


def test_runner_stops_after_required_stable_batch_checks():
    model = IsingModel(5, initialization="up")
    sampler = MetropolisSampler(model, T=0, seed=8)
    monitor = ConvergenceMonitor(
        [
            ConvergenceCriterion(
                "energy", lambda model, sampler: model.energy_per_spin, rel_tol=1e-3,
            ),
            ConvergenceCriterion(
                "acceptance", lambda model, sampler: sampler.acceptance_rate,
                custom_check=lambda previous, recent: previous == recent == 0,
            ),
        ],
        batch_size=2, window_size=1, min_batches=3, stable_checks_required=2,
    )
    history = SimulationRunner(sampler).run(
        burn_in_sweeps=3, max_sweeps=20, sample_interval=2, monitor=monitor,
    )
    assert history.converged
    assert history.sweeps_completed == 3 + 4 * 2 * 2
    assert monitor.batch_history == {"energy": [-2] * 4, "acceptance": [0] * 4}
    monitor.reset()
    assert not monitor.converged
    assert monitor.batch_history == {"energy": [], "acceptance": []}
