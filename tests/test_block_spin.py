"""Tests for the majority rule and empirical reference matching."""

import numpy as np
import pytest

from analysis.block_spin import block_spin, match_reference_features, patch_features
from problems.problem_8_32 import run_experiment


@pytest.mark.parametrize("spin", [-1, 1])
def test_block_spin_preserves_uniform_order_and_reduces_shape(spin):
    spins = np.full((9, 9), spin)
    reduced = block_spin(spins)
    assert reduced.dtype == np.int8
    np.testing.assert_array_equal(reduced, np.full((3, 3), spin))
    np.testing.assert_array_equal(block_spin(reduced), [[spin]])


def test_block_spin_uses_local_majorities_without_changing_input():
    up = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.int8)
    down = -up
    spins = np.block([[up, down], [down, up]])
    original = spins.copy()
    spins.setflags(write=False)

    reduced = block_spin(spins)
    np.testing.assert_array_equal(reduced, [[1, -1], [-1, 1]])
    np.testing.assert_array_equal(spins, original)
    assert not np.shares_memory(reduced, spins)


@pytest.mark.parametrize("spins", [np.ones((3, 6)), np.ones((4, 4)), np.zeros((3, 3))])
def test_block_spin_validates_its_input(spins):
    with pytest.raises(ValueError):
        block_spin(spins)


def test_patch_features_use_short_axis_pairs_with_open_edges():
    patch = np.tile([1, 1, -1], (3, 1))
    np.testing.assert_allclose(patch_features(patch, 2), [1 / 3, 0.5, 0.0])
    np.testing.assert_allclose(patch_features(-patch, 2), patch_features(patch, 2))


def test_matching_scales_reference_features_and_ignores_constants():
    references = [
        [[-1.0, 0.0, 5.0], [1.0, 0.01, 5.0]],
        [[9.0, 0.2, 5.0], [11.0, 0.21, 5.0]],
    ]
    transformed = [[7.0, 0.015, 100.0], [10.0, 0.205, 0.0]]
    match = match_reference_features(transformed, references)
    np.testing.assert_array_equal(match.best_index, [0, 1])
    assert match.scale[-1] == 0
    assert match.scores[1, 1] == pytest.approx(0)


def test_matching_requires_at_least_one_varying_observable():
    with pytest.raises(ValueError, match="no variation"):
        match_reference_features(np.ones((1, 2)), np.ones((2, 3, 2)))


def test_experiment_records_blocked_features_from_the_saved_snapshots():
    result = run_experiment(
        [1.8, 3], [1, 3, 8], L=18, burn_in=4, samples=4, sample_spacing=2,
        runs=2, max_workers=1, seed=5, snapshots=2,
    )
    assert result.snapshots.shape == (2, 2, 18, 18)
    assert result.reference_patches.shape == (3, 6, 6)
    assert result.reference_features.shape == (3, 2, 4, 6)
    np.testing.assert_array_equal(result.snapshot_sweeps, [6, 12])
    assert np.isin(result.T_prime, [1, 3, 8]).all()
    for i, originals in enumerate(result.snapshots):
        for spins, sample in zip(originals, [0, 3], strict=True):
            actual = result.blocked_features[i, 0, sample]
            np.testing.assert_array_equal(actual, patch_features(block_spin(spins)))
