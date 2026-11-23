"""Majority-rule block spins and observables for comparing square patches."""

from dataclasses import dataclass
from operator import index

import numpy as np


def block_spin(spins):
    """Reduce a square ±1 lattice by a factor of three along each axis."""
    spins = np.asarray(spins)
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1] or spins.size == 0:
        raise ValueError("spins must be a nonempty square lattice")
    L = spins.shape[0]
    if L % 3:
        raise ValueError("the lattice side must be divisible by three")
    if not np.all((spins == 1) | (spins == -1)):
        raise ValueError("spins must be -1 or +1")

    side = L // 3
    totals = spins.reshape(side, 3, side, 3).sum(axis=(1, 3), dtype=np.int16)
    return np.sign(totals).astype(np.int8)


def patch_features(spins, max_distance=5):
    """Return |m| and raw axis correlations at separations 1 through max_distance.

    Both the blocked lattice and a reference patch are measured with open
    edges. Opposite sides of a crop are generally far apart in the parent
    lattice, so they should not be joined when measuring short-range pairs.
    """
    spins = np.asarray(spins)
    max_distance = index(max_distance)
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1] or spins.size == 0:
        raise ValueError("spins must be a nonempty square lattice")
    if not 1 <= max_distance < spins.shape[0]:
        raise ValueError("max_distance must be positive and smaller than the patch side")

    features = np.empty(max_distance + 1)
    features[0] = abs(spins.mean())
    for r in range(1, max_distance + 1):
        vertical = (spins[:-r, :] * spins[r:, :]).mean()
        horizontal = (spins[:, :-r] * spins[:, r:]).mean()
        features[r] = (vertical + horizontal) / 2
    return features


@dataclass
class FeatureMatches:
    """Reference centroids, scales, and distances for the empirical comparison."""

    reference_means: np.ndarray
    scale: np.ndarray
    scores: np.ndarray
    best_index: np.ndarray


def match_reference_features(transformed_means, reference_features):
    """Find the closest reference mean in units of the pooled feature spread.

    Reference samples are shaped (temperatures, samples, features). A feature
    that never varies across those samples does not contribute to the score.
    """
    transformed_means = np.asarray(transformed_means, dtype=float)
    reference_features = np.asarray(reference_features, dtype=float)
    if (
        reference_features.ndim != 3 or transformed_means.ndim != 2
        or not reference_features.size or not transformed_means.size
        or reference_features.shape[-1] != transformed_means.shape[-1]
    ):
        raise ValueError("supply nonempty reference samples and transformed means with matching features")
    if not (np.isfinite(reference_features).all() and np.isfinite(transformed_means).all()):
        raise ValueError("features must be finite")

    reference_means = reference_features.mean(axis=1)
    pooled = reference_features.reshape(-1, reference_features.shape[-1])
    varying = np.ptp(pooled, axis=0) > 0
    if not varying.any():
        raise ValueError("reference features have no variation; use a wider temperature grid")
    scale = np.zeros(reference_features.shape[-1])
    scale[varying] = pooled[:, varying].std(axis=0)
    difference = transformed_means[:, None, varying] - reference_means[None, :, varying]
    scores = np.linalg.norm(difference / scale[varying], axis=-1)
    return FeatureMatches(reference_means, scale, scores, scores.argmin(axis=1))
