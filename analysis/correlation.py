"""Spin correlations on a periodic square lattice."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def axis_pair_correlation(spins, r=None):
    """Average products over all origins and both axes at each distance.

    By default r contains the integer distances 0 through floor(L/2).
    Explicit integer shifts may exceed L/2, including r=L for periodicity checks.
    The lattice must contain spins -1 and +1.
    """
    spins = np.asarray(spins)
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1] or spins.size == 0:
        raise ValueError("spins must be a nonempty square lattice")
    if r is None:
        r = np.arange(spins.shape[0] // 2 + 1)
    else:
        r = np.asarray(r)
        if r.ndim != 1 or not np.issubdtype(r.dtype, np.integer) or np.any(r < 0):
            raise ValueError("r must be a one-dimensional array of nonnegative integers")

    correlation = np.empty(len(r))
    for k, distance in enumerate(r):
        vertical = np.sum(spins * np.roll(spins, -int(distance), axis=0), dtype=np.int64)
        horizontal = np.sum(spins * np.roll(spins, -int(distance), axis=1), dtype=np.int64)
        correlation[k] = (vertical + horizontal) / (2 * spins.size)
    return correlation


@lru_cache(maxsize=16)
def _radial_shells(L):
    offsets = np.arange(L)
    minimum_image = np.minimum(offsets, L - offsets)
    squared = minimum_image ** 2
    squared_distances, shells = np.unique(
        (squared[:, None] + squared[None, :]).ravel(), return_inverse=True,
    )
    return np.sqrt(squared_distances), shells, np.bincount(shells)


def radial_pair_correlation(spins):
    """Return sorted distances and raw pair correlations on a square torus.

    For a lattice of ±1 spins, average over all origins and every distinct
    periodic displacement in each minimum-image shell. Group displacements
    by integer squared distance before converting to distances.
    """
    spins = np.asarray(spins)
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1] or spins.size == 0:
        raise ValueError("spins must be a nonempty square lattice")

    distances, shells, counts = _radial_shells(spins.shape[0])
    # The inverse power spectrum gives the sum over all origins for each shift.
    spectrum = np.fft.rfft2(spins)
    products = np.fft.irfft2(np.abs(spectrum) ** 2, s=spins.shape, axes=(0, 1))
    # For ±1 spins these sums are integers. Remove FFT roundoff before binning.
    totals = np.bincount(shells, weights=np.rint(products).ravel())
    return distances.copy(), totals / (counts * spins.size)


def ensemble_correlations(pair_means, magnetization_means):
    """Combine equally sampled runs and estimate errors across chains.

    Compute mean(C_chain) - mean(m_chain)**2. To align magnetized sectors,
    callers must first average |m| within each chain and pass those means.
    Connected errors use a leave-one-chain-out jackknife so the covariance
    of C and m is included. Errors are NaN for a single chain.
    """
    pair_means = np.asarray(pair_means, dtype=float)
    magnetization_means = np.asarray(magnetization_means, dtype=float)
    if pair_means.ndim != 2 or len(pair_means) == 0:
        raise ValueError("pair_means must contain at least one chain")
    count = len(pair_means)
    if magnetization_means.shape != (count,):
        raise ValueError("magnetization_means must contain one value per chain")

    raw_mean = pair_means.mean(axis=0)
    connected_mean = raw_mean - magnetization_means.mean() ** 2
    if count == 1:
        unavailable = np.full_like(raw_mean, np.nan)
        return raw_mean, connected_mean, unavailable, unavailable.copy()

    raw_sem = pair_means.std(axis=0, ddof=1) / np.sqrt(count)
    leave_one_out_raw = (pair_means.sum(axis=0) - pair_means) / (count - 1)
    leave_one_out_m = (magnetization_means.sum() - magnetization_means) / (count - 1)
    jackknife = leave_one_out_raw - leave_one_out_m[:, None] ** 2
    centered = jackknife - jackknife.mean(axis=0)
    connected_sem = np.sqrt((count - 1) / count * np.sum(centered ** 2, axis=0))
    return raw_mean, connected_mean, raw_sem, connected_sem


ONE_OVER_E = 1 / np.e


@dataclass(frozen=True)
class EmpiricalCorrelationLength:
    """First 1/e crossing, or a lower bound at the last measured distance."""

    normalized: np.ndarray
    status: str
    xi: float = np.nan
    lower_bound: bool = False


def estimate_empirical_correlation_length(r, correlation):
    """Interpolate the first downward 1/e crossing of c(r)/c(0).

    Distances must start at zero and increase strictly. The two bracketing
    measurements must be finite and adjacent. A missing point before a
    crossing leaves it undetermined, so it cannot establish a full-range bound.
    """
    r = np.asarray(r, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    if (
        r.ndim != 1 or r.size == 0 or r.shape != correlation.shape or r[0] != 0
        or not np.all(np.isfinite(r)) or np.any(np.diff(r) <= 0)
    ):
        raise ValueError("r must start at zero and increase strictly, matching correlation")

    c0 = correlation[0]
    if not np.isfinite(c0) or c0 <= 0:
        return EmpiricalCorrelationLength(
            np.full_like(r, np.nan), "invalid c(0): expected a positive finite value",
        )

    with np.errstate(over="ignore", invalid="ignore"):
        normalized = correlation / c0
    finite = np.isfinite(normalized)
    normalized[~finite] = np.nan
    if r.size == 1:
        return EmpiricalCorrelationLength(normalized, "no positive distance measured")

    stops = np.flatnonzero(~finite | (normalized <= ONE_OVER_E))
    if stops.size == 0:
        return EmpiricalCorrelationLength(
            normalized, "lower bound: no 1/e crossing within measured range",
            xi=float(r[-1]), lower_bound=True,
        )

    right = stops[0]
    if not finite[right]:
        return EmpiricalCorrelationLength(normalized, "nonfinite correlation before 1/e crossing")
    left = right - 1
    fraction = (normalized[left] - ONE_OVER_E) / (normalized[left] - normalized[right])
    xi = r[left] + fraction * (r[right] - r[left])
    return EmpiricalCorrelationLength(normalized, "ok", xi=float(xi))


@dataclass
class CorrelationFit:
    """Exponential fit parameters, log-space R², and the points used."""

    used: np.ndarray
    status: str
    xi: float = np.nan
    amplitude: float = np.nan
    r_squared: float = np.nan


def fit_correlation_length(
    r, correlation, r_min=2, r_max=None, *, standard_error=None, min_r_squared=0.9,
):
    """Fit log(c) to a line on a contiguous, positive distance interval.

    With the default r=0…floor(L/2) grid, the default upper bound is L/4.
    Each point must be positive and finite and, when its standard error is
    available, exceed twice that error. Search contiguous windows within
    those constraints, favoring more points and then smaller distances.
    Require at least three points, a negative slope, and the requested R².
    """
    r = np.asarray(r, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    if (
        r.ndim != 1 or len(r) == 0 or r.shape != correlation.shape
        or not np.all(np.isfinite(r)) or r[0] < 0 or np.any(np.diff(r) <= 0)
    ):
        raise ValueError("r must be a strictly increasing finite distance grid matching correlation")
    if not np.isfinite(r_min) or r_min < 0 or (
        r_max is not None and (not np.isfinite(r_max) or r_max < r_min)
    ):
        raise ValueError("fit bounds must be finite and satisfy 0 <= r_min <= r_max")
    if not 0 <= min_r_squared <= 1:
        raise ValueError("min_r_squared must be between 0 and 1")
    if r_max is None:
        r_max = r[-1] / 2

    valid = np.isfinite(correlation) & (correlation > 0)
    if standard_error is not None:
        standard_error = np.asarray(standard_error, dtype=float)
        if standard_error.shape != r.shape or np.any(standard_error < 0):
            raise ValueError("standard_error must match r and be nonnegative")
        # NaN means uncertainty is unavailable, for example with a single chain.
        valid &= np.isnan(standard_error) | (correlation > 2 * standard_error)

    in_bounds = (r > 0) & (r >= r_min) & (r <= r_max)
    used = np.zeros_like(r, dtype=bool)
    if not in_bounds.any():
        return CorrelationFit(used, "no distances inside fit bounds")

    eligible = np.flatnonzero(in_bounds & valid)
    blocks = np.split(eligible, np.flatnonzero(np.diff(eligible) > 1) + 1)
    longest = max(blocks, key=len)
    if len(longest) < 3:
        used[longest] = True
        return CorrelationFit(used, "fewer than three usable distances")

    log_values = np.zeros_like(r)
    log_values[eligible] = np.log(correlation[eligible])
    best_rejected = longest
    best_r_squared = np.nan
    for count in range(len(longest), 2, -1):
        for block in blocks:
            for start in range(len(block) - count + 1):
                indices = block[start:start + count]
                x = r[indices]
                y = log_values[indices]
                centered_x = x - x.mean()
                centered_y = y - y.mean()
                slope = np.dot(centered_x, centered_y) / np.dot(centered_x, centered_x)
                if slope >= 0 or np.ptp(y) <= 1e-12:
                    continue
                residuals = centered_y - slope * centered_x
                r_squared = float(
                    1 - np.dot(residuals, residuals) / np.dot(centered_y, centered_y)
                )
                if r_squared >= min_r_squared:
                    used[indices] = True
                    return CorrelationFit(
                        used, "ok", xi=float(-1 / slope),
                        amplitude=float(np.exp(y.mean() - slope * x.mean())), r_squared=r_squared,
                    )
                if np.isnan(best_r_squared) or r_squared > best_r_squared:
                    best_rejected = indices
                    best_r_squared = r_squared

    used[best_rejected] = True
    status = "no decreasing signal" if np.isnan(best_r_squared) else "poor log-linear fit"
    return CorrelationFit(used, status, r_squared=best_r_squared)
